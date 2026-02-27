# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Examples for Qwen3ASRModel (Transformers backend).

Covers:
  - single-sample inference (URL audio)
  - batch inference (mixed URL / base64 / (np.ndarray, sr))
  - forcing language (text-only output)
  - returning time_stamps (single + batch) via Qwen3ForcedAligner

对 tgt wav 跑 ASR 并保存 transcript + 词级时间戳（供后续算 latency）:
  python example_qwen3_asr_transformers.py --manifest data/output_qwen_livetranslate/manifest.jsonl
  输出: out_dir（默认 data/output_qwen_asr）下 {basename}_asr.json。建议在项目根目录运行。
"""

import argparse
import base64
import io
import json
import os
import urllib.request
from typing import Any, List, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel

'''
python qwen3_asr.py --manifest data/output_qwen_livetranslate/manifest.jsonl --out_dir data/output_qwen_asr
'''

ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

URL_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"


def _download_audio_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _to_data_url_base64(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _print_result(title: str, results) -> None:
    print(f"\n===== {title} =====")
    for i, r in enumerate(results):
        print(f"[sample {i}] language={r.language!r}")
        print(f"[sample {i}] text={r.text!r}")
        if r.time_stamps is not None and len(r.time_stamps) > 0:
            head = r.time_stamps[0]
            tail = r.time_stamps[-1]
            print(f"[sample {i}] ts_first: {head.text!r} {head.start_time}->{head.end_time} s")
            print(f"[sample {i}] ts_last : {tail.text!r} {tail.start_time}->{tail.end_time} s")


def test_single_url(asr: Qwen3ASRModel) -> None:
    results = asr.transcribe(
        audio=URL_ZH,
        language=None,
        return_time_stamps=False,
    )
    assert isinstance(results, list) and len(results) == 1
    _print_result("single-url (no forced language, no timestamps)", results)


def test_batch_mixed(asr: Qwen3ASRModel) -> None:
    zh_bytes = _download_audio_bytes(URL_ZH)
    en_bytes = _download_audio_bytes(URL_EN)

    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")
    en_wav, en_sr = _read_wav_from_bytes(en_bytes)

    results = asr.transcribe(
        audio=[URL_ZH, zh_b64, (en_wav, en_sr)],
        context=["", "交易 停滞", ""],
        language=[None, "Chinese", "English"],
        return_time_stamps=False,
    )
    assert len(results) == 3
    _print_result("batch-mixed (forced language for some)", results)


def test_single_with_timestamps(asr: Qwen3ASRModel) -> None:
    results = asr.transcribe(
        audio=URL_EN,
        language="English",
        return_time_stamps=True,
    )
    assert len(results) == 1
    assert results[0].time_stamps is not None
    _print_result("single-url (forced language + timestamps)", results)


def test_batch_with_timestamps(asr: Qwen3ASRModel) -> None:
    zh_bytes = _download_audio_bytes(URL_ZH)
    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")

    results = asr.transcribe(
        audio=[URL_ZH, zh_b64, URL_EN],
        context=["", "交易 停滞", ""],
        language=["Chinese", "Chinese", "English"],
        return_time_stamps=True,
    )
    assert len(results) == 3
    assert all(r.time_stamps is not None for r in results)
    _print_result("batch (forced language + timestamps)", results)


# ===========================================================\
# 在还没考虑 timeline、会话时间之前，先把 ASR 直接给我们的东西整理成一个要保存的字典
def _time_stamps_to_dicts(time_stamps: Any) -> List[dict]:
    """将 ASR 返回的 time_stamps 转为可 JSON 序列化的列表。"""
    if time_stamps is None:
        return []
    out = []
    for ts in time_stamps:
        out.append({
            "text": getattr(ts, "text", ""),
            "start_time": getattr(ts, "start_time", 0.0),
            "end_time": getattr(ts, "end_time", 0.0),
        })
    return out


# ⭐ 转换到真实时间 ！！！
def _wav_time_to_session_sec(wav_sec: float, timeline: list, t0: float) -> float:
    """把「无 gap 的 wav 内时间」映射为「考虑 gap 的会话相对时间」(秒)。

    t0 的两种取法：
    - 若 t0 = 第一个 tgt 的 receive_timestamp：表示「相对第一个译文 chunk 到达」的秒数。
    - 若 t0 = first_send_timestamp（src 开始发送时间）：表示「相对源语开始」的秒数。

    例：tgt wav 由多段首尾拼接，无静音。
    timeline 片段:
      [0] offset_sec=0,    duration_sec=0.32, receive_timestamp=100.0
      [1] offset_sec=0.32, duration_sec=0.32, receive_timestamp=100.1
      [2] offset_sec=0.64, duration_sec=0.08, receive_timestamp=100.2
      [3] offset_sec=0.72, duration_sec=0.32, receive_timestamp=105.0   # 中间有约 4.8s gap
    如果 t0 = 98.0:
    - wav_sec=0.16 → 落在片段 0 [o0, o0+d0] → session = (100.0-98.0) + (0.16-0) = 0.16
    - wav_sec=0.50 → 落在片段 1 [o1, o1+d1] → session = (100.1-98.0) + (0.50-0.32) = 0.1 + 0.18 = 0.28
    - wav_sec=0.80 → 落在片段 3 [o3, o3+d3] → session = (105.0-98.0) + (0.80-0.72) = 5.0 + 0.08 = 5.08

    - wav_sec → 落在片段 [oi, oi + di]，因为 oi <= wav_sec < oi + di
              → session = (ri - t0) + (wav_sec - oi)
    """
    for t in timeline:
        o = t["offset_sec"]
        d = t["duration_sec"]
        if o <= wav_sec < o + d:
            return (t["receive_timestamp"] - t0) + (wav_sec - o)
    return (timeline[-1]["receive_timestamp"] - t0) + max(0, wav_sec - (timeline[-1]["offset_sec"] + timeline[-1]["duration_sec"]))


# 根据 manifest.jsonl 里列出的每条「译文 wav」，对每个 tgt wav 跑一次 Qwen3 ASR（带词级时间戳）
# 并把转写结果和两种时间戳写到对应的 .json
def run_tgt_asr_from_manifest(
    manifest_path: str,
    asr: Qwen3ASRModel,
    tgt_language: str = "Chinese",
    use_timeline_for_gap: bool = True,
    t0_first_receive: bool = False,
    out_dir: str = "data/output_qwen_asr",
) -> None:
    """
    读 manifest.jsonl，对每条 tgt wav 跑 ASR（带时间戳），写出 out_dir/{basename}_asr.json
    time_stamps：相对 wav 起点的连续时间（无 gap）
    
    若有 tgt_timeline 且 use_timeline_for_gap：
    额外写出 time_stamps_session_sec（会话相对时间）；t0 优先 first_send_timestamp，否则首段 receive
    """
    # 1. 读 manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. 对每个 tgt wav 跑 ASR
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(lines)}] 跳过：无效 JSON - {e}")
            continue
        tgt_wav = rec.get("tgt")
        if not tgt_wav:
            print(f"[{i+1}/{len(lines)}] 跳过：无 tgt 路径")
            continue
        tgt_wav = os.path.normpath(tgt_wav)
        if not os.path.isfile(tgt_wav):
            print(f"[{i+1}/{len(lines)}] 跳过：文件不存在 {tgt_wav}")
            continue

        print(f"[{i+1}/{len(lines)}] ASR: {tgt_wav}")

        # 用 asr.transcribe 对这一段 tgt 音频做 ASR，得到一条结果（含整段 text 和词级 time_stamps）
        results = asr.transcribe(
            audio=tgt_wav,
            language=tgt_language,
            return_time_stamps=True,
        )
        # 一条音频 一个结果
        if not results or len(results) != 1:
            print(f"  警告：未得到单条结果，跳过")
            continue

        r = results[0]
        ts_list = _time_stamps_to_dicts(r.time_stamps)
        out = {
            "src": rec.get("src", ""),
            "tgt": tgt_wav,
            "tgt_timeline": rec.get("tgt_timeline", ""),
            "text": r.text or "",
            "time_stamps": ts_list,
            "time_reference": "wav_relative_no_gap",
        }
        # 4. 可选：考虑 gap 的会话时间
        if use_timeline_for_gap:
            tgt_timeline_path = rec.get("tgt_timeline")
            if tgt_timeline_path and os.path.isfile(os.path.normpath(tgt_timeline_path)):
                with open(os.path.normpath(tgt_timeline_path), "r", encoding="utf-8") as f:
                    tl_data = json.load(f)
                timeline = tl_data.get("timeline") or []
                if timeline:
                    # t0：默认优先 src 开始（first_send_timestamp），否则或指定 --t0_first_receive 则用第一个 tgt receive
                    if t0_first_receive:
                        t0 = timeline[0]["receive_timestamp"]
                        t0_ref = "first_receive_timestamp"
                    else:
                        t0_send = tl_data.get("first_send_timestamp")
                        if t0_send is not None:
                            t0 = float(t0_send)
                            t0_ref = "first_send_timestamp"
                        else:
                            t0 = timeline[0]["receive_timestamp"]
                            t0_ref = "first_receive_timestamp"
                    session_sec = [
                        {
                            "text": t["text"],
                            "start_time": _wav_time_to_session_sec(t["start_time"], timeline, t0),
                            "end_time": _wav_time_to_session_sec(t["end_time"], timeline, t0),
                        }
                        for t in ts_list
                    ]
                    out["time_stamps_session_sec"] = session_sec
                    out["t0_reference"] = t0_ref
                    out["time_reference"] = "wav_relative_no_gap; time_stamps_session_sec considers gap via tgt_timeline (t0=%s)" % t0_ref
            else:
                out["_note"] = "tgt_timeline missing or not file; only wav-relative timestamps saved (no gap)."
        out_dir = os.path.normpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
        out_path = os.path.join(out_dir, base_name + "_asr.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {out_path}")
# ===========================================================/


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 ASR 示例 / 对 manifest 中的 tgt wav 跑 ASR 并保存 transcript+timestamp")
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="output_qwen_livetranslate 的 manifest.jsonl 路径；指定后只跑 manifest 流程并退出",
    )
    parser.add_argument(
        "--tgt_language",
        type=str,
        default="Chinese",
        help="manifest 模式下 tgt 音频的语言，如 Chinese / English（默认 Chinese）",
    )
    parser.add_argument(
        "--no_timeline_gap",
        action="store_true",
        help="manifest 模式下不根据 tgt_timeline 计算考虑 gap 的会话时间，只输出 wav 相对时间戳",
    )
    parser.add_argument(
        "--t0_first_receive",
        action="store_true",
        help="会话时间以第一个 tgt receive 为 0；默认是 first_send_timestamp（src 开始）为 0（若 timeline 里有）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/output_qwen_asr",
        help="ASR 结果输出目录（默认 data/output_qwen_asr）",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="仅跑内置示例（单条/批量、带/不带时间戳），不读 manifest",
    )
    args = parser.parse_args()

    asr = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    if args.manifest:
        run_tgt_asr_from_manifest(
            args.manifest,
            asr,
            tgt_language=args.tgt_language,
            use_timeline_for_gap=not args.no_timeline_gap,
            t0_first_receive=args.t0_first_receive,
            out_dir=args.out_dir,
        )
        return
    # ---------------------------------------------------------------
    # 不用管 ！
    if args.test:
        test_single_url(asr)
        test_batch_mixed(asr)
        test_single_with_timestamps(asr)
        test_batch_with_timestamps(asr)
        return
    # 默认：跑内置示例
    test_single_url(asr)
    test_batch_mixed(asr)
    test_single_with_timestamps(asr)
    test_batch_with_timestamps(asr)


if __name__ == "__main__":
    main()
