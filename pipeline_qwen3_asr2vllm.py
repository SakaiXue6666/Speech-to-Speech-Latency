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
Qwen3 ASR 的 vLLM 后端：更高吞吐、适合大批量。

使用前安装: pip install -U qwen-asr[vllm]
（需 timestamps 时建议再装: pip install -U flash-attn --no-build-isolation，给 ForcedAligner 加速）

对 tgt wav 跑 ASR 并保存 transcript + 词级时间戳:
  python pipeline_qwen3_asr2vllm.py --manifest data/.../manifest.jsonl --out_dir data/.../output_qwen_asr --tgt_language Chinese
  python pipeline_qwen3_asr2vllm.py --manifest data/.../manifest.jsonl --out_dir data/.../output_qwen_asr --tgt_language Chinese --batch_size 32

注意：vLLM 要求主逻辑在 if __name__ == '__main__' 下执行，避免 spawn 报错。
"""

import argparse
import base64
import gc
import io
import json
import os
import urllib.request
from typing import Any, List, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel

import types

'''
conda activate s2s_latency
python qwen3_asr.py --manifest data/output_qwen_livetranslate3/manifest.jsonl --out_dir data/output_qwen_asr3
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
# 根据 manifest.jsonl 里列出的每条「译文 wav」，对每个 tgt wav 跑一次 Qwen3 ASR（带词级时间戳）
# 并把转写结果和两种时间戳写到对应的 .json
def run_tgt_asr_from_manifest(
    manifest_path: str,
    asr: Qwen3ASRModel,
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
) -> None:
    """
    读 manifest.jsonl，对每条 tgt wav 跑 ASR（带时间戳），写出 out_dir/{basename}_asr.json
    time_stamps：相对 wav 起点的连续时间

    长音频说明：Qwen3 ASR 内部按 chunk 处理，结果有时会少最后几秒（最后一 chunk 未返回）。
    若出现「ASR 比 wav 短约 Xs」的提示，可查阅 qwen_asr 是否支持 chunk_length_s / max_duration 等参数，
    或考虑将长音频先按静音切分为多段再分别识别后拼接。
    """
    # 1. 读 manifest
    # jsonl:
    # - "src": <path of srcwav>
    # - "tgt": <path of tgt wav>
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. 对每个 tgt wav 跑 ASR
    for i, line in enumerate(lines):
        # 加载 jsonl
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(lines)}] 跳过：无效 JSON - {e}")
            continue

        # tgt wav 路径
        tgt_wav = rec.get("tgt")
        if not tgt_wav:
            print(f"[{i+1}/{len(lines)}] 跳过：无 tgt 路径")
            continue
        tgt_wav = os.path.normpath(tgt_wav)
        if not os.path.isfile(tgt_wav):
            print(f"[{i+1}/{len(lines)}] 跳过：文件不存在 {tgt_wav}")
            continue

        print(f"[{i+1}/{len(lines)}] ASR: {tgt_wav}")

        # 3. 用 asr.transcribe 对这一段 tgt 音频做 ASR，得到一条结果（含整段 text 和词级 time_stamps）
        # ================================================\
        # 注意：Qwen3 ASR 内部按 chunk 处理长音频（如 2s 流式窗口），尾部可能少几秒未返回，属模型/库行为。
        results = asr.transcribe(
            audio=tgt_wav,
            language=tgt_language,
            return_time_stamps=True,
        )
        # 一条音频 一个结果
        if not results or len(results) != 1:
            print(f"  警告：未得到单条结果，跳过")
            continue

        r = results[0]  # 一条音频 一个结果

        # 在还没考虑 timeline、会话时间之前，先把 ASR 直接给我们的东西整理成一个要保存的字典
        if r.time_stamps is None:
            ts_list = []
        else:
            ts_list = [
                {
                    "text": getattr(ts, "text", ""),
                    "start_time": getattr(ts, "start_time", 0.0),
                    "end_time": getattr(ts, "end_time", 0.0),
                }
                for ts in r.time_stamps
            ]
        # ================================================/
        
        out = {
            "src": rec.get("src", ""),
            "tgt": tgt_wav,
            "text": r.text or "",
            "time_stamps": ts_list,
        }
       
        # 👈 prediction_length 用最后一个 token 的结束时间
        ts_for_len =  out.get("time_stamps") or []
        if ts_for_len: last_end = ts_for_len[-1]["end_time"]
        else: last_end = 0.0
        out["prediction_length"] = round(last_end, 2)

        # 5. 保存结果
        out_dir = os.path.normpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
        out_path = os.path.join(out_dir, base_name + "_asr.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {out_path}")


def run_tgt_asr_from_manifest_batch(
    manifest_path: str,
    asr: Qwen3ASRModel,
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
    batch_size: int = 10,
) -> None:
    """
    读 manifest.jsonl，对每条 tgt wav 跑 ASR（带时间戳），写出 out_dir/{basename}_asr.json
    time_stamps：相对 wav 起点的连续时间
    
    长音频说明：Qwen3 ASR 内部按 chunk 处理，结果有时会少最后几秒（最后一 chunk 未返回）。
    若出现「ASR 比 wav 短约 Xs」的提示，可查阅 qwen_asr 是否支持 chunk_length_s / max_duration 等参数，
    或考虑将长音频先按静音切分为多段再分别识别后拼接。
    """
    # 1. 读 manifest
    # jsonl:
    # - "src": <path of srcwav>
    # - "tgt": <path of tgt wav>
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 2. 对每个 tgt wav
    records = []
    for i, line in enumerate(lines):
        # 加载 jsonl
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(lines)}] 跳过：无效 JSON - {e}")
            continue

        # tgt wav 路径
        tgt_wav = rec.get("tgt")
        if not tgt_wav:
            print(f"[{i+1}/{len(lines)}] 跳过：无 tgt 路径")
            continue
        tgt_wav = os.path.normpath(tgt_wav)
        if not os.path.isfile(tgt_wav):
            print(f"[{i+1}/{len(lines)}] 跳过：文件不存在 {tgt_wav}")
            continue

        rec["_index"] = i + 1
        rec["_tgt_wav_norm"] = tgt_wav
        records.append(rec)

    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for start in range(0, len(records), batch_size):
        batch_recs = records[start:start + batch_size]
        batch_audio = [r["_tgt_wav_norm"] for r in batch_recs]
        batch_lang = [tgt_language] * len(batch_recs)

        print(f"\n处理 batch {start//batch_size + 1}: {len(batch_recs)} 条")

        # 3. 用 asr.transcribe 对这一段 tgt 音频做 ASR，得到一条结果（含整段 text 和词级 time_stamps）
        # ================================================\
        # 注意：Qwen3 ASR 内部按 chunk 处理长音频（如 2s 流式窗口），尾部可能少几秒未返回，属模型/库行为。
        try:
            results = asr.transcribe(
                audio=batch_audio,
                language=batch_lang,
                return_time_stamps=True,
            )
        except Exception as e:
            print(f"  batch 推理失败：{e}")
            continue

        # 几条音频 几条结果
        if not results or len(results) != len(batch_recs):
            print(f"  警告：结果数 {len(results)} != 输入数 {len(batch_recs)}，跳过该 batch")
            continue

        for rec, r in zip(batch_recs, results):
            tgt_wav = rec["_tgt_wav_norm"]

            # 在还没考虑 timeline、会话时间之前，先把 ASR 直接给我们的东西整理成一个要保存的字典
            if r.time_stamps is None:
                ts_list = []
            else:
                ts_list = [
                    {
                        "text": getattr(ts, "text", ""),
                        "start_time": getattr(ts, "start_time", 0.0),
                        "end_time": getattr(ts, "end_time", 0.0),
                    }
                    for ts in r.time_stamps
                ]
            # ================================================/
            
            out = {
                "src": rec.get("src", ""),
                "tgt": tgt_wav,
                "text": r.text or "",
                "time_stamps": ts_list,
            }
            
            # # 👈 prediction_length 用最后一个 token 的结束时间
            # ts_for_len =  out.get("time_stamps") or []
            # if ts_for_len: last_end = ts_for_len[-1]["end_time"]
            # else: last_end = 0.0
            # out["prediction_length"] = round(last_end, 2)

            # 5. 保存结果
            base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
            out_path = os.path.join(out_dir, base_name + "_asr.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {out_path}")
# ===========================================================/

def step1_asr_vllm(
    manifest: str = "data/output_qwen_livetranslate/manifest.jsonl",
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
    batch_size: int = 32,
    max_new_tokens: int = 4096,
    gpu_memory_utilization: float = 0.7,
    asr_model: str = None,
    forced_aligner: str = None,
):
    """
    vLLM 后端：用 Qwen3ASRModel.LLM 加载，transcribe 接口与 transformers 版一致。
    需先安装: pip install -U qwen-asr[vllm]
    """
    asr = Qwen3ASRModel.LLM(
        model=asr_model or ASR_MODEL_PATH,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner=forced_aligner or FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
    )
    run_tgt_asr_from_manifest_batch(
        manifest,
        asr,
        tgt_language=tgt_language,
        out_dir=out_dir,
        batch_size=batch_size,
    )
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

'''
pip install qwen-asr[vllm]
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 ASR (vLLM 后端) 按 manifest 跑 tgt 音频，写出 *_asr.json")
    parser.add_argument("--manifest", type=str, default="data/output_qwen_livetranslate/manifest.jsonl", help="manifest.jsonl 路径")
    parser.add_argument("--out_dir", type=str, default="data/output_qwen_asr", help="输出目录")
    parser.add_argument("--tgt_language", type=str, default="Chinese", help="目标语种，如 Chinese / English")
    parser.add_argument("--batch_size", type=int, default=32, help="每批条数，-1 表示不限制")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="单条最大生成 token 数，长音频可调大")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="vLLM 占 GPU 显存比例")
    parser.add_argument("--asr_model", type=str, default=None, help="ASR 模型名，默认 Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--forced_aligner", type=str, default=None, help="ForcedAligner 模型名，默认 Qwen/Qwen3-ForcedAligner-0.6B")
    args = parser.parse_args()

    step1_asr_vllm(
        manifest=args.manifest,
        tgt_language=args.tgt_language,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        asr_model=args.asr_model,
        forced_aligner=args.forced_aligner,
    )