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
  python qwen3_asr.py --manifest data/output_qwen_livetranslate/manifest.jsonl
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


# 根据 manifest.jsonl 里列出的每条「译文 wav」，对每个 tgt wav 跑一次 Qwen3 ASR（带词级时间戳）
# 并把转写结果和两种时间戳写到对应的 .json
def run_tgt_asr_from_manifest(
    manifest_path: str,
    asr: Qwen3ASRModel,
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
) -> None:
    """
    读 manifest.jsonl，对每条 tgt wav 跑 ASR（带时间戳），写出 out_dir/{basename}_asr.json。
    只输出 wav 相对时间（time_stamps_no_gap / time_stamps）；会话时间映射由 build_instances_log 负责。

    长音频说明：Qwen3 ASR 内部按 chunk 处理，结果有时会少最后几秒（最后一 chunk 未返回）。
    若出现「ASR 比 wav 短约 Xs」的提示，可查阅 qwen_asr 是否支持 chunk_length_s / max_duration 等参数，
    或考虑将长音频先按静音切分为多段再分别识别后拼接。
    """
    # 1. 读 manifest
    # jsonl:
    # - "src": <path of srcwav>
    # - "tgt": <path of tgt wav>
    # - "tgt_timeline": <path of json>
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
            "tgt_timeline": rec.get("tgt_timeline", ""),
            "text": r.text or "",
            "time_stamps_no_gap": ts_list,
            "time_stamps": ts_list,
            "time_reference": "tgt wav no gap (ASR only; session mapping in build_instances_log)",
        }
        last_end = ts_list[-1]["end_time"] if ts_list else 0.0
        out["prediction_length"] = round(last_end, 2)

        # 5. 保存结果
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
        max_new_tokens=1024,
    )

    # ===========================================================\
    # DEBUG / SAFE
    # def _infer_asr_transformers_debug(self, contexts, wavs, languages):
    #     outs = []
    #     texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

    #     batch_size = self.max_inference_batch_size
    #     if batch_size is None or batch_size < 0:
    #         batch_size = len(texts)

    #     for i in range(0, len(texts), batch_size):
    #         sub_text = texts[i : i + batch_size]
    #         sub_wavs = wavs[i : i + batch_size]

    #         inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
    #         inputs = inputs.to(self.model.device).to(self.model.dtype)
    #         # ++++++++++++++++++++++++++++++++++++++++\
    #         gen_out = self.model.generate(
    #             **inputs,
    #             max_new_tokens=self.max_new_tokens,
    #         )

    #         # 兼容不同 transformers 返回
    #         seq = getattr(gen_out, "sequences", None)
    #         if seq is None:
    #             # 有些实现直接返回 tensor
    #             seq = gen_out

    #         gen = seq[:, inputs["input_ids"].shape[1]:]

    #         eos_ids = {151645, 151643}
    #         for b in range(gen.shape[0]):
    #             row = gen[b].tolist()
    #             eos_pos = next((t for t, tok in enumerate(row) if tok in eos_ids), None)
    #             if eos_pos is None:
    #                 real_len = len(row)
    #                 truncated = True
    #             else:
    #                 real_len = eos_pos + 1
    #                 truncated = False
    #             print(
    #                 f"[ASR DEBUG] batch_start={i} item={b} "
    #                 f"real_len={real_len} truncated={truncated} last_token_id={row[-1]}"
    #             )
    #         # ++++++++++++++++++++++++++++++++++++++++/
    #         decoded = self.processor.batch_decode(
    #             gen,
    #             skip_special_tokens=True,
    #             clean_up_tokenization_spaces=False,
    #         )
    #         outs.extend(list(decoded))

    #     return outs

    # # 把实例方法替换掉（只影响你当前脚本里的这个 asr 对象）
    # asr._infer_asr_transformers = types.MethodType(_infer_asr_transformers_debug, asr)
    # ===========================================================/
    if args.manifest:
        run_tgt_asr_from_manifest(
            args.manifest,
            asr,
            tgt_language=args.tgt_language,
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