"""ASR 跑批通用逻辑：被 ``qwen_transformers`` 和 ``qwen_vllm`` 两个后端共享。

两个后端唯一的区别是模型加载方式（``from_pretrained`` vs ``LLM``），加载完得到
的 ``asr`` 对象都暴露相同的 ``transcribe(audio, language, return_time_stamps)``
接口，因此可以共用同一份 manifest 读取 / batch 推理 / 结果保存逻辑。
"""

import gc
import json
import os
from typing import Any, List

import torch


ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"


def run_tgt_asr_from_manifest_batch(
    manifest_path: str,
    asr: Any,
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
    batch_size: int = 10,
) -> None:
    """对 manifest 中每条 tgt wav 跑 ASR（带词级时间戳），分 batch 处理。

    输出：``out_dir/{basename}_asr.json``。

    长音频说明：Qwen3 ASR 内部按 chunk 处理，结果有时会少最后几秒（最后一
    chunk 未返回）。如出现「ASR 比 wav 短约 Xs」的提示，可考虑将长音频按
    静音切分后再分别识别拼接。
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    records: List[dict] = []
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

        rec["_index"] = i + 1
        rec["_tgt_wav_norm"] = tgt_wav
        records.append(rec)

    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for start in range(0, len(records), batch_size):
        batch_recs = records[start:start + batch_size]
        batch_audio = [r["_tgt_wav_norm"] for r in batch_recs]
        batch_lang = [tgt_language] * len(batch_recs)

        print(f"\n处理 batch {start // batch_size + 1}: {len(batch_recs)} 条")

        try:
            results = asr.transcribe(
                audio=batch_audio,
                language=batch_lang,
                return_time_stamps=True,
            )
        except Exception as e:
            print(f"  batch 推理失败：{e}")
            continue

        if not results or len(results) != len(batch_recs):
            print(f"  警告：结果数 {len(results)} != 输入数 {len(batch_recs)}，跳过该 batch")
            continue

        for rec, r in zip(batch_recs, results):
            tgt_wav = rec["_tgt_wav_norm"]

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

            out = {
                "src": rec.get("src", ""),
                "tgt": tgt_wav,
                "text": r.text or "",
                "time_stamps": ts_list,
            }

            base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
            out_path = os.path.join(out_dir, base_name + "_asr.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {out_path}")


def release_asr(asr: Any) -> None:
    """释放 ASR 模型显存。两个后端跑完后都需要做。"""
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
