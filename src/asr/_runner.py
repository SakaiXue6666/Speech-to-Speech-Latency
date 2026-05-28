"""Shared ASR batch-processing logic used by both ``qwen_transformers`` and ``qwen_vllm`` backends.

The only difference between the two backends is how the model is loaded
(``from_pretrained`` vs ``LLM``). Both resulting ``asr`` objects expose the
same ``transcribe(audio, language, return_time_stamps)`` interface, so manifest
reading, batch inference, and result saving can all be shared here.
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
    """Run ASR with word-level timestamps on every tgt WAV in the manifest.

    Dispatches the WAVs in fixed-size batches to ``asr.transcribe`` and
    writes one ``{basename}_asr.json`` per WAV under ``out_dir``. Malformed
    manifest lines, missing files, and per-batch inference errors are logged
    and skipped without raising.

    Args:
        manifest_path: Path to a JSONL manifest produced by
            :meth:`PipelineConfig.build_manifest`; each line must contain at
            least a ``tgt`` field.
        asr: ASR model exposing
            ``transcribe(audio, language, return_time_stamps) -> List[Result]``.
            Both the Transformers and vLLM backends in this package satisfy
            this contract.
        tgt_language: Target language name passed to the model
            (e.g. ``"Chinese"``, ``"Japanese"``); broadcast to every WAV in
            the batch.
        out_dir: Directory where ``{basename}_asr.json`` files are written.
        batch_size: Number of WAVs sent to ``asr.transcribe`` per call.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    records: List[dict] = []
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(lines)}] Skipping: invalid JSON - {e}")
            continue

        tgt_wav = rec.get("tgt")
        if not tgt_wav:
            print(f"[{i+1}/{len(lines)}] Skipping: no tgt path")
            continue
        tgt_wav = os.path.normpath(tgt_wav)
        if not os.path.isfile(tgt_wav):
            print(f"[{i+1}/{len(lines)}] Skipping: file not found {tgt_wav}")
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

        print(f"\nProcessing batch {start // batch_size + 1}: {len(batch_recs)} items")

        try:
            results = asr.transcribe(
                audio=batch_audio,
                language=batch_lang,
                return_time_stamps=True,
            )
        except Exception as e:
            print(f"  Batch inference failed: {e}")
            continue

        if not results or len(results) != len(batch_recs):
            print(f"  Warning: result count {len(results)} != input count {len(batch_recs)}, skipping batch")
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
            print(f"  Saved: {out_path}")


def release_asr(asr: Any) -> None:
    """Delete the ASR model and free GPU memory (``gc.collect`` + ``empty_cache``).

    Args:
        asr: The ASR model returned by either ``Qwen3ASRModel.from_pretrained``
            or ``Qwen3ASRModel.LLM``.
    """
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
