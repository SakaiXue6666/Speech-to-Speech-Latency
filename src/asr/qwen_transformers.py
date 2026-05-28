"""Qwen3 ASR Transformers backend.

Handles model loading only; manifest reading, inference, and result saving
are handled by the shared logic in ``_runner.py``.
"""

import argparse
import os

import torch

from qwen_asr import Qwen3ASRModel

from ._runner import (
    ASR_MODEL_PATH,
    FORCED_ALIGNER_PATH,
    release_asr,
    run_tgt_asr_from_manifest_batch,
)


def step1_asr(
    manifest: str = "data/output_qwen_livetranslate/manifest.jsonl",
    tgt_language: str = "Chinese",
    out_dir: str = "data/output_qwen_asr",
    batch_size: int = 10,
    max_new_tokens: int = 1024,
):
    """Run Qwen3 ASR via the HuggingFace Transformers backend over a manifest.

    Loads :data:`ASR_MODEL_PATH` (with the Qwen3 ForcedAligner attached) onto
    ``cuda:0`` in bfloat16, transcribes every tgt WAV in the manifest via
    :func:`run_tgt_asr_from_manifest_batch`, and releases the model afterwards.

    Args:
        manifest: Path to a JSONL manifest with one ``{"src": ..., "tgt": ...}``
            record per line.
        tgt_language: Target language name passed to the ASR model
            (e.g. ``"Chinese"``, ``"Japanese"``).
        out_dir: Directory where ``{basename}_asr.json`` files are written.
        batch_size: Both the manifest dispatch batch size and the model's
            ``max_inference_batch_size``.
        max_new_tokens: Maximum generated tokens per sample.
    """
    print(f"[ASR] loading model: {ASR_MODEL_PATH}")
    print(f"[ASR] forced aligner: {FORCED_ALIGNER_PATH}")
    print(
        f"[ASR] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}"
    )
    asr = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    print("[ASR] model ready, start inference...")
    run_tgt_asr_from_manifest_batch(
        manifest,
        asr,
        tgt_language=tgt_language,
        out_dir=out_dir,
        batch_size=batch_size,
    )
    release_asr(asr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3 ASR (Transformers backend): run tgt audio from manifest "
                    "and write *_asr.json"
    )
    parser.add_argument("--manifest", type=str,
                        default="data/output_qwen_livetranslate/manifest.jsonl",
                        help="manifest.jsonl path")
    parser.add_argument("--out_dir", type=str,
                        default="data/output_qwen_asr",
                        help="output directory")
    parser.add_argument("--tgt_language", type=str, default="Chinese",
                        help="target language, e.g. Chinese / English")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="max generated tokens per sample")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    step1_asr(
        manifest=args.manifest,
        tgt_language=args.tgt_language,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
