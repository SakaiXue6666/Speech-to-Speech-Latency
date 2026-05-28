"""Qwen3 ASR vLLM backend.

Higher throughput; recommended for large batches. Install before use::

    pip install -U qwen-asr[vllm]
    # Also recommended for timestamp acceleration (ForcedAligner)
    pip install -U flash-attn --no-build-isolation

Handles model loading only; manifest reading, inference, and result saving
are handled by the shared logic in ``_runner.py``.

Note: vLLM requires the main logic to run under ``if __name__ == '__main__'``
to avoid multiprocessing spawn errors.
"""

import argparse

import torch

from qwen_asr import Qwen3ASRModel

from ._runner import (
    ASR_MODEL_PATH,
    FORCED_ALIGNER_PATH,
    release_asr,
    run_tgt_asr_from_manifest_batch,
)


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
    """Run Qwen3 ASR via the vLLM backend over a manifest.

    Loads :data:`ASR_MODEL_PATH` (with the Qwen3 ForcedAligner attached) via
    ``Qwen3ASRModel.LLM``, transcribes every tgt WAV in the manifest via
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
        gpu_memory_utilization: Fraction of GPU memory reserved by vLLM.
        asr_model: Override for :data:`ASR_MODEL_PATH`; ``None`` uses the default.
        forced_aligner: Override for :data:`FORCED_ALIGNER_PATH`; ``None`` uses the default.
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
    release_asr(asr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3 ASR (vLLM backend): run tgt audio from manifest "
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
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="max generated tokens per sample")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization ratio for vLLM")
    parser.add_argument("--asr_model", type=str, default=None,
                        help="ASR model name")
    parser.add_argument("--forced_aligner", type=str, default=None,
                        help="Forced aligner model name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
