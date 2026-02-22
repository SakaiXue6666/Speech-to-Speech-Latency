import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from qwen_asr import Qwen3ASRModel
from SEGALE import segale_align as sa
from LongYAAL.softsegmenter import Instance, YAALScorer

'''
把SEGALE\segale_align.py的 compute_embedding_api 改成分批处理
conda run --no-capture-output -n s2s_latency python main.py --input_json data/input_samples.json --output_json data/output/merge_output.json --task_lang zh --language Chinese --proc_device cuda:0
'''

@dataclass
class WordTS:
    """Simple timestamp unit used across this pipeline.

    Attributes:
        text: Token text (word/subword/character).
        start: Start time in seconds.
        end: End time in seconds.
    """

    text: str
    start: float
    end: float


def read_json(path: str) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """Write a UTF-8 JSON file with pretty indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for approximate matching.

    We keep this intentionally simple:
    - lowercase
    - remove whitespace
    """
    return "".join(text.lower().split())


def parse_word_timestamps(raw_ts: Sequence[Any]) -> List[WordTS]:
    """Convert mixed timestamp records to internal `WordTS` list.

    Supported item forms:
    1) dict form:
       {"text": "...", "start": 0.1, "end": 0.2}
       {"text": "...", "start_time": 0.1, "end_time": 0.2}
    2) object form (from qwen-asr):
       item.text, item.start_time, item.end_time
    """
    parsed: List[WordTS] = []
    for item in raw_ts:
        if isinstance(item, dict):
            txt = str(item.get("text", ""))
            st = float(item.get("start", item.get("start_time", 0.0)))
            ed = float(item.get("end", item.get("end_time", st)))
            parsed.append(WordTS(text=txt, start=st, end=ed))
        else:
            txt = str(getattr(item, "text", ""))
            st = float(getattr(item, "start_time", 0.0))
            ed = float(getattr(item, "end_time", st))
            parsed.append(WordTS(text=txt, start=st, end=ed))
    return parsed


def init_qwen3_asr_model(
    asr_model_path: str,
    forced_aligner_path: str,
    device_map: str,
    dtype: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
):
    """Initialize Qwen3-ASR model once and reuse for all samples.

    Why:
    - Loading model per sample is very slow.
    - Reusing one model instance is the standard production setup.
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    return Qwen3ASRModel.from_pretrained(
        asr_model_path,
        dtype=torch_dtype,
        device_map=device_map,
        forced_aligner=forced_aligner_path,
        forced_aligner_kwargs=dict(dtype=torch_dtype, device_map=device_map),
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )


def qwen3_asr_with_timestamps(
    model: Any,
    tgt_audio_path: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Step 1: run target ASR + forced alignment.

    Output format:
    {
      "tgt_hyp_transcript": str,
      "tgt_hyp_timestamps": [{"text", "start", "end"}, ...]
    }
    """
    result = model.transcribe(
        audio=tgt_audio_path,
        language=language,
        return_time_stamps=True,
    )[0]

    hyp_text = result.text
    hyp_ts = parse_word_timestamps(result.time_stamps or [])
    return {
        "tgt_hyp_transcript": hyp_text,
        "tgt_hyp_timestamps": [ts.__dict__ for ts in hyp_ts],
    }


def segale_segment_and_align(
    hyp_text: str,
    ref_text: str,
    segmenter: str,
    task_lang: str,
    embedding_model: str,
    proc_device: str,
    save_folder: str,
    max_size: int = 8,
) -> Dict[str, Any]:
    """Step 2: segment + align `hyp_text` and `ref_text` with SEGALE.

    Notes:
    - SEGALE internally calls vecalign via subprocess.
    - We use SEGALE's Python functions directly to keep this in one script.
    """
    os.makedirs(save_folder, exist_ok=True)
    sa.set_seed(42)
    sa.VERBOSE = 0
    sa.SPACY = segmenter
    sa.STOP_JUMP = 0.15
    sa.COST_MAX = 0.30
    sa.COST_MIN = 0.30

    # Choose sentence segmentation backend.
    if segmenter == "spacy":
        sa.init_config(task_lang)
        segment_fn = sa.segment_sentences_by_spacy
    else:
        segment_fn = sa.segment_sentences_by_ersatz

    # Segment both sides into sentence-like units.
    ref_segments = [s for s in segment_fn(ref_text) if s.strip()]
    hyp_segments = [s for s in segment_fn(hyp_text) if s.strip()]

    # Fallback if splitter returns nothing.
    if not ref_segments:
        ref_segments = [ref_text]
    if not hyp_segments:
        hyp_segments = [hyp_text]

    # Build overlap embeddings for vecalign.
    tokenizer, model = sa.load_alternative_model(proc_device, embedding_model)
    ref_overlap, ref_embed = sa.generate_overlap_and_embedding(
        ref_segments, model, tokenizer, max_size=max_size
    )
    hyp_overlap, hyp_embed = sa.generate_overlap_and_embedding(
        hyp_segments, model, tokenizer, max_size=max_size
    )

    raw_alignment = sa.run_vecalign_explore(
        # In this setup:
        # src side = reference segments
        # tgt side = hypothesis segments
        "\n".join(ref_segments),
        "\n".join(hyp_segments),
        "\n".join(ref_overlap),
        "\n".join(hyp_overlap),
        ref_embed,
        hyp_embed,
        "merge_doc",
        save_folder,
        max_size=max_size,
    )

    # Convert SEGALE tuple format into JSON-friendly records.
    pairs = []
    for ref_idx, hyp_idx in raw_alignment:
        pairs.append(
            {
                "ref_indices": ref_idx,
                "hyp_indices": hyp_idx,
                "ref_text": " ".join(ref_segments[i] for i in ref_idx) if ref_idx else "",
                "hyp_text": " ".join(hyp_segments[i] for i in hyp_idx) if hyp_idx else "",
            }
        )

    return {
        "ref_segments": ref_segments,
        "hyp_segments": hyp_segments,
        "alignment_pairs": pairs,
    }


def match_segment_to_token_span(
    segment_text: str,
    token_texts: Sequence[str],
    start_search_idx: int,
) -> Optional[Tuple[int, int]]:
    """Find token span [i, j] whose concatenation matches `segment_text`.

    This is monotonic (left-to-right) matching:
    - start from `start_search_idx`
    - greedily find the first span that matches after normalization
    """
    target = normalize_text(segment_text)
    if not target:
        return None

    n = len(token_texts)
    for i in range(start_search_idx, n):
        acc = ""
        for j in range(i, n):
            acc += normalize_text(token_texts[j])
            if acc == target:
                return i, j
            # Stop early if current prefix can no longer match the target.
            if len(acc) > len(target) and not acc.startswith(target):
                break
    return None


def build_hyp_segment_time_map(
    hyp_segments: Sequence[str],
    hyp_timestamps: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Map each hypothesis segment index to time span + covered tokens."""
    tokens = parse_word_timestamps(hyp_timestamps)
    token_texts = [t.text for t in tokens]
    seg_time: Dict[int, Dict[str, Any]] = {}
    cursor = 0

    for idx, seg in enumerate(hyp_segments):
        matched = match_segment_to_token_span(seg, token_texts, cursor)
        if matched is None:
            # If a segment cannot be mapped, skip it safely.
            continue
        s, e = matched
        cursor = e + 1
        seg_time[idx] = {
            "token_start": s,
            "token_end": e,
            "start": tokens[s].start,
            "end": tokens[e].end,
            "tokens": [tokens[k].__dict__ for k in range(s, e + 1)],
        }
    return seg_time


def compute_longyaal(
    alignment_pairs: Sequence[Dict[str, Any]],
    hyp_seg_time: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Step 3: compute LongYAAL from alignment and hypothesis timing.

    For each alignment pair:
    - collect time spans of all aligned hypothesis segments
    - flatten their token timestamps
    - build one `Instance` for YAALScorer
    """
    instances: Dict[int, Instance] = {}
    details: List[Dict[str, Any]] = []

    for idx, pair in enumerate(alignment_pairs):
        hyp_indices = pair.get("hyp_indices", [])
        if not hyp_indices:
            continue

        spans = [hyp_seg_time[i] for i in hyp_indices if i in hyp_seg_time]
        if not spans:
            continue

        # Pair-level time window.
        start = min(s["start"] for s in spans)
        end = max(s["end"] for s in spans)

        # Collect all tokens in this aligned pair.
        seg_tokens: List[Dict[str, Any]] = []
        for s in spans:
            seg_tokens.extend(s["tokens"])

        # Approximate emission delays in milliseconds.
        delays_ms = [
            max(0.0, (float(tok["end"]) - start) * 1000.0) for tok in seg_tokens
        ]
        source_len_ms = max(1.0, (end - start) * 1000.0)
        reference = pair.get("ref_text", "")
        prediction = pair.get("hyp_text", "")

        ins_payload = {
            "reference": reference,
            "prediction": prediction,
            "source_length": source_len_ms,
            "delays": delays_ms,
            "elapsed": delays_ms,
            "recording_end": source_len_ms,
        }
        instances[idx] = Instance(ins_payload, latency_unit="word")

    scorer = YAALScorer(is_longform=True)
    overall = scorer(instances) if instances else float("nan")

    for idx, ins in instances.items():
        details.append(
            {
                "alignment_id": idx,
                "longyaal": scorer.compute(ins),
                "reference": ins.reference,
                "prediction": ins.prediction,
            }
        )

    return {
        "longyaal_overall": overall,
        "longyaal_by_alignment": details,
    }


def run_pipeline_for_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    asr_model: Any,
) -> Dict[str, Any]:
    """Run complete 3-step pipeline for one sample."""
    import time as _time

    sample_id = sample.get("sample_id", "unknown")
    tgt = sample["tgt"]
    tgt_audio = tgt["audio_path"]
    tgt_ref = tgt["reference_transcript"]

    # Step 1: ASR + forced align timestamps.
    print(f"[{sample_id}] Step 1/3: ASR + forced alignment on {tgt_audio} ...")
    t0 = _time.time()
    step1 = qwen3_asr_with_timestamps(
        model=asr_model,
        tgt_audio_path=tgt_audio,
        language=args.language,
    )
    print(f"[{sample_id}] Step 1/3 done. ({_time.time()-t0:.1f}s, "
          f"{len(step1['tgt_hyp_timestamps'])} words)")

    # ⭐《我电脑就8G显存》Free ASR model from GPU to make room for SEGALE embedding model.
    print(f"[{sample_id}] Offloading ASR model from GPU ...")
    if hasattr(asr_model, "model") and asr_model.model is not None:
        asr_model.model.cpu()
    if hasattr(asr_model, "forced_aligner") and asr_model.forced_aligner is not None:
        asr_model.forced_aligner.model.cpu()
    torch.cuda.empty_cache()

    # Step 2: text segmentation + alignment.
    print(f"[{sample_id}] Step 2/3: SEGALE segment + align ...")
    t0 = _time.time()
    step2 = segale_segment_and_align(
        hyp_text=step1["tgt_hyp_transcript"],
        ref_text=tgt_ref,
        segmenter=args.segmenter,
        task_lang=args.task_lang,
        embedding_model=args.embedding_model,
        proc_device=args.proc_device,
        save_folder=os.path.join(args.tmp_dir, sample_id),
        max_size=args.max_size,
    )
    print(f"[{sample_id}] Step 2/3 done. ({_time.time()-t0:.1f}s, "
          f"{len(step2['alignment_pairs'])} aligned pairs)")

    # Bridge step: map aligned hyp segments to real timestamps.
    print(f"[{sample_id}] Bridge: mapping timestamps ...")
    seg_time = build_hyp_segment_time_map(
        hyp_segments=step2["hyp_segments"],
        hyp_timestamps=step1["tgt_hyp_timestamps"],
    )

    # Step 3: compute latency metric.
    print(f"[{sample_id}] Step 3/3: computing LongYAAL latency ...")
    t0 = _time.time()
    step3 = compute_longyaal(
        alignment_pairs=step2["alignment_pairs"],
        hyp_seg_time=seg_time,
    )
    print(f"[{sample_id}] Step 3/3 done. ({_time.time()-t0:.1f}s)")
    print(f"[{sample_id}] Pipeline complete.")

    return {
        "sample_id": sample_id,
        "input": sample,
        "step1_qwen3_asr_forcedalign": step1,
        "step2_segale": step2,
        "step3_longyaal": step3,
    }


def validate_sample(sample: Dict[str, Any]) -> None:
    """Validate minimum fields required by this pipeline."""
    required = [
        ("src", "audio_path"),
        ("src", "transcript"),
        ("tgt", "audio_path"),
        ("tgt", "reference_transcript"),
    ]
    for parent, key in required:
        if parent not in sample or key not in sample[parent]:
            raise ValueError(f"Missing field: {parent}.{key}")
    # src.word_timestamps is optional; only needed for archival purposes.
    sample.setdefault("src", {}).setdefault("word_timestamps", [])


def write_input_template(path: str) -> None:
    """Write one sample template showing expected input format."""
    template = {
        "samples": [
            {
                "sample_id": "utt_0001",
                "src": {
                    "audio_path": "data/src/utt_0001.wav",
                    "transcript": "today weather is good we go to park",
                    "word_timestamps": [
                        {"text": "today", "start": 0.12, "end": 0.36},
                        {"text": "weather", "start": 0.38, "end": 0.62},
                        {"text": "good", "start": 0.64, "end": 0.89},
                    ],
                },
                "tgt": {
                    "audio_path": "data/tgt/utt_0001.wav",
                    "reference_transcript": "today weather is good, we go to park.",
                },
            }
        ]
    }
    write_json(path, template)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for this merge pipeline."""
    parser = argparse.ArgumentParser(
        description="S2S metric merge: Qwen3-ASR + SEGALE + LongYAAL"
    )
    parser.add_argument("--input_json", type=str, help="Input samples json path.")
    parser.add_argument("--output_json", type=str, default="merge_output.json")
    parser.add_argument("--write_template", type=str, default="")

    parser.add_argument("--asr_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument(
        "--forced_aligner", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B"
    )
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--max_inference_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--segmenter", choices=["spacy", "ersatz"], default="spacy")
    parser.add_argument("--task_lang", type=str, default="zh")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/LaBSE",
    )
    parser.add_argument("--proc_device", type=str, default="cuda:0")
    parser.add_argument("--max_size", type=int, default=8)
    parser.add_argument("--tmp_dir", type=str, default=".merge_tmp")
    return parser


def main() -> None:
    """Script entry.

    Flow:
    1) Parse arguments.
    2) Optional: write template and exit.
    3) Read input samples.
    4) Initialize ASR model once.
    5) Run pipeline for each sample.
    6) Save final result JSON.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.write_template:
        write_input_template(args.write_template)
        print(f"Template written: {args.write_template}")
        return

    if not args.input_json:
        raise ValueError("Please set --input_json or use --write_template.")

    payload = read_json(args.input_json)
    samples = payload["samples"] if isinstance(payload, dict) else payload

    print(f"Loading ASR model: {args.asr_model} ...")
    asr_model = init_qwen3_asr_model(
        asr_model_path=args.asr_model,
        forced_aligner_path=args.forced_aligner,
        device_map=args.device_map,
        dtype=args.dtype,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print("ASR model loaded.")

    outputs: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*50}")
        print(f"Processing sample {i}/{len(samples)}: {sample.get('sample_id', 'unknown')}")
        print(f"{'='*50}")
        validate_sample(sample)
        outputs.append(run_pipeline_for_sample(sample, args, asr_model))

    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, {"results": outputs})
    print(f"Done. Saved to {args.output_json}")


if __name__ == "__main__":
    main()
