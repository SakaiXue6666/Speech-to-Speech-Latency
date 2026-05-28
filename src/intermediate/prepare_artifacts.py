"""Intermediate artifact generation: combines ASR output, segments YAML, and
source/reference text into the ``instances.log`` / ``hyp.jsonl`` / ``ref.jsonl``
files consumed by the downstream SEGALE alignment and LongYAAL evaluation steps.
"""

import json
import os
import wave
from typing import Dict, List, Tuple

import yaml


def _get_wav_duration_sec(wav_path: str) -> float:
    """Return WAV file duration in seconds; returns 0.0 on any error."""
    try:
        with wave.open(wav_path, "r") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _build_instances_log_s2s(
    ref_segments_yaml: str, asr_dir: str, output_file: str
) -> None:
    """Generate a speech-to-speech instances.log from ref_segments.yaml and asr_dir/*_asr.json.

    YAML wav entries are matched to ASR src fields by basename; each record
    corresponds to one source WAV file.
    """
    with open(ref_segments_yaml, "r", encoding="utf-8") as f:
        segments = yaml.safe_load(f) or []

    unique_wavs: List[str] = []
    seen = set()
    for seg in segments:
        w = seg.get("wav")
        if w is not None and w not in seen:
            seen.add(w)
            unique_wavs.append(w)
    print(f"Unique wavs in yaml: \n{unique_wavs}")

    asr_by_src: Dict[str, dict] = {}
    asr_dir = os.path.normpath(asr_dir)
    for fn in os.listdir(asr_dir):
        if not fn.endswith("_asr.json"):
            continue
        path = os.path.join(asr_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        src = data.get("src")
        if not src:
            continue
        key = os.path.basename(os.path.normpath(src))
        asr_by_src[key] = data

    instances: List[dict] = []
    for idx, yaml_wav in enumerate(unique_wavs):
        key = os.path.basename(os.path.normpath(yaml_wav))
        asr = asr_by_src.get(key)
        if asr is None:
            continue

        ts = asr.get("time_stamps") or []
        delays = [int(round(t["end_time"] * 1000)) for t in ts]
        prediction_offset = delays[0] if delays else 0

        tgt_path = os.path.normpath(asr.get("tgt") or "")
        src_path = os.path.normpath(asr.get("src", ""))

        prediction_length = _get_wav_duration_sec(tgt_path) if tgt_path else 0.0
        source_length_sec = _get_wav_duration_sec(src_path)

        prediction_text = asr.get("text") or ""
        prediction_units = [t.get("text", "") for t in ts]
        prediction_unit_char_starts_ends: List[Tuple[int, int]] = [
            (t.get("char_start", -1), t.get("char_end", -1)) for t in ts
        ]

        instances.append({
            "index": idx,
            "source": src_path,
            "reference": src_path,
            "prediction": tgt_path,
            "delays": delays,
            "elapsed": [],
            "prediction_offset": prediction_offset,
            "prediction_length": prediction_length,
            "source_length": source_length_sec,
            "prediction_text": prediction_text,
            "prediction_units": prediction_units,
            "prediction_unit_char_starts_ends": prediction_unit_char_starts_ends,
        })

    # Records separated by newline; no trailing newline after the last record
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, rec in enumerate(instances):
            if i > 0:
                f.write("\n")
            f.write(json.dumps(rec, ensure_ascii=False))


def asr_to_instances(
    s2s: bool,
    yaml_file: str,
    asr_dir: str,
    output_file: str,
) -> None:
    """Convert ASR JSON outputs to a SimulEval-style ``instances.log``.

    Currently only ``s2s=True`` (speech-to-speech) is supported; other modes
    are reserved for future use.

    Args:
        s2s: Must be ``True``; otherwise :class:`NotImplementedError` is raised.
        yaml_file: Reference segments YAML; the per-segment ``wav`` field is
            used to enumerate documents in order.
        asr_dir: Directory containing ``*_asr.json`` (already enriched with
            char spans).
        output_file: Destination path of ``instances.log`` (one JSON record
            per line, no trailing newline).
    """
    if not s2s:
        raise NotImplementedError("only s2s=True mode is currently supported")
    if not (yaml_file and asr_dir and output_file):
        raise ValueError("yaml_file, asr_dir, and output_file are all required")
    _build_instances_log_s2s(yaml_file, asr_dir, output_file)


def get_jsonl(
    src_txt: str,
    tgt_ref_txt: str,
    src_segments_yaml: str,
    instances: str,
) -> Tuple[List[dict], List[dict]]:
    """Build ref and hyp dict lists from source text, reference translation, segments YAML, and instances.log.

    Returns:
        (ref_dicts, hyp_dicts), to be written as ref.jsonl and hyp.jsonl respectively.
    """
    with open(src_txt, "r", encoding="utf-8") as f:
        src_text_lines = [line.rstrip("\n") for line in f]
    with open(tgt_ref_txt, "r", encoding="utf-8") as f:
        tgt_ref_text_lines = [line.rstrip("\n") for line in f]
    with open(src_segments_yaml, "r", encoding="utf-8") as f:
        src_yaml_lines = yaml.safe_load(f)

    src_for_hyp_docs: List[dict] = []
    ref_dicts: List[dict] = []
    curr_doc = None

    for i in range(len(src_yaml_lines)):
        ref_dicts.append({
            "src": src_text_lines[i],
            "tgt": tgt_ref_text_lines[i],
            "sys_id": None,
            "doc_id": src_yaml_lines[i]["wav"],
            "seg_id": i + 1,
        })

        # Concatenate src sentences of the same doc_id for hyp.jsonl
        if src_yaml_lines[i]["wav"] != curr_doc:
            src_for_hyp_docs.append({"doc_id": src_yaml_lines[i]["wav"], "src": ""})
            curr_doc = src_yaml_lines[i]["wav"]
        src_for_hyp_docs[-1]["src"] += " " + src_text_lines[i]

    with open(instances, "r", encoding="utf-8") as f:
        instances_lines = [line.rstrip("\n") for line in f]

    hyp_dicts: List[dict] = []
    for j, line in enumerate(instances_lines):
        instance_dict = json.loads(line)
        hyp_text = instance_dict["prediction_text"]
        instance_src_basename = os.path.basename(
            str(instance_dict["source"] or "")
        ).replace("\\", "/")
        for src_for_hyp_doc in src_for_hyp_docs:
            if src_for_hyp_doc["doc_id"] == instance_src_basename:
                hyp_dicts.append({
                    "src": src_for_hyp_doc["src"],
                    "tgt": hyp_text,
                    "sys_id": None,
                    "doc_id": src_for_hyp_doc["doc_id"],
                    "seg_id": j + 1,
                })
                break

    return ref_dicts, hyp_dicts


def instances_to_segale(
    src_txt: str,
    tgt_ref_txt: str,
    src_segments_yaml: str,
    instances: str,
    out_dir: str,
) -> None:
    """Generate ``ref.jsonl`` and ``hyp.jsonl`` for the SEGALE alignment step.

    Args:
        src_txt: Source-language transcript file (one sentence per line).
        tgt_ref_txt: Target-language reference translation file (aligned line
            by line with ``src_txt``).
        src_segments_yaml: Per-sentence segments YAML (provides the ``wav``
            field that defines doc grouping).
        instances: Path to ``instances.log`` produced by :func:`asr_to_instances`.
        out_dir: Directory where ``ref.jsonl`` and ``hyp.jsonl`` are written.
    """
    ref_dicts, hyp_dicts = get_jsonl(
        src_txt, tgt_ref_txt, src_segments_yaml, instances
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path_ref = os.path.join(out_dir, "ref.jsonl")
    out_path_hyp = os.path.join(out_dir, "hyp.jsonl")

    with open(out_path_ref, "w", encoding="utf-8") as f:
        for obj in ref_dicts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_path_hyp, "w", encoding="utf-8") as f:
        for obj in hyp_dicts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Saved:", out_path_ref, out_path_hyp)
