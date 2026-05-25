"""中间产物生成：把 ASR 输出 + segments yaml + 原文/参考译文，整理成
后续 SEGALE 对齐和 LongYAAL 评测要消费的 ``instances.log`` / ``hyp.jsonl`` /
``ref.jsonl``。
"""

import json
import os
import wave
from typing import Dict, List, Tuple

import yaml


def _get_wav_duration_sec(wav_path: str) -> float:
    """读 wav 文件时长（秒）。失败返回 0.0。"""
    try:
        with wave.open(wav_path, "r") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _build_instances_log_s2s(
    ref_segments_yaml: str, asr_dir: str, output_file: str
) -> None:
    """从 ref_segments.yaml + asr_dir/*_asr.json 生成 s2s 格式 instances.log。

    yaml 的 wav 与 asr 的 src 按 basename 匹配；每条记录对应一个 src wav。
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

    # 行间用 \n 分隔，最后一行后不写 \n，避免多出一行
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
    """把 ASR 结果转成 instances.log。

    当前仅支持 ``s2s=True``（speech-to-speech）。其它模式留作扩展位。
    """
    if not s2s:
        raise NotImplementedError("当前只支持 s2s=True 模式")
    if not (yaml_file and asr_dir and output_file):
        raise ValueError("yaml_file / asr_dir / output_file 都必须提供")
    _build_instances_log_s2s(yaml_file, asr_dir, output_file)


def get_jsonl(
    src_txt: str,
    tgt_ref_txt: str,
    src_segments_yaml: str,
    instances: str,
) -> Tuple[List[dict], List[dict]]:
    """根据原文/参考译文/segments yaml/instances.log，构造 ref/hyp 两份 dict 列表。

    返回:
        (ref_dicts, hyp_dicts)，分别供 ref.jsonl 和 hyp.jsonl 写出。
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

        # 同一 doc_id 的 src 拼到一起，给 hyp.jsonl 用
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
    """生成 SEGALE 阶段需要的 ``ref.jsonl`` 和 ``hyp.jsonl``。"""
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

    print("saved:", out_path_ref, out_path_hyp)
