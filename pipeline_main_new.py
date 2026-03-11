import gc
import os
import torch
from pipeline_qwen3_asr2 import step1_asr
from pipeline_segale_align_new import step2_segale
from pipeline_longyaal_new import step3_longyaal

import argparse
import json
import os
import wave
import yaml


# ============================================================================\
import json
import os
import unicodedata
from typing import Dict, List, Tuple


def is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def norm_char_stream_with_mapping(text: str) -> Tuple[str, List[int]]:
    """
    把原文 text 转成规范化字符流，并记录：
    norm_text[i] 对应原文 raw_text 的哪个字符下标。

    返回:
        norm_text, norm_to_raw
    """
    raw = unicodedata.normalize("NFKC", text or "")
    norm_chars: List[str] = []
    norm_to_raw: List[int] = []

    for raw_idx, ch in enumerate(raw):
        if is_kept_char(ch):
            norm_chars.append(ch.lower())
            norm_to_raw.append(raw_idx)

    return "".join(norm_chars), norm_to_raw


def norm_unit_text(text: str) -> str:
    raw = unicodedata.normalize("NFKC", text or "")
    return "".join(ch.lower() for ch in raw if is_kept_char(ch))


def add_char_spans_to_asr_json(infile: str, outfile: str = None) -> str:
    context_chars = 80  ###

    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = unicodedata.normalize("NFKC", data.get("text", "") or "")
    ts_list: List[Dict] = data.get("time_stamps", []) or []

    full_norm, norm_to_raw = norm_char_stream_with_mapping(full_text)

    norm_cursor = 0
    new_ts_list: List[Dict] = []

    # ++++++++++++++++++++++++++++++++++++++++++++++
    last_ok_idx = -1
    last_ok_raw_span = (-1, -1)
    last_ok_unit_text = ""

    basename = os.path.basename(infile)
    # ++++++++++++++++++++++++++++++++++++++++++++++

    for idx, ts in enumerate(ts_list):
        unit_text = ts.get("text", "") or ""
        unit_norm = norm_unit_text(unit_text)

        new_ts = dict(ts)

        if not unit_norm:
            new_ts["char_start"] = -1
            new_ts["char_end"] = -1
            new_ts_list.append(new_ts)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            print(
                    f"[empty-unit] file={basename} idx={idx} "
                    f"unit_text={unit_text!r}"
                )
            # ++++++++++++++++++++++++++++++++++++++++++++++
            continue

        pos = full_norm.find(unit_norm, norm_cursor)

        if pos == -1:
            new_ts["char_start"] = -1
            new_ts["char_end"] = -1
            new_ts_list.append(new_ts)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            # norm 全文上下文
            norm_left = max(0, norm_cursor - context_chars)
            norm_right = min(len(full_norm), norm_cursor + context_chars)
            norm_ctx = full_norm[norm_left:norm_right]

            # raw 全文上下文（通过 norm_cursor 尽量映射到 raw）
            if 0 <= norm_cursor < len(norm_to_raw):
                raw_cursor = norm_to_raw[norm_cursor]
            elif norm_to_raw:
                raw_cursor = norm_to_raw[-1] + 1
            else:
                raw_cursor = 0

            raw_left = max(0, raw_cursor - context_chars)
            raw_right = min(len(full_text), raw_cursor + context_chars)
            raw_ctx = full_text[raw_left:raw_right]

            print("=" * 80)
            print(f"[match-fail] file={basename} idx={idx}")
            print(f"unit_text      = {unit_text!r}")
            print(f"unit_norm      = {unit_norm!r}")
            print(f"norm_cursor    = {norm_cursor}")
            print(f"last_ok_idx    = {last_ok_idx}")
            print(f"last_ok_unit   = {last_ok_unit_text!r}")
            print(f"last_ok_span   = {last_ok_raw_span}")
            print(f"norm_ctx       = {_clip(norm_ctx, 200)!r}")
            print(f"raw_ctx        = {_clip(raw_ctx, 200)!r}")

            # 再给一个“从头搜”的参考，看看是不是 cursor 走偏了
            global_pos = full_norm.find(unit_norm)
            print(f"global_find_pos= {global_pos}")

            if global_pos != -1:
                g_start = norm_to_raw[global_pos]
                g_end = norm_to_raw[global_pos + len(unit_norm) - 1] + 1
                print(f"global_raw_span= ({g_start}, {g_end})")
                print(f"global_raw_txt = {_clip(full_text[g_start:g_end], 200)!r}")

            print("=" * 80)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            continue

        norm_start = pos
        norm_end = pos + len(unit_norm)   # 开区间

        raw_start = norm_to_raw[norm_start]
        raw_end = norm_to_raw[norm_end - 1] + 1   # 开区间

        new_ts["char_start"] = raw_start
        new_ts["char_end"] = raw_end
        new_ts_list.append(new_ts)

        # ++++++++++++++++++++++++++++++++++++++++++++++
        last_ok_idx = idx
        last_ok_raw_span = (raw_start, raw_end)
        last_ok_unit_text = unit_text
        # ++++++++++++++++++++++++++++++++++++++++++++++

        norm_cursor = norm_end

    data["text"] = full_text
    data["time_stamps"] = new_ts_list

    if outfile is None:
        base, ext = os.path.splitext(infile)
        outfile = base + "_charspan" + ext

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return outfile

def add_char_spans_for_dir(input_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(input_dir):
        if not fn.endswith("_asr.json"):
            continue

        infile = os.path.join(input_dir, fn)
        outfile = os.path.join(out_dir, fn)

        out = add_char_spans_to_asr_json(infile, outfile)
        print("saved:", out)

# ============================================================================/
# ============================================================================\
def _get_wav_duration_sec(wav_path: str) -> float:
    """读 wav 文件时长（秒）。失败返回 0.0。"""
    try:
        with wave.open(wav_path, "r") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _build_instances_log_s2s(ref_segments_yaml: str, asr_dir: str, output_file: str) -> None:
    """
    从 ref_segments.yaml + asr_dir 下的 *_asr.json 生成 s2s 格式 instances.log。
    yaml 的 wav 与 asr 的 src 按 basename 匹配；每条记录对应一个 source（src wav）。
    """
    # 1. 读 yaml，按出现顺序收集不重复的 wav（每个 wav 对应一个 source / 一行 instances.log）
    with open(ref_segments_yaml, "r", encoding="utf-8") as f:
        segments = yaml.safe_load(f)
    if not segments:
        segments = []
    unique_wavs = []
    seen = set()
    for seg in segments:
        w = seg.get("wav")
        if w is not None and w not in seen:
            seen.add(w)
            unique_wavs.append(w)
    print(f"Unique wavs in yaml: \n{unique_wavs}")

    # 2. 扫描 asr_dir，按 src 的 basename 建表，便于和 yaml 的 wav 匹配
    asr_by_src = {}
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

    # 3. 对每个 yaml 中的 wav 找对应 asr，拼一条 s2s 记录
    instances = []
    for idx, yaml_wav in enumerate(unique_wavs):
        key = os.path.basename(os.path.normpath(yaml_wav))
        asr = asr_by_src.get(key)
        if asr is None:
            continue

        # 时间戳：优先 wav 相对 time_stamps，否则 time_stamps_no_gap
        ts = asr.get("time_stamps") or []

        # delays  # len(delays) == len(units)
        delays = [int(round(t["end_time"] * 1000)) for t in ts]

        # prediction_offset
        prediction_offset = delays[0] if delays else 0

        tgt_path = os.path.normpath(asr.get("tgt") or "")
        src_path = os.path.normpath(asr.get("src", ""))

        # prediction_length / source_length：统一用 wav 文件时长（秒）
        prediction_length = _get_wav_duration_sec(tgt_path) if tgt_path else 0.0
        source_length_sec = _get_wav_duration_sec(src_path)

        prediction_text = asr.get("text") or ""
        prediction_units = [t.get("text", "") for t in ts]

        # ++++++++++++++++++++++++++++++++++++++++++++++
        prediction_unit_char_starts = [t.get("char_start", -1) for t in ts]
        prediction_unit_char_ends = [t.get("char_end", -1) for t in ts]
        prediction_unit_char_starts_ends = list(zip(prediction_unit_char_starts, prediction_unit_char_ends))
        # ++++++++++++++++++++++++++++++++++++++++++++++

        rec = {
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
            "prediction_unit_char_starts_ends": prediction_unit_char_starts_ends,   # 新增
        }
        instances.append(rec)

    # 4. 写出 JSONL（行间用 \n 分隔，最后一行后不写 \n，避免多出一行）
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, rec in enumerate(instances):
            if i > 0:
                f.write("\n")  # 👈 行间用 \n 分隔，最后一行后不写 \n，避免多出一行
            f.write(json.dumps(rec, ensure_ascii=False))


def asr_to_instances(
    s2s: bool = True,
    yaml_file: str = "data/input/ACL.ACLdev2023.en-xx.gold_segments.yaml",
    asr_dir: str = "data/output_qwen_asr",
    output_file: str = "data/output_qwen_asr/instances.log",
):
    if s2s and yaml_file and asr_dir and output_file:
        _build_instances_log_s2s(yaml_file, asr_dir, output_file)
        return
    if not s2s or not yaml_file or not asr_dir or not output_file:
        pass
# ============================================================================/
# ============================================================================\

def get_jsonl(src_txt, tgt_ref_txt, src_segments_yaml, instances):
    # [src_row1, src_row2, ...]
    with open(src_txt, "r", encoding="utf-8") as f:
        src_text_lines = [line.rstrip("\n") for line in f]
    # [ref_row1, ref_row2, ...]
    with open(tgt_ref_txt, "r", encoding="utf-8") as f:
        tgt_ref_text_lines = [line.rstrip("\n") for line in f]
    # [{wav1, start_time_row1: end_time_row1}, {wav1, start_time_row2: end_time_row2}, ...]
    with open(src_segments_yaml, "r", encoding="utf-8") as f:
        src_yaml_lines = yaml.safe_load(f)

    src_for_hyp_docs = []
    ref_dicts = []
    curr_doc = None

    for i in range(len(src_yaml_lines)):
        ref_dict = {
            "src": src_text_lines[i],  # src_row*
            "tgt": tgt_ref_text_lines[i],  # ref_row*
            "sys_id": None,
            "doc_id": src_yaml_lines[i]['wav'],  # {wav?, start_time_row*: end_time_row*}
            "seg_id": i + 1  # * + 1
        }
        ref_dicts.append(ref_dict)
        # -----------------------------------------------------------
        # 给 hyp.jsonl 用的，把同一个 doc_id 的 src 拼在一起
        if src_yaml_lines[i]['wav'] != curr_doc:
            src_for_hyp_docs.append({"doc_id": src_yaml_lines[i]['wav'], "src": ""})
            curr_doc = src_yaml_lines[i]['wav']
        src_for_hyp_docs[-1]["src"] += " " + src_text_lines[i]
    # -----------------------------------------------------------
    with open(instances, "r", encoding="utf-8") as f:
        instances_lines = [line.rstrip("\n") for line in f]
    
    hyp_dicts = []
    # [({source1, prediction_text1, ...}), ({source2, prediction_text2, ...}), ...]
    for j in range(len(instances_lines)):
        instance_dict = json.loads(instances_lines[j])
        hyp_text = instance_dict["prediction_text"]
        for src_for_hyp_doc in src_for_hyp_docs:
            # wav? 和 source? 都按 basename 匹配
            if src_for_hyp_doc["doc_id"] == os.path.basename(str(instance_dict["source"] or "")).replace("\\", "/"):
                hyp_dict = {
                    "src": src_for_hyp_doc["src"],
                    "tgt": hyp_text,
                    "sys_id": None,
                    "doc_id": src_for_hyp_doc["doc_id"],
                    "seg_id": j + 1
                }
                hyp_dicts.append(hyp_dict)
                break

    return ref_dicts, hyp_dicts

def instances_to_segale(
    src_txt: str = "data/input/ACL.6060.dev.en-xx.en.txt", 
    tgt_ref_txt: str = "data/input/ACL.6060.dev.en-xx.zh.txt", 
    src_segments_yaml: str = "data/input/ACL.ACLdev2023.en-xx.gold_segments.yaml", 
    instances: str = "data/output_qwen_asr3_debug/instances2.log",
    out_dir: str = "data/output_segale"
):
    ref_dicts, hyp_dicts = get_jsonl(
        src_txt, tgt_ref_txt, src_segments_yaml, instances
    )

    # 保存 jsonl
    out_path_ref = os.path.join(out_dir, "ref.jsonl")
    out_path_hyp = os.path.join(out_dir, "hyp.jsonl")
    os.makedirs(os.path.dirname(out_path_ref), exist_ok=True)
    with open(out_path_ref, "w", encoding="utf-8") as f:
        for obj in ref_dicts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(out_path_hyp), exist_ok=True)
    with open(out_path_hyp, "w", encoding="utf-8") as f:
        for obj in hyp_dicts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("saved:", out_path_ref, out_path_hyp)
# ============================================================================/

input_name = "input"
output_name = "output"
model_name = "volcengine"
input_version = "2_100ms"
output_version = "2_100ms_new"

def main():
    manifest = f"data/{output_name}_{model_name}_wav{input_version}/manifest.jsonl"
    tgt_language = "Chinese"
    output_dir_asr = f"data/{output_name}_{model_name}_asr{output_version}"
    batch_size = 10

    # print("\n" + "=" * 60)
    # print("Starting ASR...")
    # step1_asr(
    #     manifest=manifest,
    #     tgt_language=tgt_language,
    #     out_dir=output_dir_asr,
    #     batch_size=batch_size,
    # )
    # ### 释放显存
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # print("ASR finished.")

    output_dir_asr_ = output_dir_asr + "_"


    # add_char_spans_for_dir(
    #     f"data/{output_name}_{model_name}_asr{input_version}",
    #     output_dir_asr_
    # )

    src_segments_yaml = f"data/{input_name}/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    output_path_instances = os.path.join(output_dir_asr_, "instances.log")


    # asr_to_instances(
    #     s2s=True,
    #     yaml_file=src_segments_yaml,
    #     asr_dir=output_dir_asr_,
    #     output_file=output_path_instances,
    # )

    src_txt = f"data/{input_name}/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt = f"data/{input_name}/ACL.6060.dev.en-xx.zh.txt"
    output_dir_segale = f"data/{output_name}_{model_name}_segale{output_version}"

    # instances_to_segale(
    #     src_txt=src_txt, 
    #     tgt_ref_txt=tgt_ref_txt, 
    #     src_segments_yaml=src_segments_yaml, 
    #     instances=output_path_instances,
    #     out_dir=output_dir_segale
    # )

    task_lang = "zh"

    # print("\n" + "=" * 60)
    # print("Starting Segale...")
    # step2_segale(
    #     system_file=os.path.join(output_dir_segale, "hyp.jsonl"),
    #     ref_file=os.path.join(output_dir_segale, "ref.jsonl"),
    #     segmenter="spacy",
    #     task_lang=task_lang,
    #     proc_device="cuda",
    #     embedding_model= "sentence-transformers/LaBSE"  # "BAAI/bge-m3"
    # )
    # print("Segale finished.")

    segale_file = os.path.join(output_dir_segale, "hyp/aligned_spacy_hyp.jsonl")
    output_dir_longyaal = f"data/{output_name}_{model_name}_longyaal{output_version}"

    print("\n" + "=" * 60)
    print("Starting Longyaal...")
    step3_longyaal(
        yaml_file=src_segments_yaml,
        source_sentences_file=src_txt,
        instances_log=output_path_instances,
        segale_file=segale_file,
        output_folder=output_dir_longyaal,
        bleu_tokenizer=task_lang,
    )
    print("Longyaal finished.")

if __name__ == "__main__":
    main()