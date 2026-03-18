# -*- coding: utf-8 -*-

import logging
from typing import List, Dict, Any, Tuple, Optional

from argparse import ArgumentParser

import unicodedata
import yaml
import json
import os

from pipeline_qwen3_forcealign_tokenizer2 import Qwen3ForceAlignTokenizer
from .metrics import Instance, SacreBLEUScorer, YAALScorer, EndingOffsetScorer, evaluate_instances
from .matching import (
    _compact_span_to_raw_span,
    _match_src_span_for_seg,
    _match_src_span_for_seg_by_ids,
    _match_tgt_units_for_seg,
    _match_tgt_units_for_seg_by_raw_char_span,
    _norm_char_stream,
    build_target_matcher_context,
)

qwen_tok = Qwen3ForceAlignTokenizer()


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _qwen_units(text: str, language: str) -> List[str]:
    return qwen_tok.encode_timestamp(_norm(text), language)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INF = float("inf")


def _to_ms(x: Any) -> float:
    if x is None:
        return INF
    try:
        v = float(x)
    except Exception:
        return INF
    if v <= 0:
        return INF
    if v < 1e4:
        return v * 1000.0
    return v


# ============================================================
# segale / yaml / instances
# ============================================================
def _load_segale(segale_jsonl: str) -> Dict[str, List[Dict[str, Any]]]:
    '''
    return:
    {
        "2022.acl-long.268.wav": [
            {
                "doc_id": "2022.acl-long.268.wav",
                "seg_id": "0",
                "src": "...",
                "tgt": "...",
                "ref": "...",
                "src_ref_ids": [...],
                "mt_char_start": 123,
                "mt_char_end": 145,
                ...
            },
            ...
        ],
        "<next doc_id>": ...
    }
    '''
    mp: Dict[str, List[Dict[str, Any]]] = {}
    with open(segale_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            doc_id = r["doc_id"]
            mp.setdefault(doc_id, []).append(r)

    for k in mp:
        mp[k].sort(
            key=lambda x: int(x["seg_id"]) if str(x["seg_id"]).isdigit() else 10**9
        )
    return mp


def _load_yaml_and_source_sentences(
    yaml_file: str, source_txt: str
) -> Dict[str, List[Tuple[int, float, float, List[str]]]]:
    '''
    return:
    {
        "2022.acl-long.268.wav": [
            (0, off_ms, end_ms, toks),
            (1, off_ms, end_ms, toks),
            ...
        ],
        "<next doc_id>": ...
    }
    '''
    with open(yaml_file, "r", encoding="utf-8") as f:
        segs = yaml.safe_load(f) or []
    with open(source_txt, "r", encoding="utf-8") as f:
        src_lines = [x.rstrip("\n") for x in f]

    assert len(segs) == len(src_lines), "ref_segments.yaml ????? == source.txt ???"

    mp: Dict[str, List[Tuple[int, float, float, List[str]]]] = {}
    for i, (seg, sline) in enumerate(zip(segs, src_lines)):
        doc_id = seg["wav"]
        off_ms = float(seg["offset"]) * 1000.0
        dur_ms = float(seg["duration"]) * 1000.0
        end_ms = off_ms + dur_ms
        toks = _qwen_units(sline, "english")
        mp.setdefault(doc_id, []).append((i, off_ms, end_ms, toks))
    return mp


def _load_instances_log(instances_log: str) -> Dict[str, Dict[str, Any]]:
    '''
    return:
    {
        "2022.acl-long.268.wav": {
            ...原始日志字段...,
            "_doc_id": "2022.acl-long.268.wav",
            "_source_length_ms": 737440.0,
            "_delays": [...],
            "_durations": [...],
            "_elapsed": [...],
            "_intervals": [...],
            "_raw_units": [...],
            "_raw_starts_ends": [(s0, e0), (s1, e1), ...],
            "_asr_units": [...],              # compact 后，去掉空 token
            "_compact_to_raw": [0, 1, 3, 4], # compact 索引 -> raw 索引
            "_compact_starts_ends": [...]
        },
        "<next doc_id>": ...
    }
    '''
    mp = {}
    with open(instances_log, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            h = json.loads(line)

            src = h.get("source")
            src_path = (src[0] if isinstance(src, list) else src) or ""
            doc_id = os.path.basename(str(src_path)).replace("\\", "/")

            delays = h.get("delays") or []
            durations = h.get("durations") or []
            elapsed = h.get("elapsed") or []
            intervals = h.get("intervals") or []

            if (not elapsed) or (len(elapsed) != len(delays)):
                elapsed = list(delays)

            m = min(len(delays), len(elapsed))
            delays = list(delays[:m])
            elapsed = list(elapsed[:m])
            if durations:
                durations = list(durations[:m])
            if intervals:
                intervals = list(intervals[:m])

            ts = list((h.get("prediction_units") or [])[:m])
            starts_ends = list((h.get("prediction_unit_char_starts_ends") or [])[:m])

            raw_units: List[str] = []
            raw_starts_ends: List[Tuple[int, int]] = []
            compact_units: List[str] = []
            compact_to_raw: List[int] = []
            compact_starts_ends: List[Tuple[int, int]] = []

            for raw_idx in range(m):
                x = ts[raw_idx] if raw_idx < len(ts) else ""
                u = _norm(x)
                se = starts_ends[raw_idx] if raw_idx < len(starts_ends) else (-1, -1)

                raw_units.append(u)
                raw_starts_ends.append(tuple(se))

                if u and u.strip():
                    compact_to_raw.append(raw_idx)
                    compact_units.append(u)
                    compact_starts_ends.append(tuple(se))

            h["_doc_id"] = doc_id
            h["_source_length_ms"] = _to_ms(h.get("source_length", INF))
            h["_delays"] = delays
            h["_durations"] = durations
            h["_elapsed"] = elapsed
            h["_intervals"] = intervals

            h["_raw_units"] = raw_units
            h["_raw_starts_ends"] = raw_starts_ends
            h["_asr_units"] = compact_units
            h["_compact_to_raw"] = compact_to_raw
            h["_compact_starts_ends"] = compact_starts_ends

            mp[doc_id] = h
    return mp


def step3_longyaal(
    yaml_file: str,
    source_sentences_file: str,
    instances_log: str,
    segale_file: str,
    output_folder: str,
    bleu_tokenizer: str,
):
    os.makedirs(output_folder, exist_ok=True)

    segale_by_doc = _load_segale(segale_file)
    sent_by_doc = _load_yaml_and_source_sentences(yaml_file, source_sentences_file)
    inst_by_doc = _load_instances_log(instances_log)

    instances: List[Instance] = []
    instances_dict: List[Dict[str, Any]] = []

    global_idx = 0

    for doc_id, segs in segale_by_doc.items():
        if doc_id not in sent_by_doc:
            logger.warning(f"[skip] doc missing in yaml/source.txt: {doc_id}")
            continue
        if doc_id not in inst_by_doc:
            logger.warning(f"[skip] doc missing in instances.log: {doc_id}")
            continue

        inst = inst_by_doc[doc_id]

        asr_units = inst.get("_asr_units", [])
        compact_to_raw = inst.get("_compact_to_raw", [])

        raw_starts_ends = inst.get("_raw_starts_ends", [])

        if not asr_units:
            logger.warning(f"[skip] doc missing asr units in instances.log: {doc_id}")
            continue

        delays_all = inst["_delays"]
        durations_all = inst["_durations"]
        elapsed_all = inst["_elapsed"]

        # compact 空间和 raw 时间空间长度本来就不必相等；
        # 这里只检查 raw delays / elapsed
        m_raw = min(len(delays_all), len(elapsed_all))
        delays_all = delays_all[:m_raw]
        elapsed_all = elapsed_all[:m_raw]
        if durations_all:
            durations_all = durations_all[:m_raw]

        # compact_to_raw 的最后一个 raw 索引不能超出 raw 时间长度
        valid_compact_len = 0
        for ridx in compact_to_raw:
            if 0 <= ridx < m_raw:
                valid_compact_len += 1
            else:
                break

        if valid_compact_len != len(compact_to_raw):
            logger.warning(
                "[warn] compact_to_raw out of raw range for %s: compact=%d valid=%d raw=%d",
                doc_id, len(compact_to_raw), valid_compact_len, m_raw
            )
            compact_to_raw = compact_to_raw[:valid_compact_len]
            asr_units = asr_units[:valid_compact_len]

        char_index = build_target_matcher_context(asr_units)

        cursor_unit = 0
        cursor_sent = 0
        doc_sent_list = sent_by_doc[doc_id]


        '''
        src=""/src_ref_ids=[]出现时不算letency和quality，因为是over-translation；而tgt=""也会存在，这时也不算letency和quality因为是under-translation
        下一步，默认我现在不再使用_match_src_span_for_seg和 _match_tgt_units_for_seg，默认_match_src_span_for_seg_by_ids和_match_tgt_units_for_seg_by_raw_char_spa处理不好的他们也处理不好，要怎么改，不影响原有计算
        '''
        for seg_idx, seg in enumerate(segs):
            seg_src = seg.get("src", "")
            seg_tgt = seg.get("tgt", "")
            seg_ref = seg.get("ref", "")
            src_ref_ids = seg.get("src_ref_ids") or []
            char_start = seg.get("mt_char_start", -1)
            char_end = seg.get("mt_char_end", -1)
            
            # over-translation
            if not seg_src.strip():
                instances_dict.append({
                    "index": global_idx,
                    "doc_id": doc_id,
                    "seg_id": seg.get("seg_id"),
                    "source": "",
                    "reference": seg_ref,
                    "prediction": seg_tgt,
                    "source_length": None,
                    "delays": [],
                    "elapsed": [],
                    "recording_end": None,
                    "_seg_start_ms": None,
                    "_seg_end_ms": None,
                    "_compact_u_start": None,
                    "_compact_u_end": None,
                    "_raw_u_start": None,
                    "_raw_u_end": None,
                    "_match_method": "skip_over_translation",
                    "_start_offset": None,
                    "_end_offset": None,
                    "raw_units": [],
                })
                global_idx += 1
                logger.info(
                    "[skip][over-translation] doc=%s seg_id=%s src_empty=%s src_ref_ids_empty=%s",
                    doc_id, seg.get("seg_id"), not seg_src.strip(), not src_ref_ids,
                )
                continue

            # under-translation
            if not seg_tgt.strip():
                instances_dict.append({
                    "index": global_idx,
                    "doc_id": doc_id,
                    "seg_id": seg.get("seg_id"),
                    "source": seg_src,
                    "reference": seg_ref,
                    "prediction": seg_tgt,
                    "source_length": None,
                    "delays": [],
                    "elapsed": [],
                    "recording_end": None,
                    "_seg_start_ms": None,
                    "_seg_end_ms": None,
                    "_compact_u_start": None,
                    "_compact_u_end": None,
                    "_raw_u_start": None,
                    "_raw_u_end": None,
                    "_match_method": "skip_under_translation",
                    "_start_offset": None,
                    "_end_offset": None,
                    "raw_units": [],
                })
                global_idx += 1
                logger.info(
                    "[skip][under-translation] doc=%s seg_id=%s",
                    doc_id, seg.get("seg_id"),
                )
                continue


            # 3.1 segale 与 src yaml, src, ref 对应
            if src_ref_ids:
                try:
                    # 优先使用
                    cursor_sent, _, seg_start_ms, seg_end_ms = _match_src_span_for_seg_by_ids(
                        src_ref_ids, doc_sent_list
                    )
                except Exception as e:
                    logger.warning(
                        "[fallback][src-ref-ids-invalid] doc=%s seg_id=%s src_ref_ids=%s err=%s",
                        doc_id, seg.get("seg_id"), src_ref_ids, e
                    )
                    list_j = max(0, min(cursor_sent - 1, len(doc_sent_list) - 1))
                    cursor_sent, _, seg_start_ms, seg_end_ms = _match_src_span_for_seg(
                        seg_src, doc_sent_list, list_j
                    )
            else:
                list_j = max(0, min(cursor_sent - 1, len(doc_sent_list) - 1))
                cursor_sent, _, seg_start_ms, seg_end_ms = _match_src_span_for_seg(
                    seg_src, doc_sent_list, list_j
                )

            seg_source_len_ms = max(1.0, seg_end_ms - seg_start_ms)

            # 3.2 segale 与 tgt units 对应
            cursor_unit_before = cursor_unit

            # 优先使用
            hit = _match_tgt_units_for_seg_by_raw_char_span(
                raw_starts_ends=raw_starts_ends,
                compact_to_raw=compact_to_raw,
                seg_char_start=char_start,
                seg_char_end=char_end,
                cursor_unit=cursor_unit,
            )

            if hit is not None:
                c_start, c_end = hit
                cursor_unit = c_end
                match_method = "char_span_from_segale"
                start_offset = 0
                end_offset = len(_norm_char_stream(asr_units[c_end - 1])) if c_end > c_start else 0

            else:
                # char_span 未命中则不再做 _match_tgt_units_for_seg，直接视为无匹配
                logger.warning(
                    "[char_span_miss] doc=%s seg_id=%s char_start=%s char_end=%s cursor_unit=%d -> 无匹配，跳过",
                    doc_id, seg.get("seg_id"), char_start, char_end, cursor_unit_before,
                )
                mr = _match_tgt_units_for_seg(
                    segale_tgt_seg=seg_tgt,
                    tgt_asr_units=asr_units,
                    unit_i=cursor_unit,
                    doc_id=doc_id,
                    seg_id=str(seg.get("seg_id")),
                    char_index=char_index,
                )
                cursor_unit = mr.new_cursor_unit
                c_start = mr.u_start
                c_end = mr.u_end
                match_method = mr.method
                start_offset = mr.start_offset
                end_offset = mr.end_offset

            # compact -> raw
            r_start, r_end = _compact_span_to_raw_span(compact_to_raw, c_start, c_end)

            seg_delays = delays_all[r_start:r_end]
            seg_elapsed = elapsed_all[r_start:r_end]

            if not seg_delays:
                logger.warning(
                    "[skip][empty-matched-units] doc=%s seg_id=%s cursor_unit=%d c_start=%d c_end=%d r_start=%d r_end=%d "
                    "len(asr_units)=%d len(delays_all)=%d len(elapsed_all)=%d seg_tgt_len=%d",
                    doc_id,
                    seg.get("seg_id"),
                    cursor_unit,
                    c_start,
                    c_end,
                    r_start,
                    r_end,
                    len(asr_units),
                    len(delays_all),
                    len(elapsed_all),
                    len((seg_tgt or "").strip()),
                )
                continue

            # 3.3 转成 segment 相对时间
            seg_delays_rel = [max(0.0, float(d) - seg_start_ms) for d in seg_delays]
            seg_elapsed_rel = [max(0.0, float(e) - seg_start_ms) for e in seg_elapsed]

            new_seg_dict = {
                "index": global_idx,
                "doc_id": doc_id,
                "seg_id": seg.get("seg_id"),

                "source": seg_src,
                "reference": seg_ref,

                "source_length": seg_source_len_ms,
                "delays": seg_delays_rel,
                "elapsed": seg_elapsed_rel,
                "recording_end": seg_source_len_ms,

                "_compact_u_start": c_start,
                "_compact_u_end": c_end,
                "_raw_u_start": r_start,
                "_raw_u_end": r_end,
                "_seg_start_ms": seg_start_ms,
                "_seg_end_ms": seg_end_ms,

                "_match_method": match_method,
                "_start_offset": start_offset,
                "_end_offset": end_offset,

                "prediction": seg.get("tgt", ""),
                "raw_units": inst["_raw_units"][r_start:r_end]
            }

            instances_dict.append(new_seg_dict)
            instances.append(Instance(new_seg_dict, latency_unit="word"))
            global_idx += 1

    with open(
        os.path.join(output_folder, "instances.resegmented.json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(instances_dict, ensure_ascii=False, indent=2) + "\n")

    scores = evaluate_instances(instances, bleu_tokenizer)
    with open(
        os.path.join(output_folder, "scores.resegmented.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\t".join(scores.keys()) + "\n")
        f.write("\t".join([f"{v:.4f}" for v in scores.values()]) + "\n")

    logger.info(f"Done. segments={len(instances)}. Output -> {output_folder}")


# ============================================================
# CLI
# ============================================================
def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="data/input/ACL.ACLdev2023.en-xx.gold_segments.yaml",
    )
    parser.add_argument(
        "--source_sentences_file",
        type=str,
        default="data/input/ACL.6060.dev.en-xx.en.txt",
    )
    parser.add_argument(
        "--instances_log",
        type=str,
        default="data/output_qwen_asr3/instances.log",
    )
    parser.add_argument(
        "--segale_file",
        type=str,
        default="data/output_segale3/hyp/aligned_spacy_hyp.jsonl",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/output_longyaal3_fixed",
    )
    parser.add_argument(
        "--bleu_tokenizer",
        type=str,
        default="zh",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    step3_longyaal(
        yaml_file=args.yaml_file,
        source_sentences_file=args.source_sentences_file,
        instances_log=args.instances_log,
        segale_file=args.segale_file,
        output_folder=args.output_folder,
        bleu_tokenizer=args.bleu_tokenizer,
    )


if __name__ == "__main__":
    main()
