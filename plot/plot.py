import csv
import json
import math
import os
from collections import defaultdict
from typing import Iterable, Optional


def load_instances(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def to_number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_rows(instances):
    grouped = defaultdict(list)

    for item in instances:
        doc_id = str(item.get("doc_id", "unknown"))
        seg_id = item.get("seg_id", "")

        src = str(item.get("source", "") or "")
        tgt = str(item.get("prediction", "") or "")

        if not src.strip() or not tgt.strip():
            first_delay = math.nan
            last_delay = math.nan
            recording_end = math.nan
            delay_span = math.nan
            gap = math.nan
            diff_ms_span = math.nan
        else:
            delays = item.get("delays") or []
            first_delay = to_number(delays[0]) if delays else math.nan
            recording_end = to_number(item.get("source_length"))
            last_delay = to_number(delays[-1]) if delays else math.nan
            delay_span = (
                last_delay - first_delay
                if not math.isnan(last_delay) and not math.isnan(first_delay)
                else math.nan
            )
            gap = (
                last_delay - recording_end
                if not math.isnan(last_delay) and not math.isnan(recording_end)
                else math.nan
            )
            diff_ms_span = (
                delay_span - recording_end
                if not math.isnan(delay_span) and not math.isnan(recording_end)
                else math.nan
            )

        grouped[doc_id].append(
            {
                "doc_id": doc_id,
                "seg_id": seg_id,
                "first_delay": first_delay,
                "last_delay": last_delay,
                "recording_end": recording_end,
                "delay_span": delay_span,
                "gap": gap,
                "diff_ms_span": diff_ms_span,
            }
        )

    for doc_id in grouped:
        grouped[doc_id].sort(
            key=lambda row: int(row["seg_id"]) if str(row["seg_id"]).isdigit() else 10**9
        )

    return grouped


def save_csv(grouped, out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "seg_id",
                "first_delay",
                "last_delay",
                "recording_end",
                "delay_span",
                "gap",
                "diff_ms_span",
            ],
        )
        writer.writeheader()
        for doc_id in sorted(grouped):
            for row in grouped[doc_id]:
                writer.writerow(row)


def doc_id_matches(doc_id: str, short_id: str) -> bool:
    """Return True if doc_id (e.g. '2022.acl-long.110.wav') matches the abbreviated id (e.g. '110')."""
    return doc_id == short_id or f"acl-long.{short_id}." in doc_id
