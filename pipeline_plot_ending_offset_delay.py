import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
        delays = item.get("delays") or []
        recording_end = to_number(item.get("source_length"))
        last_delay = to_number(delays[-1]) if delays else math.nan
        gap = last_delay - recording_end if not math.isnan(last_delay) and not math.isnan(recording_end) else math.nan

        grouped[doc_id].append(
            {
                "doc_id": doc_id,
                "seg_id": seg_id,
                "last_delay": last_delay,
                "recording_end": recording_end,
                "gap": gap,
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
            fieldnames=["doc_id", "seg_id", "last_delay", "recording_end", "gap"],
        )
        writer.writeheader()
        for doc_id in sorted(grouped):
            for row in grouped[doc_id]:
                writer.writerow(row)


def plot_grouped(grouped, out_png: str, title: str):
    doc_ids = sorted(grouped)
    n_docs = len(doc_ids)
    n_cols = 2 if n_docs > 1 else 1
    n_rows = math.ceil(n_docs / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 3.6 * n_rows),
        squeeze=False,
        sharey=True,
    )

    for idx, doc_id in enumerate(doc_ids):
        ax = axes[idx // n_cols][idx % n_cols]
        rows = grouped[doc_id]
        xs = list(range(1, len(rows) + 1))
        ys = [row["gap"] for row in rows]
        seg_labels = [str(row["seg_id"]) for row in rows]

        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3)
        ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(doc_id)
        ax.set_xlabel("Segment order")
        ax.set_ylabel("last_delay - recording_end (ms)")
        ax.grid(True, alpha=0.3)

        if len(xs) <= 25:
            ax.set_xticks(xs)
            ax.set_xticklabels(seg_labels, rotation=45, ha="right", fontsize=8)

    for idx in range(n_docs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ending_offset_delay(
        instances: str,
        out_png: str,
        out_csv: str,
):
    instances = load_instances(instances)
    grouped = build_rows(instances)
    save_csv(grouped, out_csv)
    plot_grouped(
        grouped,
        out_png,
        title="Per-doc last_delay - recording_end by segment",
    )

    print(f"Saved plot to: {out_png}")
    print(f"Saved csv to: {out_csv}")
