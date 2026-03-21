import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Iterable, Optional

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
            # 时间差：delay_span - source_length (ms)，用于直方图
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

# ---------------------------------------------------

def plot_ending_offset_delay(grouped, out_png: str, title: str):
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
    plot_ending_offset_delay(
        grouped,
        out_png,
        title="Per-doc last_delay - recording_end by segment",
    )

    print(f"Saved plot to: {out_png}")
    print(f"Saved csv to: {out_csv}")


# ---------------------------------------------------

def plot_delay_span_vs_source_length(grouped, out_png: str, title: str):
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
        xs = [
            int(row["seg_id"]) if str(row["seg_id"]).isdigit() else i + 1
            for i, row in enumerate(rows)
        ]
        ys_delay_span = [row["delay_span"] for row in rows]
        ys_source_length = [row["recording_end"] for row in rows]

        ax.plot(xs, ys_delay_span, marker="o", linewidth=1.5, markersize=3, label="target_length")
        ax.plot(xs, ys_source_length, marker="s", linewidth=1.5, markersize=3, label="source_length")
        ax.set_title(doc_id)
        ax.set_xlabel("seg_id")
        ax.set_ylabel("ms")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(n_docs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def delay_span_vs_source_length(
        instances: str,
        out_png: str,
):
    instances = load_instances(instances)
    grouped = build_rows(instances)
    plot_delay_span_vs_source_length(
        grouped,
        out_png,
        title="Per-doc delay span vs source_length by seg_id",
    )
    print(f"Saved span plot to: {out_png}")

# ---------------------------------------------------

def _doc_id_passes_filter(doc_id: str, only_doc_ids: Optional[Iterable[str]]) -> bool:
    """only_doc_ids 为 None 表示全要；否则可传完整 doc_id，或简写 \"110\"（匹配 acl-long.110）。"""
    if only_doc_ids is None:
        return True
    for x in only_doc_ids:
        if doc_id == x or f"acl-long.{x}." in doc_id:
            return True
    return False


def _flatten_diff_ms_from_grouped(
    grouped, only_doc_ids: Optional[Iterable[str]] = None
):
    """收集所有 segment 的 delay_span − recording_end（ms），用于直方图。"""
    diff_ms_span = []
    for doc_id in sorted(grouped):
        if not _doc_id_passes_filter(doc_id, only_doc_ids):
            continue
        for row in grouped[doc_id]:
            dm = row.get("diff_ms_span")
            if dm is not None and not math.isnan(dm):
                diff_ms_span.append(float(dm))
    return diff_ms_span


def plot_tgt_minus_src_histogram(
    grouped,
    out_png: str,
    title: str,
    src_lang: str,
    tgt_lang: str,
    only_doc_ids: Optional[Iterable[str]] = None,
):
    """
    每句一个值：delay_span − source_length，横轴为秒 (s)，默认显示范围 [-9, 9] s。
    delay_span = last_delay − first_delay；source_length 即 recording_end。
    y 轴左侧两条独立竖排文字：{src}_{tgt} 与 count (segments)，不要连成一条。
    """
    diff_ms_span = _flatten_diff_ms_from_grouped(grouped, only_doc_ids=only_doc_ids)
    # ms → s
    diff_s = [x / 1000.0 for x in diff_ms_span]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=0.20)
    if diff_s:
        # 只在 [-9, 9] s 内分箱，与 x 轴范围一致
        ax.hist(
            diff_s,
            bins=36,
            range=(-9.0, 9.0),
            color="tab:orange",
            edgecolor="black",
            alpha=0.75,
        )
    ax.set_xlim(-9.0, 9.0)
    ax.set_xlabel("target_length − source_length (s)")
    ax.set_ylabel("")
    # 左起第一条：竖排 src_tgt；紧挨右侧第二条：竖排 count (segments)
    ax.text(
        -0.10,
        0.5,
        f"{src_lang}-{tgt_lang}",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    ax.text(
        -0.055,
        0.5,
        "count (segments)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    fig.tight_layout(rect=[0.06, 0, 1, 1])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def tgt_minus_src_length_histogram(
    instances: str,
    out_png: str,
    src_lang: str,
    tgt_lang: str,
    only_doc_ids: Optional[Iterable[str]] = None,
):
    instances = load_instances(instances)
    grouped = build_rows(instances)
    plot_tgt_minus_src_histogram(
        grouped,
        out_png,
        title="Histogram: target_length − source_length (per segment, s), x∈[−9, 9]",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        only_doc_ids=only_doc_ids,
    )
    print(f"Saved length-diff histogram to: {out_png}")
