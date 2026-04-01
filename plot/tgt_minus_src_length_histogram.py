import math
import os
from typing import Sequence, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import load_instances, build_rows, doc_id_matches


def _collect_diff_s(grouped, only_doc_ids: Optional[Sequence[str]] = None):
    """收集 (delay_span − recording_end) 并转为秒。"""
    values = []
    for doc_id, rows in grouped.items():
        if only_doc_ids is not None:
            if not any(doc_id_matches(doc_id, x) for x in only_doc_ids):
                continue
        for row in rows:
            dm = row.get("diff_ms_span")
            if dm is not None and not math.isnan(dm):
                values.append(dm / 1000.0)
    return values


def tgt_minus_src_length_histogram(
    instances: Sequence[str],
    tgt_langs: Sequence[str],
    src_lang: str = "En",
    only_doc_ids: Optional[Tuple[str, ...]] = ("110", "117"),
    out_png: str = "tgt_minus_src_length_histogram.png",
):
    """
    1 行 N 列直方图，每列对应一个 tgt_lang。
    instances 与 tgt_langs 一一对应，各自是 instances.resegmented.json 的路径。
    """
    assert len(instances) == len(tgt_langs), "instances 和 tgt_langs 长度必须一致"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
    })

    n = len(tgt_langs)
    fig, axes = plt.subplots(1, n, figsize=(6.8 * n, 6.6), squeeze=False, sharey=True)

    for i, (inst_path, tgt_lang) in enumerate(zip(instances, tgt_langs)):
        ax = axes[0][i]

        if not os.path.isfile(inst_path):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        else:
            grouped = build_rows(load_instances(inst_path))
            diff_s = _collect_diff_s(grouped, only_doc_ids)
            if diff_s:
                ax.hist(diff_s, bins=36, range=(-9.0, 9.0),
                        color="tab:blue", edgecolor="black", alpha=0.75)

        ax.set_xlim(-9.0, 9.0)
        ax.set_xlabel("Target − Source Duration Difference (s)")
        ax.set_title(f"{src_lang}-{tgt_lang}")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_ylabel("Frequency")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tgt_minus_src histogram to: {out_png}")

'''
python plot/tgt_minus_src_length_histogram.py
'''

if __name__ == "__main__":
    tgt_minus_src_length_histogram(
        instances=(
            "output/seed/en_zh/output_longyaal/instances.resegmented.json",
            "output/seed/en_de/output_longyaal/instances.resegmented.json",
            "output/seed/en_ja/output_longyaal/instances.resegmented.json",
        ),
        tgt_langs=("Zh", "De", "Ja"),
        src_lang="En",
        only_doc_ids=("110", "117"),
        out_png="output/seed/plot/tgt_minus_src_length_histogram.png",
    )
