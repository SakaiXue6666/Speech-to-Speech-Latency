import os
from typing import Sequence, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import load_instances, build_rows, save_csv, doc_id_matches


def ending_offset_delay(
    instances: Sequence[str],
    tgt_langs: Sequence[str],
    doc_ids: Sequence[str],
    src_lang: str = "en",
    out_png: str = "ending_offset_delay.png",
    out_csv: Optional[str] = None,
):
    """
    Grid plot: rows = tgt_langs, columns = doc_ids.
    instances and tgt_langs correspond one-to-one; each element is a path to instances.resegmented.json.
    """
    assert len(instances) == len(tgt_langs), "instances and tgt_langs must have the same length"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    n_rows = len(tgt_langs)
    n_cols = len(doc_ids)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        sharey=True,
    )

    for r, (inst_path, tgt_lang) in enumerate(zip(instances, tgt_langs)):
        if not os.path.isfile(inst_path):
            for c in range(n_cols):
                axes[r][c].text(0.5, 0.5, "no data", transform=axes[r][c].transAxes, ha="center", va="center")
                if r == 0:
                    axes[r][c].set_title(doc_ids[c])
            axes[r][0].set_ylabel(f"{src_lang}-{tgt_lang}")
            continue

        grouped = build_rows(load_instances(inst_path))

        if out_csv:
            csv_path = out_csv.replace(".csv", f"_{src_lang}_{tgt_lang}.csv")
            save_csv(grouped, csv_path)

        for c, did in enumerate(doc_ids):
            ax = axes[r][c]
            rows = []
            for full_doc_id, doc_rows in grouped.items():
                if doc_id_matches(full_doc_id, did):
                    rows.extend(doc_rows)

            if not rows:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            else:
                rows.sort(key=lambda x: int(x["seg_id"]) if str(x["seg_id"]).isdigit() else 10**9)
                xs = list(range(1, len(rows) + 1))
                ys = [row["gap"] for row in rows]
                ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3)
                ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
                if len(xs) <= 25:
                    seg_labels = [str(row["seg_id"]) for row in rows]
                    ax.set_xticks(xs)
                    ax.set_xticklabels(seg_labels, rotation=45, ha="right", fontsize=7)

            if r == 0:
                ax.set_title(did)
            if r == n_rows - 1:
                ax.set_xlabel("Segment order")
            if c == 0:
                ax.set_ylabel(f"{src_lang}-{tgt_lang}\nlast_delay − rec_end (ms)")

            ax.grid(True, alpha=0.3)

    fig.suptitle("Ending Offset Delay", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ending_offset_delay plot to: {out_png}")


if __name__ == "__main__":
    ending_offset_delay(
        instances=(
            "output/seed/en_zh/output_longyaal/instances.resegmented.json",
            "output/seed/en_de/output_longyaal/instances.resegmented.json",
            "output/seed/en_ja/output_longyaal/instances.resegmented.json",
        ),
        tgt_langs=("zh", "de", "ja"),
        doc_ids=("110", "117", "268", "367", "590"),
        out_png="output/seed/plot/ending_offset_delay.png",
        out_csv="output/seed/plot/ending_offset_delay.csv",
    )
