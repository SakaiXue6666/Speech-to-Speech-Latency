import html
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


NULL_ALIGNMENT_TYPES = {"over_translation", "under_translation"}


def classify_alignment(instance: Dict[str, Any]) -> str:
    explicit = str(instance.get("null_alignment_type") or "")
    if explicit:
        if explicit not in NULL_ALIGNMENT_TYPES:
            raise ValueError(f"unsupported null alignment type: {explicit}")
        return explicit
    if not str(instance.get("source") or "").strip():
        return "over_translation"
    if not str(instance.get("prediction") or "").strip():
        return "under_translation"
    return "normal"


def latency_eligible(instance: Dict[str, Any]) -> bool:
    return (
        classify_alignment(instance) == "normal"
        and bool(instance.get("delays"))
        and bool(instance.get("elapsed"))
        and instance.get("source_length") is not None
    )


def quality_input_row(instance: Dict[str, Any]) -> Dict[str, Any]:
    category = classify_alignment(instance)
    return {
        "index": instance.get("index"),
        "doc_id": instance.get("doc_id"),
        "seg_id": instance.get("seg_id"),
        "src": instance.get("source", ""),
        "mt": instance.get("prediction", ""),
        "ref": instance.get("reference", ""),
        "null_alignment_type": "" if category == "normal" else category,
        "score_override": 0.0 if category in NULL_ALIGNMENT_TYPES else None,
        "latency_eligible": latency_eligible(instance),
    }


def summarize_alignments(instances: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    rows = list(instances)
    categories = Counter(classify_alignment(row) for row in rows)
    return {
        "segments": len(rows),
        "valid_segments": categories["normal"],
        "latency_segments": sum(latency_eligible(row) for row in rows),
        "over_translation_alignments": categories["over_translation"],
        "under_translation_alignments": categories["under_translation"],
        "null_alignments": (
            categories["over_translation"] + categories["under_translation"]
        ),
    }


def _escaped(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _offset(instance: Dict[str, Any], timestamp_key: str, first: bool) -> Optional[float]:
    if not latency_eligible(instance):
        return None
    timestamps = instance[timestamp_key]
    timestamp = timestamps[0] if first else timestamps[-1]
    return float(timestamp) - float(instance["source_length"])


def _number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def render_sentence_details(
    instances: List[Dict[str, Any]],
    summary: Dict[str, int],
) -> str:
    cards = []
    for instance in instances:
        category = classify_alignment(instance)
        category_label = category.replace("_", "-")
        cards.append(
            f"<details class='case' data-category='{category}'>"
            "<summary>"
            f"{_escaped(instance.get('doc_id'))} · segment {_escaped(instance.get('seg_id'))} · "
            f"{_escaped(category_label)} · {_escaped(instance.get('_match_method'))}"
            "</summary>"
            f"<p><strong>Source:</strong> {_escaped(instance.get('source'))}</p>"
            f"<p><strong>Reference:</strong> {_escaped(instance.get('reference'))}</p>"
            f"<p><strong>Hypothesis:</strong> {_escaped(instance.get('prediction'))}</p>"
            "<dl>"
            f"<dt>Latency eligible</dt><dd>{str(latency_eligible(instance)).lower()}</dd>"
            f"<dt>First speech offset (ms)</dt><dd>{_number(_offset(instance, 'elapsed', True))}</dd>"
            f"<dt>Ending offset (ms)</dt><dd>{_number(_offset(instance, 'elapsed', False))}</dd>"
            f"<dt>Quality score override</dt><dd>{'0.0' if category in NULL_ALIGNMENT_TYPES else '-'}</dd>"
            "</dl>"
            "</details>"
        )
    tabs = "".join(
        (
            f"<button type='button' class='filter-tab{' active' if category == 'all' else ''}' "
            f"data-filter='{category}' aria-selected='{'true' if category == 'all' else 'false'}'>"
            f"{label} <span>{count}</span></button>"
        )
        for category, label, count in (
            ("all", "All", summary["segments"]),
            ("normal", "Normal", summary["valid_segments"]),
            (
                "under_translation",
                "Under-translation",
                summary["under_translation_alignments"],
            ),
            (
                "over_translation",
                "Over-translation",
                summary["over_translation_alignments"],
            ),
        )
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>SEGALE sentence details</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#20242a;background:#fafbfc;max-width:1200px}}
p{{line-height:1.45}}.note{{background:#fff8db;border-left:4px solid #ca8a04;padding:10px}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#fafbfc;padding:10px 0}}
.filter-tab{{border:1px solid #aeb7c2;background:#fff;padding:8px 12px;border-radius:6px;cursor:pointer;font-weight:600}}
.filter-tab span{{color:#59636f;font-weight:400}}.filter-tab.active{{background:#20242a;color:#fff;border-color:#20242a}}
.filter-tab.active span{{color:#e5e7eb}}details{{margin:10px 0;background:#fff;border:1px solid #d8dde3;border-radius:6px;padding:10px}}
summary{{cursor:pointer;font-weight:600}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:5px 12px}}dt{{font-weight:600}}dd{{margin:0}}
.case[hidden]{{display:none}}
</style></head><body>
<h1>SEGALE sentence details</h1>
<p class='note'>Structural null alignments remain visible and receive a quality score override of 0.0. Latency is undefined for null alignments and for normal alignments without a matched target-speech timeline.</p>
<div class='tabs' role='tablist' aria-label='SEGALE case category'>{tabs}</div>
{''.join(cards)}
<script>
document.querySelectorAll('.filter-tab').forEach((button) => {{
  button.addEventListener('click', () => {{
    const filter = button.dataset.filter;
    document.querySelectorAll('.filter-tab').forEach((item) => {{
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', selected ? 'true' : 'false');
    }});
    document.querySelectorAll('.case').forEach((item) => {{
      item.hidden = filter !== 'all' && item.dataset.category !== filter;
    }});
  }});
}});
</script>
</body></html>"""


def write_evaluation_report(
    instances: List[Dict[str, Any]],
    output_folder: str,
) -> Dict[str, int]:
    os.makedirs(output_folder, exist_ok=True)
    summary = summarize_alignments(instances)
    with open(
        os.path.join(output_folder, "alignment_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(
        os.path.join(output_folder, "quality_inputs.jsonl"),
        "w",
        encoding="utf-8",
    ) as handle:
        for instance in instances:
            json.dump(quality_input_row(instance), handle, ensure_ascii=False)
            handle.write("\n")
    with open(
        os.path.join(output_folder, "sentence_details.html"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(render_sentence_details(instances, summary))
    return summary
