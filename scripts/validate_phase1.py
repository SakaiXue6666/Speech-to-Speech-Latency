import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {lineno} must be a JSON object")
            rows.append(row)
    return rows


def check_monotonic(ts_list: Iterable[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    prev_end = None
    for idx, ts in enumerate(ts_list):
        start = ts.get("start_time")
        end = ts.get("end_time")
        if start is None or end is None:
            errors.append(f"time_stamps[{idx}] missing start_time or end_time")
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            errors.append(f"time_stamps[{idx}] has non-numeric start_time/end_time")
            continue
        if start_f > end_f:
            errors.append(f"time_stamps[{idx}] has start_time > end_time")
        if prev_end is not None and start_f < prev_end:
            errors.append(f"time_stamps[{idx}] is not monotonic")
        prev_end = end_f
    return errors


def validate_manifest(path: Path) -> List[str]:
    errors: List[str] = []
    rows = read_jsonl(path)
    if not rows:
        errors.append("manifest is empty")
        return errors
    for idx, row in enumerate(rows):
        src = row.get("src")
        tgt = row.get("tgt")
        if not src:
            errors.append(f"manifest row {idx} missing src")
        if not tgt:
            errors.append(f"manifest row {idx} missing tgt")
            continue
        if not Path(tgt).exists():
            errors.append(f"manifest row {idx} tgt does not exist: {tgt}")
    return errors


def validate_asr_json(path: Path) -> List[str]:
    errors: List[str] = []
    data = read_json(path)
    if not isinstance(data, dict):
        return ["ASR JSON root must be an object"]
    if not isinstance(data.get("text", ""), str):
        errors.append("field 'text' must be a string")
    ts_list = data.get("time_stamps")
    if not isinstance(ts_list, list):
        errors.append("field 'time_stamps' must be a list")
        return errors
    errors.extend(check_monotonic(ts_list))
    text = data.get("text", "")
    for idx, ts in enumerate(ts_list):
        if not isinstance(ts, dict):
            errors.append(f"time_stamps[{idx}] must be an object")
            continue
        if "text" not in ts:
            errors.append(f"time_stamps[{idx}] missing text")
        char_start = ts.get("char_start")
        char_end = ts.get("char_end")
        if char_start is None and char_end is None:
            continue
        if char_start is None or char_end is None:
            errors.append(f"time_stamps[{idx}] char span is partially missing")
            continue
        try:
            s = int(char_start)
            e = int(char_end)
        except (TypeError, ValueError):
            errors.append(f"time_stamps[{idx}] char span is not integer-like")
            continue
        if s < 0 or e < 0:
            continue
        if s > e:
            errors.append(f"time_stamps[{idx}] has char_start > char_end")
        if e > len(text):
            errors.append(f"time_stamps[{idx}] char_end exceeds len(text)")
    return errors


def validate_jsonl_objects(path: Path, required_fields: Tuple[str, ...]) -> List[str]:
    errors: List[str] = []
    rows = read_jsonl(path)
    if not rows:
        errors.append("file is empty")
        return errors
    for idx, row in enumerate(rows):
        for field in required_fields:
            if field not in row:
                errors.append(f"row {idx} missing field '{field}'")
    return errors


def report(label: str, path: Path, errors: List[str]) -> int:
    if errors:
        print(f"[FAIL] {label}: {path}")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"[OK] {label}: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight Phase 1 artifact validator")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asr-json", action="append", type=Path, default=[])
    parser.add_argument("--ref-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--hyp-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--alignment-jsonl", action="append", type=Path, default=[])
    args = parser.parse_args()

    failures = 0

    if args.manifest:
        failures += report("manifest", args.manifest, validate_manifest(args.manifest))

    for path in args.asr_json:
        failures += report("asr-json", path, validate_asr_json(path))

    for path in args.ref_jsonl:
        failures += report("ref-jsonl", path, validate_jsonl_objects(path, ("doc_id", "tgt")))

    for path in args.hyp_jsonl:
        failures += report("hyp-jsonl", path, validate_jsonl_objects(path, ("doc_id", "tgt")))

    for path in args.alignment_jsonl:
        failures += report("alignment-jsonl", path, validate_jsonl_objects(path, ("doc_id",)))

    if failures:
        print(f"\nValidation finished with {failures} failing artifact(s).")
        return 1

    print("\nValidation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
