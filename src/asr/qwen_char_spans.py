import json
import os
import unicodedata
from typing import Dict, List, Tuple


def _clip(text: str, limit: int = 200) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."



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