from typing import List, Tuple

from .utils import _qwen_units


# ============================================================
# # Segale src 和 yaml 的匹配
# ============================================================
# 优先的匹配方法：基于 src_ref_ids 的匹配
def _match_src_span_for_seg_by_ids(
    src_ref_ids: List[int],
    src_sent_list: List[Tuple[int, float, float, List[str]]],
) -> Tuple[int, int, float, float]:
    if not src_ref_ids:
        raise ValueError("src_ref_ids is empty")

    global_to_local = {t[0]: i for i, t in enumerate(src_sent_list)}

    indices = []
    for sid in src_ref_ids:
        g0 = int(sid) - 1
        if g0 in global_to_local:
            indices.append(global_to_local[g0])

    if not indices:
        raise ValueError(f"Invalid src_ref_ids: {src_ref_ids} (not in this doc's segments)")

    start_idx = min(indices)
    end_idx = max(indices)

    _, start_ms, _, _ = src_sent_list[start_idx]
    _, _, end_ms, _ = src_sent_list[end_idx]

    new_cursor_sent = end_idx + 1
    num_sent_used = end_idx - start_idx + 1
    return new_cursor_sent, num_sent_used, float(start_ms), float(end_ms)


# ---------------------------------------------------
# 兜底的匹配方法：基于文本内容的匹配
def _match_src_span_for_seg(
    segale_src_seg: str,
    src_sent_list: List[Tuple[int, float, float, List[str]]],
    list_j: int,
) -> Tuple[int, int, float, float]:
    target = _qwen_units(segale_src_seg, "english")
    if not target:
        _, s_ms, e_ms, _ = src_sent_list[list_j]
        return list_j + 1, 1, s_ms, e_ms

    acc: List[str] = []
    start_ms = None
    end_ms = None
    used = 0
    j = list_j

    while j < len(src_sent_list):
        _, s_ms, e_ms, toks = src_sent_list[j]
        if start_ms is None:
            start_ms = s_ms
        end_ms = e_ms
        acc.extend(toks)
        used += 1

        if acc == target:
            return j + 1, used, float(start_ms), float(end_ms)

        if len(acc) > len(target) and target:
            break
        j += 1

    _, s_ms, e_ms, _ = src_sent_list[list_j]
    return list_j + 1, 1, float(s_ms), float(e_ms)