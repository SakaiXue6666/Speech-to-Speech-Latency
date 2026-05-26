# -*- coding: utf-8 -*-

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .utils import _norm, _qwen_units



logger = logging.getLogger(__name__)


# ============================================================
# Segale tgt 和 ASR unit 的匹配
# ============================================================
# 优先的匹配方法：基于 Segale tgt 与 ASR units 在原文字符区间的匹配
def _match_tgt_units_for_seg_by_raw_char_span(
    raw_starts_ends: List[Tuple[int, int]],
    compact_to_raw: List[int],
    seg_char_start: int,
    seg_char_end: int,
    cursor_unit: int,
) -> Optional[Tuple[int, int]]:
    """
    根据 seg 的字符区间 [seg_char_start, seg_char_end)，
    在 raw unit char spans 里找所有“有重叠”的 unit，
    再映射回 compact unit span [c_start, c_end)。

    判定重叠条件：
        unit_s < seg_char_end AND unit_e > seg_char_start
    """
    if seg_char_start < 0 or seg_char_end <= seg_char_start:
        return None
    if not raw_starts_ends or not compact_to_raw:
        return None

    # 从当前 compact cursor 对应的 raw 下标开始往后找
    raw_cursor = (
        compact_to_raw[cursor_unit]
        if 0 <= cursor_unit < len(compact_to_raw)
        else 0
    )

    matched_raw = []

    for raw_i in range(raw_cursor, len(raw_starts_ends)):
        unit_s, unit_e = raw_starts_ends[raw_i]

        # 跳过无效 span
        if unit_s < 0 or unit_e <= unit_s:
            continue

        # 若 unit 已经完全在 seg 后面了，可以停
        if unit_s >= seg_char_end:
            break

        # 区间相交：这个 unit 属于当前 seg
        if unit_s < seg_char_end and unit_e > seg_char_start:
            matched_raw.append(raw_i)

    # logger.info(
    #     "[raw-overlap] seg=(%d,%d) raw_cursor=%d matched_raw=%s",
    #     seg_char_start,
    #     seg_char_end,
    #     raw_cursor,
    #     matched_raw[:20],
    # )

    if not matched_raw:
        return None

    raw_start_found = matched_raw[0]
    raw_end_found = matched_raw[-1]

    # raw -> compact
    c_start = next(
        (c for c, r in enumerate(compact_to_raw) if r >= raw_start_found),
        None,
    )
    c_end_inclusive = next(
        (c for c in range(len(compact_to_raw) - 1, -1, -1)
         if compact_to_raw[c] <= raw_end_found),
        None,
    )

    if c_start is None or c_end_inclusive is None or c_end_inclusive < c_start:
        return None

    # 不允许回退
    c_start = max(c_start, cursor_unit)
    if c_end_inclusive < c_start:
        return None

    return c_start, c_end_inclusive + 1


# ---------------------------------------------------
# 兜底的匹配方法：基于 Segale tgt 与 ASR units 的文本内容做匹配
def _fix_decimal_for_segale_text(s: str) -> str:
    """
    "84. 22" / "84 .22" / "84 . 22" -> "84.22"
    """
    s = _norm(s)
    return re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", s)


def _normalize_token_for_match(tok: str) -> str:
    """
    用于匹配时的 token 归一化：
    - NFKC
    - lower
    - 去掉标点 / 空白 / 控制字符
    """
    s = _norm(tok).lower().strip()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "Z", "C"):
            continue
        out.append(ch)
    return "".join(out)


def _is_latin_token(tok: str) -> bool:
    tok = (tok or "").strip()
    if not tok:
        return False
    return all(0x20 <= ord(c) < 0x4E00 for c in tok)


def _token_match_loose(n_tok: str, h_tok: str) -> bool:
    """
    更保守的 loose match：
    - 优先 exact / normalized exact
    - 对一般 token 不再随便用 startswith，避免把
      'transformer' 和 'transformerbased' 提前误判成 1:1
    - 只给 very short latin token 保留一点前缀容错（如 QED vs QEDQED）
    """
    n = (n_tok or "").strip()
    h = (h_tok or "").strip()
    if not n:
        return True

    if n == h:
        return True

    n_norm = _normalize_token_for_match(n)
    h_norm = _normalize_token_for_match(h)
    if not n_norm:
        return True

    if n_norm == h_norm:
        return True

    # 只对很短的拉丁 token 允许前缀容错，避免误吞 merge 场景
    if _is_latin_token(n) and _is_latin_token(h):
        if len(n_norm) <= 4 and len(h_norm) <= 12:
            if h_norm.startswith(n_norm) or n_norm.startswith(h_norm):
                return True

    return False


def _find_sublist(
    hay: List[str], needle: List[str], start: int
) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    if start >= len(hay):
        return None
    n = len(needle)
    for i in range(start, max(start, len(hay) - n) + 1):
        if hay[i:i + n] == needle:
            return i, i + n
    return None


def _find_sublist_loose(
    hay: List[str], needle: List[str], start: int
) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    if start >= len(hay):
        return None
    n = len(needle)
    for i in range(start, max(start, len(hay) - n) + 1):
        ok = True
        for k in range(n):
            if not _token_match_loose(needle[k], hay[i + k]):
                ok = False
                break
        if ok:
            return i, i + n
    return None


def _try_align_from_start(
    hay_norm: List[str],
    needle_norm: List[str],
    start_i: int,
    max_hay_skip: int = 3,
    max_needle_skip: int = 2,
) -> Optional[Tuple[int, int, float, int, int]]:
    """
    从 hay[start_i] 开始，做顺序受限对齐：
    优先级：
      1) exact 1:1
      2) 2 needle -> 1 hay
      3) 1 needle -> 2 hay
      4) loose/prefix 1:1
      5) 少量 skip
    """
    j = start_i
    k = 0
    used_hay_skip = 0
    used_needle_skip = 0
    matched_needle = 0

    while k < len(needle_norm) and j < len(hay_norm):
        hn = hay_norm[j]
        nn = needle_norm[k]

        if not hn:
            j += 1
            continue
        if not nn:
            k += 1
            continue

        # --------------------------------------------------
        # 1) exact 1:1
        # --------------------------------------------------
        if hn == nn:
            j += 1
            k += 1
            matched_needle += 1
            continue

        # --------------------------------------------------
        # 2) 2 needle -> 1 hay
        #    transformer + based -> transformerbased
        # --------------------------------------------------
        if k + 1 < len(needle_norm):
            nn2 = needle_norm[k] + needle_norm[k + 1]
            if nn2 and hn == nn2:
                j += 1
                k += 2
                matched_needle += 2
                continue

        # --------------------------------------------------
        # 3) 1 needle -> 2 hay
        # --------------------------------------------------
        if j + 1 < len(hay_norm):
            hn2 = hay_norm[j] + hay_norm[j + 1]
            if hn2 and hn2 == nn:
                j += 2
                k += 1
                matched_needle += 1
                continue

        # --------------------------------------------------
        # 4) loose/prefix 1:1
        #    放到 merge/split 后面，避免误吞
        # --------------------------------------------------
        if hn.startswith(nn) or nn.startswith(hn):
            # 对很长 token，prefix 太危险，限制一下长度差
            if abs(len(hn) - len(nn)) <= 2:
                j += 1
                k += 1
                matched_needle += 1
                continue

        # --------------------------------------------------
        # 5) 更宽一点的 merge/split
        # --------------------------------------------------
        if k + 1 < len(needle_norm):
            nn2 = needle_norm[k] + needle_norm[k + 1]
            if nn2 and (hn.startswith(nn2) or nn2.startswith(hn)):
                if abs(len(hn) - len(nn2)) <= 2:
                    j += 1
                    k += 2
                    matched_needle += 2
                    continue

        if j + 1 < len(hay_norm):
            hn2 = hay_norm[j] + hay_norm[j + 1]
            if hn2 and (hn2.startswith(nn) or nn.startswith(hn2)):
                if abs(len(hn2) - len(nn)) <= 2:
                    j += 2
                    k += 1
                    matched_needle += 1
                    continue

        # --------------------------------------------------
        # 6) skip
        # --------------------------------------------------
        if used_hay_skip < max_hay_skip:
            used_hay_skip += 1
            j += 1
            continue

        if used_needle_skip < max_needle_skip:
            used_needle_skip += 1
            k += 1
            continue

        break

    while k < len(needle_norm) and not needle_norm[k]:
        k += 1

    if k < len(needle_norm):
        return None

    consumed_hay = max(1, j - start_i)
    non_empty_needle = len([x for x in needle_norm if x])
    denom = max(non_empty_needle, consumed_hay)
    score = matched_needle / max(1, denom)

    return start_i, j, score, matched_needle, consumed_hay


def _find_sublist_merge_split_window(
    hay: List[str],
    needle: List[str],
    start: int,
    # window_size: int = 220,
    # min_score: float = 0.90,
    # max_hay_skip: int = 2,
    # max_needle_skip: int = 1,
    # max_start_drift: int = 160,
    window_size=280,
    min_score=0.86,
    max_hay_skip=3,
    max_needle_skip=2,
    max_start_drift=180,
) -> Optional[Tuple[int, int, float]]:
    """
    在 hay[start : start+window_size] 内，找一个更稳的局部对齐：
    - 顺序匹配
    - 支持 1:1 / 2:1 / 1:2
    - 允许少量 skip
    - 必须把 needle 基本吃完
    """
    if not hay or not needle or start >= len(hay):
        return None

    hay_norm = [_normalize_token_for_match(x) for x in hay]
    needle_norm = [_normalize_token_for_match(x) for x in needle]

    best = None
    scan_end = min(len(hay), start + window_size)

    for i in range(start, scan_end):
        if i - start > max_start_drift:
            break

        res = _try_align_from_start(
            hay_norm,
            needle_norm,
            i,
            max_hay_skip=max_hay_skip,
            max_needle_skip=max_needle_skip,
        )
        if res is None:
            continue

        s, e, score, matched_needle, consumed_hay = res

        # 再加一些安全阀
        non_empty_needle = len([x for x in needle_norm if x])
        if matched_needle < max(1, non_empty_needle - max_needle_skip):
            continue

        length_gap = abs(consumed_hay - max(1, non_empty_needle))
        if length_gap > max(6, non_empty_needle // 3):
            continue

        cand = (s, e, score)
        if best is None:
            best = cand
        else:
            # 先看分数，再看离 cursor 近不近
            if (score > best[2]) or (score == best[2] and s < best[0]):
                best = cand

        if score >= 0.98 and s == start:
            break

    if best is not None and best[2] >= min_score:
        return best
    return None


# ------------------------------------------------------------
# # normalize helpers for char-stream matching
# ------------------------------------------------------------

def _is_kept_char(ch: str) -> bool:
    if ch == "'":  # 单引号
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")  # 字母或数字


def _norm_char_stream(text: str) -> str:
    """
    把一段文本转成一个适合做字符流匹配的“规范串”："hello world!" -> "helloworld"
    - NFKC
    - 仅保留字母/数字/单引号
    - 英文转小写
    """
    text = _norm(text)
    return "".join(ch.lower() for ch in text if _is_kept_char(ch))


# ------------------------------------------------------------
# # data structures
# ------------------------------------------------------------

@dataclass
class CompactCharIndex:
    """
    基于 compact units 构建的字符流索引。

    norm_units:
        每个 compact unit 规范化后的字符串
    global_norm:
        全部 norm_units 直接拼接后的全局串
    global_pos_to_unit:
        global_norm 的每个字符对应哪个 compact unit
    global_pos_to_char_in_unit:
        global_norm 的每个字符在对应 compact unit 里的偏移
    unit_start_global:
        每个 compact unit 在 global_norm 中的起始位置
    unit_end_global:
        每个 compact unit 在 global_norm 中的结束位置（开区间）
    """
    norm_units: List[str]
    global_norm: str
    global_pos_to_unit: List[int]
    global_pos_to_char_in_unit: List[int]
    unit_start_global: List[int]
    unit_end_global: List[int]


@dataclass
class UnitMatchResult:
    """
    统一的匹配结果。

    new_cursor_unit:
        下一个 segment 从哪个 compact unit 开始继续找
    u_start, u_end:
        命中的 compact unit span，开区间 [u_start, u_end)
    start_offset:
        在 u_start 这个 unit 的 norm token 内起始偏移
    end_offset:
        在 u_end-1 这个 unit 的 norm token 内结束偏移（开区间）
    method:
        strict / loose / skip_latin / char_exact / merge_split / empty / not_found
    score:
        对 strict/loose/char_exact 可固定为 1.0；fuzzy 用真实分数
    """
    new_cursor_unit: int
    u_start: int
    u_end: int
    start_offset: int = 0
    end_offset: int = 0
    method: str = ""
    score: float = 0.0


# ------------------------------------------------------------
# # build char index over compact units
# ------------------------------------------------------------

def _build_compact_char_index(asr_units: List[str]) -> CompactCharIndex:
    norm_units: List[str] = []
    global_parts: List[str] = []
    global_pos_to_unit: List[int] = []
    global_pos_to_char_in_unit: List[int] = []
    unit_start_global: List[int] = []
    unit_end_global: List[int] = []

    g = 0
    for i, unit in enumerate(asr_units):
        nu = _norm_char_stream(unit)
        norm_units.append(nu)
        unit_start_global.append(g)

        for j, ch in enumerate(nu):
            global_parts.append(ch)
            global_pos_to_unit.append(i)
            global_pos_to_char_in_unit.append(j)
            g += 1

        unit_end_global.append(g)

    return CompactCharIndex(
        norm_units=norm_units,
        global_norm="".join(global_parts),
        global_pos_to_unit=global_pos_to_unit,
        global_pos_to_char_in_unit=global_pos_to_char_in_unit,
        unit_start_global=unit_start_global,
        unit_end_global=unit_end_global,
    )


def _unit_cursor_to_char_cursor(index: CompactCharIndex, unit_i: int) -> int:
    if unit_i <= 0:
        return 0
    if not index.unit_start_global:
        return 0
    if unit_i >= len(index.unit_start_global):
        return len(index.global_norm)
    return index.unit_start_global[unit_i]


# ------------------------------------------------------------
# # char-stream exact matcher over compact units
# ------------------------------------------------------------

def _find_by_char_stream_exact(
    seg_text: str,
    char_index: CompactCharIndex,
    unit_i: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    在 compact units 构建出的 global normalized char stream 里做 exact match。

    返回:
        (u_start, u_end, start_offset, end_offset)

    其中:
    - unit span 是 [u_start, u_end)
    - start_offset 是 u_start 这个 unit 内的起始偏移
    - end_offset 是 u_end-1 这个 unit 内的结束偏移（开区间）
    """
    needle_norm = _norm_char_stream(seg_text)
    if not needle_norm:
        return None

    if not char_index.global_norm:
        return None

    char_cursor = _unit_cursor_to_char_cursor(char_index, unit_i)
    pos = char_index.global_norm.find(needle_norm, char_cursor)
    if pos == -1:
        return None

    start = pos
    end = pos + len(needle_norm)  # 开区间

    start_unit = char_index.global_pos_to_unit[start]
    start_offset = char_index.global_pos_to_char_in_unit[start]

    last = end - 1
    end_unit_inclusive = char_index.global_pos_to_unit[last]
    end_offset = char_index.global_pos_to_char_in_unit[last] + 1  # 开区间

    return start_unit, end_unit_inclusive + 1, start_offset, end_offset


# ------------------------------------------------------------
# # raw/compact span mapping
# ------------------------------------------------------------

def _compact_span_to_raw_span(
    compact_to_raw: List[int],
    c_start: int,
    c_end: int,
) -> Tuple[int, int]:
    """
    compact [c_start, c_end) -> raw [r_start, r_end)
    用于切 delays_all / elapsed_all
    """
    if c_start >= c_end or not compact_to_raw:
        return 0, 0

    if c_start < 0 or c_end > len(compact_to_raw):
        return 0, 0

    r_start = compact_to_raw[c_start]
    r_end = compact_to_raw[c_end - 1] + 1
    return r_start, r_end


# ------------------------------------------------------------
# # main matcher
# 依赖以下外部函数/对象已经在你的工程里存在：
#   _fix_decimal_for_segale_text
#   _find_sublist
#   _find_sublist_loose
#   _find_sublist_merge_split_window
#   _is_latin_token
#   logger
# ------------------------------------------------------------

def _match_tgt_units_for_seg(
    segale_tgt_seg: str,
    tgt_asr_units: List[str],
    unit_i: int,
    doc_id: str,
    seg_id: str,
    char_index: Optional[CompactCharIndex] = None,
) -> UnitMatchResult:
    """
    返回 compact unit 空间下的匹配结果。

    优先级：
    1) strict
    2) loose
    3) skip leading latin tokens
    4) char-stream exact
    5) merge/split local alignment
    """
    segale_tgt_seg = _fix_decimal_for_segale_text(segale_tgt_seg)
    needle = _qwen_units(segale_tgt_seg, "chinese")

    if not needle:
        return UnitMatchResult(
            new_cursor_unit=unit_i,
            u_start=unit_i,
            u_end=unit_i,
            start_offset=0,
            end_offset=0,
            method="empty",
            score=1.0,
        )

    # --------------------------------------------------------
    # 1) strict
    # --------------------------------------------------------
    hit = _find_sublist(tgt_asr_units, needle, unit_i)
    if hit is not None:
        u_start, u_end = hit
        return UnitMatchResult(
            new_cursor_unit=u_end,
            u_start=u_start,
            u_end=u_end,
            start_offset=0,
            end_offset=len(_norm_char_stream(tgt_asr_units[u_end - 1])) if u_end > u_start else 0,
            method="strict",
            score=1.0,
        )

    # --------------------------------------------------------
    # 2) loose
    # --------------------------------------------------------
    hit = _find_sublist_loose(tgt_asr_units, needle, unit_i)
    if hit is not None:
        u_start, u_end = hit
        return UnitMatchResult(
            new_cursor_unit=u_end,
            u_start=u_start,
            u_end=u_end,
            start_offset=0,
            end_offset=len(_norm_char_stream(tgt_asr_units[u_end - 1])) if u_end > u_start else 0,
            method="loose",
            score=1.0,
        )

    # --------------------------------------------------------
    # 3) skip leading latin tokens
    # --------------------------------------------------------
    max_skip = min(3, len(needle) - 1)
    for skip in range(1, max_skip + 1):
        if not _is_latin_token(needle[skip - 1]):
            break

        sub = needle[skip:]
        if not sub:
            break

        h = _find_sublist(tgt_asr_units, sub, unit_i)
        if h is None:
            h = _find_sublist_loose(tgt_asr_units, sub, unit_i)

        if h is not None:
            u_start, u_end = h
            return UnitMatchResult(
                new_cursor_unit=u_end,
                u_start=u_start,
                u_end=u_end,
                start_offset=0,
                end_offset=len(_norm_char_stream(tgt_asr_units[u_end - 1])) if u_end > u_start else 0,
                method=f"skip_latin_{skip}",
                score=1.0,
            )

    # --------------------------------------------------------
    # 4) char-stream exact over compact units
    # --------------------------------------------------------
    if char_index is not None:
        char_hit = _find_by_char_stream_exact(segale_tgt_seg, char_index, unit_i)
        if char_hit is not None:
            u_start, u_end, start_offset, end_offset = char_hit
            try:
                logger.info(
                    "[match][char-exact] doc=%s seg_id=%s cursor=%d start=%d end=%d so=%d eo=%d needle_len=%d seg_tgt=%s",
                    doc_id,
                    seg_id,
                    unit_i,
                    u_start,
                    u_end,
                    start_offset,
                    end_offset,
                    len(needle),
                    repr(segale_tgt_seg[:200]),
                )
            except Exception:
                pass

            return UnitMatchResult(
                new_cursor_unit=u_end,
                u_start=u_start,
                u_end=u_end,
                start_offset=start_offset,
                end_offset=end_offset,
                method="char_exact",
                score=1.0,
            )

    # --------------------------------------------------------
    # 5) merge/split local alignment
    # --------------------------------------------------------
    fuzzy = _find_sublist_merge_split_window(
        tgt_asr_units,
        needle,
        unit_i,
        # window_size=240,
        # min_score=0.90,
        # max_hay_skip=2,
        # max_needle_skip=1,
        # max_start_drift=160,
        window_size=280,
        min_score=0.86,
        max_hay_skip=3,
        max_needle_skip=2,
        max_start_drift=180,
    )
    if fuzzy is not None:
        f_start, f_end, f_score = fuzzy
        try:
            logger.info(
                "[match][merge-split] doc=%s seg_id=%s cursor=%d start=%d end=%d score=%.3f needle_len=%d seg_tgt=%s",
                doc_id,
                seg_id,
                unit_i,
                f_start,
                f_end,
                f_score,
                len(needle),
                repr(segale_tgt_seg[:200]),
            )
        except Exception:
            pass

        return UnitMatchResult(
            new_cursor_unit=f_end,
            u_start=f_start,
            u_end=f_end,
            start_offset=0,
            end_offset=len(_norm_char_stream(tgt_asr_units[f_end - 1])) if f_end > f_start else 0,
            method="merge_split",
            score=float(f_score),
        )

    # --------------------------------------------------------
    # not found
    # --------------------------------------------------------
    try:
        logger.warning(
            "[skip][tgt-not-found] doc=%s seg_id=%s cursor=%d needle_len=%d "
            "seg_tgt=%s needle=%s asr_ctx=%s",
            doc_id,
            seg_id,
            unit_i,
            len(needle),
            repr(segale_tgt_seg[:200]),
            needle[:20],
            tgt_asr_units[unit_i:min(len(tgt_asr_units), unit_i + 60)],
        )
    except Exception:
        pass

    return UnitMatchResult(
        new_cursor_unit=unit_i,
        u_start=unit_i,
        u_end=unit_i,
        start_offset=0,
        end_offset=0,
        method="not_found",
        score=0.0,
    )


# ------------------------------------------------------------
# optional helper: full pipeline usage
# ------------------------------------------------------------

def build_target_matcher_context(tgt_asr_units: List[str]) -> CompactCharIndex:
    """
    在处理整篇文档前调用一次即可。
    """
    return _build_compact_char_index(tgt_asr_units)


def match_segment_and_map_raw(
    segale_tgt_seg: str,
    tgt_asr_units: List[str],
    compact_to_raw: List[int],
    unit_i: int,
    doc_id: str,
    seg_id: str,
    char_index: Optional[CompactCharIndex] = None,
):
    """
    一个方便上层直接调用的包装函数。
    返回:
        match_result, raw_span
    """
    mr = _match_tgt_units_for_seg(
        segale_tgt_seg=segale_tgt_seg,
        tgt_asr_units=tgt_asr_units,
        unit_i=unit_i,
        doc_id=doc_id,
        seg_id=seg_id,
        char_index=char_index,
    )
    raw_span = _compact_span_to_raw_span(compact_to_raw, mr.u_start, mr.u_end)
    return mr, raw_span
