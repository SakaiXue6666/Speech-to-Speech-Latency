# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import unicodedata
from multiprocessing import Pool
from statistics import mean
from typing import Any, Dict, List, Tuple

import yaml
from sacrebleu.metrics.bleu import BLEU

from pipeline_plot_ending_offset_delay import ending_offset_delay

logger = logging.getLogger(__name__)

# ---------- 原 LongYAAL/softsegmenter 内联常量与类 ----------
INF = float("inf")
PUNCT = set([".", "!", "?", ",", ";", ":", "-", "(", ")"])
CHINESE_PUNCT = set(["。", "！", "？", "，", "；", "：", "—", "（", "）"])
JAPAN_PUNCT = set(["。", "！", "？", "，", "；", "：", "ー", "（", "）"])
ALL_PUNCT = PUNCT | CHINESE_PUNCT | JAPAN_PUNCT


class Match:
    MATCH = 0
    DELETE = 1
    INSERT = 2
    NONE = 3


class Word:
    def __init__(
        self,
        text,
        delay,
        *,
        seq_id=None,
        elapsed=None,
        main=True,
        original_str=None,
        recording_length=None,
        unit_index=None,
    ):
        self.text = text
        self.delay = delay
        self.seq_id = seq_id
        self.elapsed = elapsed
        self.main = main
        self.original = original_str
        self.recording_length = recording_length
        self.unit_index = unit_index  # 在 doc 内的 unit 下标，用于从 MT 文本按 char 区间切片

    def __repr__(self):
        return f"Word(text={self.text!r}, delay={self.delay}, seq_id={self.seq_id}, recording_length={self.recording_length})"


class Instance:
    def __init__(self, info: Dict[str, Any], latency_unit: str = "word"):
        for key, value in info.items():
            setattr(self, key, value)
        self.reference = info.get("reference", "")
        self.prediction = info.get("prediction", "")
        self.latency_unit = latency_unit
        self.metrics = {}

    @property
    def reference_length(self) -> int:
        return self._string_to_len(self.reference, self.latency_unit)

    @staticmethod
    def _string_to_len(string: str, latency_unit: str) -> int:
        if latency_unit == "word":
            return len(string.split())
        if latency_unit == "char":
            return len(string.strip())
        raise ValueError(f"Unknown latency unit: {latency_unit}")


def _align_metric(ref_word, hyp_word, char_level):
    ref_text = ref_word.text
    hyp_text = hyp_word.text
    ref_t = ref_text in PUNCT
    hyp_t = hyp_text in PUNCT
    if ref_t ^ hyp_t:
        return -INF
    if char_level:
        return float(ref_text == hyp_text)
    ref_set = set(ref_text)
    hyp_set = set(hyp_text)
    inter = len(ref_set & hyp_set)
    union = len(ref_set) + len(hyp_set) - inter
    return (inter / union) if union else 0.0


def _align_sequences(seq1, seq2, char_level):
    n, m = len(seq1) + 1, len(seq2) + 1
    dp = [[0] * m for _ in range(n)]
    dp_back = [[Match.NONE] * m for _ in range(n)]
    for i in range(n):
        dp[i][0], dp_back[i][0] = 0, Match.DELETE
    for j in range(m):
        dp[0][j], dp_back[0][j] = 0, Match.INSERT
    dp[0][0], dp_back[0][0] = 0, Match.MATCH
    for i in range(1, n):
        for j in range(1, m):
            match = dp[i - 1][j - 1] + _align_metric(seq1[i - 1], seq2[j - 1], char_level)
            delete, insert = dp[i - 1][j], dp[i][j - 1]
            dp[i][j] = max(match, delete, insert)
            if dp[i][j] == match:
                dp_back[i][j] = Match.MATCH
            elif dp[i][j] == delete:
                dp_back[i][j] = Match.DELETE
            else:
                dp_back[i][j] = Match.INSERT
    aligned_1, aligned_2 = [], []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if dp_back[i][j] == Match.MATCH:
            aligned_1.append(seq1[i - 1])
            aligned_2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp_back[i][j] == Match.DELETE:
            aligned_1.append(seq1[i - 1])
            aligned_2.append(None)
            i -= 1
        elif dp_back[i][j] == Match.INSERT:
            aligned_1.append(None)
            aligned_2.append(seq2[j - 1])
            j -= 1
        else:
            break
    aligned_1.reverse()
    aligned_2.reverse()
    return aligned_1, aligned_2


def _process_alignment(ref_words, hyp_words, char_level):
    assert len(ref_words) == len(hyp_words)

    def get_next_non_none_ref(i):
        while i < len(ref_words) and ref_words[i] is None:
            i += 1
        return (i, ref_words[i]) if i < len(ref_words) else (i, None)

    new_hyp_words = []
    matched_ref_words = []  # 与 new_hyp_words 一一对应，用于写 alignment_tokens
    last_ref, nexti = None, 0
    for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
        if ref is None and i >= nexti and hyp is not None:
            nexti, next_ref = get_next_non_none_ref(i)
            assert next_ref is not None or last_ref is not None
            next_score = _align_metric(hyp, next_ref, char_level) if next_ref else -INF
            prev_score = _align_metric(hyp, last_ref, char_level) if last_ref else -INF
            if next_score > prev_score:
                ref = next_ref
            else:
                ref = last_ref
                nexti = i
            logger.debug(
                "aligned %s to %s (next %.2f vs prev %.2f)",
                getattr(hyp, "text", None),
                getattr(ref, "text", None),
                next_score,
                prev_score,
            )
        if ref is not None and i >= nexti:
            last_ref = ref
        if hyp is not None and ref is not None:
            hyp.seq_id = ref.seq_id
            new_hyp_words.append(hyp)
            matched_ref_words.append(ref)
    return new_hyp_words, matched_ref_words


def _process_audio(args):
    i, ref, hyp, char_level = args
    aligned_ref, aligned_hyp = _align_sequences(ref, hyp, char_level)
    return _process_alignment(aligned_ref, aligned_hyp, char_level)  # (new_hyp_words, matched_ref_words)


def _align_words(ref_words, hyp_words, char_level):
    """返回 (new_segmentation, ref_segmentation)。ref_segmentation[idx] 为该 segment 参与对齐的 ref token 列表（Word.text）。"""
    assert len(ref_words) == len(hyp_words)
    new_segmentation = {}
    ref_segmentation = {}
    for inst_ref in ref_words:
        for ref in inst_ref:
            new_segmentation[ref.seq_id] = []
            ref_segmentation[ref.seq_id] = []
    args_list = [
        (i, ref, hyp, char_level)
        for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words))
    ]
    new_hyp_all = []
    matched_ref_all = []
    with Pool() as pool:
        results = pool.map(_process_audio, args_list)
    assert len(results) == len(ref_words)
    for result in results:
        new_hyp_words, matched_ref_words = result
        new_hyp_all.extend(new_hyp_words)
        matched_ref_all.extend(matched_ref_words)
    for word, ref_word in zip(new_hyp_all, matched_ref_all):
        if word.main:
            new_segmentation[word.seq_id].append(word)
            ref_segmentation[word.seq_id].append(ref_word)
    return new_segmentation, ref_segmentation


def _get_segmentation_order(segmentation):
    order = []
    for seg in segmentation:
        if not order or order[-1] != seg["wav"]:
            order.append(seg["wav"])
    return order


def _fix_elapsed(words):
    new_elapsed = []
    for i, word in enumerate(words):
        if i > 0:
            new_elapsed.append(word.elapsed - words[i - 1].elapsed + words[i - 1].delay)
        else:
            new_elapsed.append(word.elapsed)
    for i, word in enumerate(words):
        word.elapsed = new_elapsed[i] if i == 0 else max(new_elapsed[i], words[i - 1].elapsed)
    return words


def _normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)


# ---------- 与 pipeline_main_new (25-193) 一致的 norm + 索引：用于 ref unit 在句内定位 ----------
def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _norm_char_stream_with_mapping(text: str) -> Tuple[str, List[int]]:
    """与 pipeline_main_new 一致：norm 流及 norm[i] -> raw 下标。"""
    raw = unicodedata.normalize("NFKC", text or "")
    norm_chars: List[str] = []
    norm_to_raw: List[int] = []
    for raw_idx, ch in enumerate(raw):
        if _is_kept_char(ch):
            norm_chars.append(ch.lower())
            norm_to_raw.append(raw_idx)
    return "".join(norm_chars), norm_to_raw


def _norm_unit_text(text: str) -> str:
    raw = unicodedata.normalize("NFKC", text or "")
    return "".join(ch.lower() for ch in raw if _is_kept_char(ch))


def _ref_units_with_spans(
    full_text: str,
    ref_unitizer_language: str,
    char_level: bool,
) -> List[Tuple[str, int, int]]:
    """Ref 句 full_text（已 NFKC）按 Qwen 或 char/空格拆成 units，用 pipeline_main_new 的 norm+find 得到每个 unit 的 (char_start, char_end)。"""
    full_norm, norm_to_raw = _norm_char_stream_with_mapping(full_text)
    if char_level:
        return [(c, i, i + 1) for i, c in enumerate(full_text)]
    try:
        from pipeline_qwen3_forcealign_tokenizer2 import Qwen3ForceAlignTokenizer
        qwen = Qwen3ForceAlignTokenizer()
        units = qwen.encode_timestamp(full_text, ref_unitizer_language)
    except Exception:
        units = full_text.split()
    if not units:
        return []
    result = []
    norm_cursor = 0
    for unit in units:
        unit_norm = _norm_unit_text(unit)
        if not unit_norm:
            continue
        pos = full_norm.find(unit_norm, norm_cursor)
        if pos == -1:
            pos = full_norm.find(unit_norm)
        if pos == -1:
            continue
        norm_start = pos
        norm_end = pos + len(unit_norm)
        if norm_end > len(norm_to_raw):
            continue
        raw_start = norm_to_raw[norm_start]
        raw_end = norm_to_raw[norm_end - 1] + 1
        result.append((unit, raw_start, raw_end))
        norm_cursor = norm_end
    return result


def _ref_tokens_with_punct(full_text: str, units_with_spans: List[Tuple[str, int, int]]) -> List[str]:
    """在 units 之间用索引取标点/空隙作为 token，得到 [unit1, punct1, unit2, punct2, ...]。纯空格不加入（de/en 等不把空格当标点）。"""
    if not units_with_spans:
        return []
    tokens = []
    prev_end = 0
    for unit_text, s, e in units_with_spans:
        gap = full_text[prev_end:s]
        # 仅当 gap 含非空白字符时才加入（标点如 ,.；等），纯空格不加入
        if gap and not gap.isspace():
            tokens.append(gap.strip() or gap)
        tokens.append(unit_text)
        prev_end = e
    return tokens


def _debug_print_ref_intermediates(seg_id: int, full_text: str, units_with_spans: List, ref_tokens: List, head: int = 50):
    """打印 ref 中间产物：Qwen/单位+索引、加标点后的 token 序列。"""
    if seg_id > 0:
        return
    print("\n" + "=" * 60)
    print("[debug] REF 中间产物 (仅打印 seg_id=0)")
    print("  ref_sentence(norm) 前80字:", repr(full_text[:80]) if full_text else "")
    print("  Qwen/单位+索引 (units_with_spans) 前%d 项:" % min(head, len(units_with_spans)))
    for j, (u, s, e) in enumerate(units_with_spans[:head]):
        print("    [%d] %r -> (%d, %d)" % (j, u, s, e))
    if len(units_with_spans) > head:
        print("    ... 共 %d 个 unit" % len(units_with_spans))
    print("  加标点后 ref_tokens 前%d 项:" % min(head, len(ref_tokens)))
    for j, t in enumerate(ref_tokens[:head]):
        print("    [%d] %r" % (j, t))
    if len(ref_tokens) > head:
        print("    ... 共 %d 个 token" % len(ref_tokens))
    print("=" * 60 + "\n")


def _hyp_tokens_with_punct(
    hyp_text: str,
    units: List[str],
    delays: List[float],
    elapsed: List[float],
    starts_ends: List[Tuple[int, int]],
) -> List[Tuple[str, float, float, Any]]:
    """Hyp 已有 units + 在 hyp text 的 (char_start, char_end)。中间补标点，标点 delay 用前一个 unit。返回 [(token_text, delay, elapsed, unit_index_or_None), ...]。"""
    if not units or len(delays) != len(units):
        return []
    n = len(units)
    use_spans = len(starts_ends) == n
    tokens = []
    prev_end = 0
    for i in range(n):
        if use_spans:
            s, e = int(starts_ends[i][0]), int(starts_ends[i][1])
        else:
            s = prev_end
            e = min(prev_end + len(units[i]), len(hyp_text))
        if prev_end < len(hyp_text) and s <= len(hyp_text):
            gap = hyp_text[prev_end:s]
            # 仅当 gap 含非空白字符时才加入（标点），纯空格不加入（de/en 等）
            if gap and not gap.isspace():
                d = delays[i - 1] if i > 0 else delays[0]
                el = elapsed[i - 1] if i > 0 and i - 1 < len(elapsed) else (elapsed[0] if elapsed else d)
                tokens.append((gap.strip() or gap, d, el, None))
        tokens.append((units[i], delays[i], elapsed[i] if i < len(elapsed) else delays[i], i))
        prev_end = e
    return tokens


def _debug_print_hyp_intermediates(doc_idx: int, hyp_text: str, units: List, token_triples: List, head: int = 50):
    """打印 hyp 中间产物：原始 units、加标点后的 (token, delay, elapsed, unit_index)。"""
    if doc_idx > 0:
        return
    print("\n" + "=" * 60)
    print("[debug] HYP 中间产物 (仅打印 doc_idx=0)")
    print("  hyp_text 前80字:", repr(hyp_text[:80]) if hyp_text else "")
    print("  原始 units 前%d 个:" % min(head, len(units)))
    for j, u in enumerate(units[:head]):
        print("    [%d] %r" % (j, u))
    if len(units) > head:
        print("    ... 共 %d 个 unit" % len(units))
    print("  加标点后 token_triples 前%d 项 (text, delay, elapsed, unit_ix):" % min(head, len(token_triples)))
    for j, (tt, d, e, uix) in enumerate(token_triples[:head]):
        print("    [%d] %r delay=%.0f unit_index=%s" % (j, tt, d, uix))
    if len(token_triples) > head:
        print("    ... 共 %d 个 token" % len(token_triples))
    print("=" * 60 + "\n")


def _load_reference_unit_punct(
    yaml_file: str,
    ref_sentences_file: str,
    ref_unitizer_language: str,
    char_level: bool,
    offset_delays: bool,
):
    """Ref：与 longyaal 一致用 Qwen 拆 unit（不用 char_level 按字切），用 pipeline_main_new 方式获索引，再补标点；每段得到 ref token 序列。"""
    with open(yaml_file, "r", encoding="utf-8") as f:
        segmentation = yaml.safe_load(f) or []
    for seg in segmentation:
        seg["duration"] = float(seg.get("duration", 0)) * 1000
        seg["offset"] = float(seg.get("offset", 0)) * 1000
    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    assert len(segmentation) == len(reference_sentences)
    words = []
    for i, (segment, ref_sentence) in enumerate(zip(segmentation, reference_sentences)):
        if i == 0 or segmentation[i - 1]["wav"] != segment["wav"]:
            first_offset = segment["offset"] if offset_delays else 0
            words.append([])
        if offset_delays:
            segment["offset"] -= first_offset
        delay = segment["offset"]
        full_text = unicodedata.normalize("NFKC", (ref_sentence or "").strip().lower())
        # ref 与 longyaal 一致：始终用 Qwen（nagisa/空格等）切 unit，不按 char_level 按字切
        units_with_spans = _ref_units_with_spans(full_text, ref_unitizer_language, char_level=False)
        ref_tokens = _ref_tokens_with_punct(full_text, units_with_spans)
        if not ref_tokens and full_text:
            ref_tokens = [full_text.strip()]
        _debug_print_ref_intermediates(i, full_text, units_with_spans, ref_tokens)
        words[-1].extend([Word(t, delay, seq_id=i) for t in ref_tokens])
    return words, segmentation, reference_sentences


def _load_hypothesis_unit_punct(
    hypothesis_file: str,
    segmentation_order: List[str],
):
    """Hyp：已有 units + delays + 在 hyp text 的索引；中间补标点作为 token，标点用前 unit 的 delay。"""
    hypotheses = {}
    source_lengths = {}
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        for line in f:
            h = json.loads(line.strip())
            name = os.path.basename(h["source"][0])
            assert name in segmentation_order, f"Missing hypothesis for {name}"
            assert name not in hypotheses, f"Duplicate hypothesis for {name}"
            source_lengths[name] = h.get("source_length", INF)
            hypotheses[name] = h
    assert len(hypotheses) == len(segmentation_order)
    hypotheses = [hypotheses[segmentation_order[i]] for i in range(len(segmentation_order))]
    source_lengths = [source_lengths[segmentation_order[i]] for i in range(len(segmentation_order))]
    words = []
    doc_mt_info = []
    for h, l in zip(hypotheses, source_lengths):
        hyp_text = h.get("prediction_text", "") or h.get("prediction", "")
        if not hyp_text and h.get("prediction_units"):
            hyp_text = " ".join(str(x) for x in h["prediction_units"])
        units = list(h.get("prediction_units") or [])
        delays = list(h.get("delays") or [])
        elapsed = list(h.get("elapsed") or delays)
        if len(elapsed) != len(delays):
            elapsed = list(delays)
        assert len(units) == len(delays), f"units vs delays: {len(units)} vs {len(delays)}"
        starts_ends = list(h.get("prediction_unit_char_starts_ends") or [])
        if len(starts_ends) != len(units):
            starts_ends = []
        doc_mt_info.append({
            "prediction_text": h.get("prediction_text", ""),
            "prediction_unit_char_starts_ends": starts_ends,
        })
        token_triples = _hyp_tokens_with_punct(hyp_text, units, delays, elapsed, starts_ends)
        _debug_print_hyp_intermediates(len(words), hyp_text, units, token_triples)
        instance_words = [
            Word(tt, d, original_str=tt, elapsed=e, recording_length=l, unit_index=uix)
            for tt, d, e, uix in token_triples
        ]
        words.append(_fix_elapsed(instance_words))
    return words, doc_mt_info


# Ref：用 Qwen tokenizer 把整句切成 units；Hyp：用 hypothesis 的 prediction_units + delays，不再额外 tokenize
def _load_reference_qwen_units(
    yaml_file: str,
    ref_sentences_file: str,
    ref_unitizer_language: str,
    offset_delays: bool,
):
    from pipeline_qwen3_forcealign_tokenizer2 import Qwen3ForceAlignTokenizer

    with open(yaml_file, "r", encoding="utf-8") as f:
        segmentation = yaml.safe_load(f) or []
    for seg in segmentation:
        seg["duration"] = float(seg.get("duration", 0)) * 1000
        seg["offset"] = float(seg.get("offset", 0)) * 1000

    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    assert len(segmentation) == len(reference_sentences)

    qwen_tok = Qwen3ForceAlignTokenizer()
    words = []
    for i, (segment, ref_sentence) in enumerate(zip(segmentation, reference_sentences)):
        if i == 0 or segmentation[i - 1]["wav"] != segment["wav"]:
            first_offset = segment["offset"] if offset_delays else 0
            words.append([])
        if offset_delays:
            segment["offset"] -= first_offset
        ref_sentence = ref_sentence.strip().lower()
        units = qwen_tok.encode_timestamp(ref_sentence, ref_unitizer_language)
        delay = segment["offset"]
        words[-1].extend([Word(unit, delay, seq_id=i) for unit in units])
    return words, segmentation, reference_sentences


def _load_hypothesis_prediction_units(
    hypothesis_file: str,
    segmentation_order: List[str],
):
    """返回 (words_per_doc, doc_mt_info_per_doc)。
    doc_mt_info_per_doc 与 segmentation_order 同序，每项为 {"prediction_text": str, "prediction_unit_char_starts_ends": list}。
    """
    hypotheses = {}
    source_lengths = {}
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        for line in f:
            h = json.loads(line.strip())
            name = os.path.basename(h["source"][0])
            assert name in segmentation_order, f"Missing hypothesis for {name}"
            assert name not in hypotheses, f"Duplicate hypothesis for {name}"
            source_lengths[name] = h.get("source_length", INF)
            hypotheses[name] = h
    assert len(hypotheses) == len(segmentation_order)
    hypotheses = [hypotheses[segmentation_order[i]] for i in range(len(segmentation_order))]
    source_lengths = [source_lengths[segmentation_order[i]] for i in range(len(segmentation_order))]
    words = []
    doc_mt_info = []
    for h, l in zip(hypotheses, source_lengths):
        units = list(h.get("prediction_units") or [])
        delays = list(h.get("delays") or [])
        elapsed = list(h.get("elapsed") or delays)
        if len(elapsed) != len(delays):
            elapsed = list(delays)
        assert len(units) == len(delays), f"units vs delays: {len(units)} vs {len(delays)}"
        assert len(units) == len(elapsed), f"units vs elapsed: {len(units)} vs {len(elapsed)}"
        starts_ends = list(h.get("prediction_unit_char_starts_ends") or [])
        if len(starts_ends) != len(units):
            starts_ends = []
        doc_mt_info.append({
            "prediction_text": h.get("prediction_text", ""),
            "prediction_unit_char_starts_ends": starts_ends,
        })
        instance_words = [
            Word(unit, d, original_str=unit, elapsed=e, recording_length=l, unit_index=i)
            for i, (unit, d, e) in enumerate(zip(units, delays, elapsed))
        ]
        words.append(_fix_elapsed(instance_words))
    return words, doc_mt_info


def _evaluate_instances(resegmented_instances: List[Instance], bleu_tokenizer: str) -> Dict[str, float]:
    class _YAALScorer:
        def __init__(self, computation_aware=False, is_longform=True):
            self.computation_aware = computation_aware
            self.is_longform = is_longform

        def get_delays_lengths(self, ins):
            timestamp_type = "elapsed" if self.computation_aware else "delays"
            delays = getattr(ins, timestamp_type, None)
            if not delays:
                return None, None, None
            tgt_len = ins.reference_length if ins.reference else len(delays)
            return delays, ins.source_length, tgt_len

        def compute(self, ins):
            out = self.get_delays_lengths(ins)
            if out[0] is None:
                return None
            delays, source_length, target_length = out
            recording_end = getattr(ins, "recording_end", float("inf"))
            is_longform = getattr(ins, "longform", None) or self.is_longform
            if (delays[0] >= source_length and not is_longform) or (delays[0] >= recording_end):
                return None
            LAAL, gamma = 0, max(len(delays), target_length) / source_length
            tau = 0
            for t_minus_1, d in enumerate(delays):
                if (d >= source_length and not is_longform) or (d >= recording_end):
                    break
                LAAL += d - t_minus_1 / gamma
                tau = t_minus_1 + 1
            return LAAL / tau if tau else None

        def __call__(self, instances_dict):
            scores = []
            for ins in instances_dict.values():
                delays = getattr(ins, "elapsed" if self.computation_aware else "delays", None)
                if not delays:
                    continue
                s = self.compute(ins)
                if s is not None:
                    scores.append(s)
            return mean(scores) if scores else float("nan")

    class _EndingOffsetScorer:
        """与 pipeline_longyaal_new 一致：last_delay - source_length。"""
        def __init__(self, computation_aware=False, use_absolute=False):
            self.computation_aware = computation_aware
            self.use_absolute = use_absolute

        def compute(self, ins):
            xs = getattr(ins, "elapsed" if self.computation_aware else "delays", None)
            if not xs:
                return None
            last_t = float(xs[-1])
            end_t = float(getattr(ins, "source_length", 0))
            v = last_t - end_t
            return abs(v) if self.use_absolute else v

        def __call__(self, instances_dict):
            scores = []
            for ins in instances_dict.values():
                v = self.compute(ins)
                if v is not None:
                    scores.append(v)
            return mean(scores) if scores else float("nan")

    ins_dict = {i: ins for i, ins in enumerate(resegmented_instances)}
    refs_list = [getattr(ins, "reference", "") or "" for ins in resegmented_instances]
    for ins in resegmented_instances:
        ins.reference = None
    try:
        ca_unaware = _YAALScorer(is_longform=True)(ins_dict)
        ca_aware = _YAALScorer(computation_aware=True, is_longform=True)(ins_dict)
        bleu_score = BLEU(tokenize=bleu_tokenizer).corpus_score(
            [ins.prediction for ins in resegmented_instances],
            [refs_list],
        ).score
        ending_offset = _EndingOffsetScorer(computation_aware=False, use_absolute=False)(ins_dict)
        ca_ending_offset = _EndingOffsetScorer(computation_aware=True, use_absolute=False)(ins_dict)
    finally:
        for ins, ref in zip(resegmented_instances, refs_list):
            ins.reference = ref
    return {
        "ca_unaware_yaal": ca_unaware,
        "ca_aware_yaal": ca_aware,
        "bleu": bleu_score,
        "ending_offset": ending_offset,
        "ca_ending_offset": ca_ending_offset,
    }


def resegment_inline(
    yaml_file: str,
    ref_sentences_file: str,
    hypothesis_file: str,
    char_level: bool,
    output_folder: str,
    bleu_tokenizer: str,
    offset_delays: bool,
    ref_unitizer_language: str,
    use_unit_punct_tokenize: bool = True,
) -> None:
    """全部逻辑在本文件。
    use_unit_punct_tokenize=True（longyaal2 默认）：ref 按 Qwen/空格拆 unit，用 pipeline_main_new 方式获索引，中间补标点；hyp 已有 unit+索引，中间补标点；ref/hyp token 对齐。
    否则：ref 用 Qwen units，hyp 用 prediction_units+delays。"""
    if use_unit_punct_tokenize:
        ref_words, segmentation, ref_sentences = _load_reference_unit_punct(
            yaml_file, ref_sentences_file, ref_unitizer_language, char_level, offset_delays
        )
        segmentation_order = _get_segmentation_order(segmentation)
        hyp_words, doc_mt_info = _load_hypothesis_unit_punct(hypothesis_file, segmentation_order)
    else:
        ref_words, segmentation, ref_sentences = _load_reference_qwen_units(
            yaml_file, ref_sentences_file, ref_unitizer_language, offset_delays
        )
        segmentation_order = _get_segmentation_order(segmentation)
        hyp_words, doc_mt_info = _load_hypothesis_prediction_units(hypothesis_file, segmentation_order)
    new_segmentation, ref_segmentation = _align_words(ref_words, hyp_words, char_level)

    def _debug_print_aligned_seg(seg_id: int, ref_sent: str, new_seg: List, prediction_str: str, head: int = 40):
        if seg_id > 0:
            return
        print("\n" + "=" * 60)
        print("[debug] 对齐后 segment 0")
        print("  reference:", repr(ref_sent[:80]) if ref_sent else "")
        print("  对齐到的 hyp tokens (new_seg) 前%d 个 (original, delay, unit_index):" % head)
        for j, w in enumerate(new_seg[:head]):
            print("    [%d] %r delay=%.0f unit_index=%s" % (j, getattr(w, "original", None), getattr(w, "delay", 0), getattr(w, "unit_index", None)))
        if len(new_seg) > head:
            print("    ... 共 %d 个 token" % len(new_seg))
        print("  prediction_str 前120字:", repr(prediction_str[:120]) if prediction_str else "")
        print("=" * 60 + "\n")

    doc_id_to_mt_idx = {doc_id: i for i, doc_id in enumerate(segmentation_order)}

    os.makedirs(output_folder, exist_ok=True)
    instances_dict = []
    alignment_tokens_list = []
    for idx, (seg, ref) in enumerate(zip(segmentation, ref_sentences)):
        new_seg = new_segmentation.get(idx, [])
        if new_seg:
            recording_lengths = [w.recording_length for w in new_seg]
            assert all(w.recording_length == recording_lengths[0] for w in new_seg), (
                f"Recording lengths mismatch segment {idx}: {recording_lengths}"
            )
        segment_duration_ms = seg["duration"]
        prediction = [w.original for w in new_seg if w.original is not None]
        prediction_str = "".join(prediction) if char_level else " ".join(prediction)
        # 若有 unit 对应 MT 的字符区间，用 MT 切片作为 prediction（与 Segale BLEU 一致）
        doc_id = seg.get("wav", "")
        mt_idx = doc_id_to_mt_idx.get(doc_id, -1)
        if mt_idx >= 0 and doc_mt_info[mt_idx].get("prediction_unit_char_starts_ends"):
            txt = doc_mt_info[mt_idx].get("prediction_text", "")
            starts_ends = doc_mt_info[mt_idx]["prediction_unit_char_starts_ends"]
            unit_indices = [getattr(w, "unit_index", None) for w in new_seg]
            unit_indices = [i for i in unit_indices if i is not None and 0 <= i < len(starts_ends)]
            if unit_indices:
                valid = [(starts_ends[i][0], starts_ends[i][1]) for i in unit_indices if len(starts_ends[i]) >= 2 and starts_ends[i][0] >= 0 and starts_ends[i][1] > 0]
                if valid:
                    char_start = min(s for s, _ in valid)
                    char_end = max(e for _, e in valid)
                    if char_end > char_start and char_start < len(txt):
                        prediction_str = txt[char_start:min(char_end, len(txt))].strip() or prediction_str
        _debug_print_aligned_seg(idx, ref, new_seg, prediction_str)
        # 标点不参与延迟：只取 unit 对应 token 的 delay/elapsed（unit_index 非 None）
        seg_words = [w for w in new_seg if getattr(w, "unit_index", None) is not None]
        new_seg_dict = {
            "index": idx,
            "doc_id": seg.get("wav", ""),
            "seg_id": idx + 1,
            "source": "",  # softsegmenter 无源语段文本，与 longyaal 结构一致留空
            "reference": ref,
            "source_length": segment_duration_ms,
            "delays": [w.delay - seg["offset"] for w in seg_words],
            "elapsed": [w.elapsed - seg["offset"] for w in seg_words],
            "recording_end": segment_duration_ms,
            "prediction": prediction_str,
        }
        instances_dict.append(new_seg_dict)
        # 本 segment 参与对齐的 ref/hyp token 列表，用于写 alignment_tokens.json
        ref_tokens = [w.text for w in ref_segmentation.get(idx, [])]
        hyp_tokens = [getattr(w, "original", None) or w.text for w in new_seg]
        alignment_tokens_list.append({
            "index": idx,
            "doc_id": seg.get("wav", ""),
            "seg_id": idx + 1,
            "reference": ref,
            "prediction": prediction_str,
            "ref_tokens": ref_tokens,
            "hyp_tokens": hyp_tokens,
        })

    # 单个 JSON 文件，ref_tokens/hyp_tokens 数组横排在一行，不竖着换行
    def _compact_tokens_line(tokens):
        return "[" + ", ".join(json.dumps(t, ensure_ascii=False) for t in tokens) + "]"

    alignment_tokens_path = os.path.join(output_folder, "alignment_tokens.json")
    with open(alignment_tokens_path, "w", encoding="utf-8") as f:
        lines = []
        for obj in alignment_tokens_list:
            ref_line = _compact_tokens_line(obj["ref_tokens"])
            hyp_line = _compact_tokens_line(obj["hyp_tokens"])
            lines.append(
                '  {\n'
                '    "index": %s,\n'
                '    "doc_id": %s,\n'
                '    "seg_id": %s,\n'
                '    "reference": %s,\n'
                '    "prediction": %s,\n'
                '    "ref_tokens": %s,\n'
                '    "hyp_tokens": %s\n'
                '  }' % (
                    json.dumps(obj["index"]),
                    json.dumps(obj["doc_id"], ensure_ascii=False),
                    json.dumps(obj["seg_id"]),
                    json.dumps(obj["reference"], ensure_ascii=False),
                    json.dumps(obj["prediction"], ensure_ascii=False),
                    ref_line,
                    hyp_line,
                )
            )
        f.write("[\n" + ",\n".join(lines) + "\n]\n")
    print(f"saved: {alignment_tokens_path}")

    out_path = os.path.join(output_folder, "instances.resegmented.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(instances_dict, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {out_path}")

    instances = [Instance(d, latency_unit="char" if char_level else "word") for d in instances_dict]
    scores = _evaluate_instances(instances, bleu_tokenizer)
    scores_path = os.path.join(output_folder, "scores.resegmented.csv")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("\t".join(scores.keys()) + "\n")
        f.write("\t".join(f"{v:.4f}" for v in scores.values()) + "\n")
    print(f"saved: {scores_path}")

# ---------- 以上为 softsegmenter 内联逻辑 ----------


LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ja": "Japanese",
}

BLEU_MAP = {
    "en": "13a",
    "zh": "zh",
    "de": "13a",
    "ja": "ja-mecab",
}


def _to_ms_if_needed(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    # pipeline_main_new instances.log stores source_length in seconds.
    if v < 1e4:
        return v * 1000.0
    return v


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_instances_log_to_softsegmenter_hypothesis(
    instances_log: str,
    hypothesis_file: str,
    char_level: bool,
) -> None:
    rows = load_jsonl(instances_log)
    converted = []

    for row in rows:
        delays = list(row.get("delays") or [])
        elapsed = list(row.get("elapsed") or [])
        prediction_units = list(row.get("prediction_units") or [])
        if not elapsed or len(elapsed) != len(delays):
            elapsed = list(delays)

        if prediction_units and len(prediction_units) == len(delays):
            if char_level:
                prediction = "".join(str(x) for x in prediction_units)
            else:
                prediction = " ".join(str(x) for x in prediction_units)
        else:
            prediction = row.get("prediction_text", "")

        converted.append(
            {
                "index": row.get("index"),
                "source": [row.get("source", "")],
                "reference": row.get("reference"),
                "prediction": prediction,
                "prediction_units": prediction_units,
                "prediction_text": row.get("prediction_text", ""),
                "prediction_unit_char_starts_ends": list(row.get("prediction_unit_char_starts_ends") or []),
                "delays": delays,
                "elapsed": elapsed,
                "source_length": _to_ms_if_needed(row.get("source_length")),
            }
        )

    save_jsonl(converted, hypothesis_file)
    print(f"saved: {hypothesis_file}")


def enrich_resegmented_instances_with_doc_info(
    yaml_file: str,
    instances_json: str,
) -> None:
    with open(yaml_file, "r", encoding="utf-8") as f:
        segmentation = yaml.safe_load(f) or []

    with open(instances_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(segmentation) != len(data):
        raise ValueError(
            f"Segment count mismatch: yaml={len(segmentation)} vs instances={len(data)}"
        )

    for idx, (seg, item) in enumerate(zip(segmentation, data)):
        item["index"] = idx
        item["doc_id"] = seg.get("wav")
        item["seg_id"] = idx + 1

    with open(instances_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"updated: {instances_json}")


def print_debug_units(
    yaml_file: str,
    reference_file: str,
    hypothesis_file: str,
    ref_unitizer_language: str,
    sample_idx: int = 0,
    head: int = 80,
):
    from pipeline_qwen3_forcealign_tokenizer2 import Qwen3ForceAlignTokenizer

    segmentation = yaml.safe_load(open(yaml_file, "r", encoding="utf-8")) or []
    references = [
        line.rstrip("\n")
        for line in open(reference_file, "r", encoding="utf-8")
    ]
    hypotheses = load_jsonl(hypothesis_file)

    if not segmentation or not references or not hypotheses:
        print("[debug] missing data for unit inspection")
        return

    idx = max(0, min(sample_idx, len(segmentation) - 1, len(references) - 1))
    tok = Qwen3ForceAlignTokenizer()

    ref_text = references[idx].strip().lower()
    ref_units = tok.encode_timestamp(ref_text, ref_unitizer_language)
    hyp = hypotheses[0]
    hyp_units = list(hyp.get("prediction_units") or [])

    print("\n" + "-" * 60)
    print(f"[debug] sample seg_id={idx + 1} doc_id={segmentation[idx].get('wav')}")
    print(f"[debug] ref_text[:120]={ref_text[:120]}")
    print(f"[debug] ref_units_len={len(ref_units)} hyp_units_len={len(hyp_units)}")
    print(f"[debug] ref_units_head={ref_units[:head]}")
    print(f"[debug] hyp_units_head={hyp_units[:head]}")

    mismatch = []
    for i, (r, h) in enumerate(zip(ref_units, hyp_units)):
        if r != h:
            mismatch.append((i, r, h))
        if len(mismatch) >= 20:
            break
    print(f"[debug] first_mismatches={mismatch}")
    print("-" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the softsegmenter branch on top of pipeline_main_new outputs."
    )
    parser.add_argument("--model_name", type=str, default="seed")
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="zh")
    parser.add_argument("--input_version", type=str, default="")
    parser.add_argument("--output_version", type=str, default="")
    parser.add_argument(
        "--instances_log",
        type=str,
        default=None,
        help="Defaults to data2/<model>/<src>_<tgt>/output_asr_/instances.log",
    )
    parser.add_argument(
        "--soft_output_folder",
        type=str,
        default=None,
        help="Defaults to data2/<model>/<src>_<tgt>/output_softsegmenter<ver>",
    )
    parser.add_argument(
        "--hypothesis_file",
        type=str,
        default=None,
        help="Defaults to <soft_output_folder>/hypothesis.jsonl",
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        default=None,
        help="Defaults to data2/input/ACL.ACLdev2023.<src>-xx.gold_segments.yaml",
    )
    parser.add_argument(
        "--source_sentences_file",
        type=str,
        default=None,
        help="Defaults to data2/input/ACL.6060.dev.<src>-xx.<tgt>.txt",
    )
    parser.add_argument("--char_level", action="store_true")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Tokenizer language for softsegmenter. Defaults to tgt_lang.",
    )
    parser.add_argument(
        "--bleu_tokenizer",
        type=str,
        default=None,
        help="Defaults to a language-safe tokenizer mapping.",
    )
    parser.add_argument("--offset_delays", action="store_true")
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    return parser

'''
python pipeline_softsegmenter_longyaal2.py
python pipeline_softsegmenter_longyaal2.py --src_lang en --tgt_lang de
python pipeline_softsegmenter_longyaal2.py --src_lang en --tgt_lang ja
'''

def main():
    args = build_parser().parse_args()
    pair = f"{args.src_lang}_{args.tgt_lang}"
    base_dir = os.path.join("data2", args.model_name, pair)

    instances_log = args.instances_log or os.path.join(base_dir, "output_asr_", "instances.log")
    soft_output_folder = args.soft_output_folder or os.path.join(
        base_dir, f"output_softsegmenter{args.output_version}2"
    )
    hypothesis_file = args.hypothesis_file or os.path.join(
        soft_output_folder, "hypothesis.jsonl"
    )
    yaml_file = args.yaml_file or os.path.join(
        "data2", "input", f"ACL.ACLdev2023.{args.src_lang}-xx.gold_segments.yaml"
    )
    source_sentences_file = args.source_sentences_file or os.path.join(
        "data2", "input", f"ACL.6060.dev.{args.src_lang}-xx.{args.tgt_lang}.txt"
    )
    lang = args.lang if args.lang is not None else args.tgt_lang
    ref_unitizer_language = LANG_MAP.get(lang, lang)
    bleu_tokenizer = args.bleu_tokenizer or BLEU_MAP.get(args.tgt_lang, "13a")
    char_level = args.char_level or args.tgt_lang in {"zh", "ja"}

    print("\n" + "=" * 60)
    print("Starting softsegmenter branch...")
    print(f"instances_log: {instances_log}")
    print(f"yaml_file: {yaml_file}")
    print(f"reference_file: {source_sentences_file}")
    print(f"output_folder: {soft_output_folder}")
    print(f"char_level: {char_level}")
    print(f"lang: {lang}")
    print(f"bleu_tokenizer: {bleu_tokenizer}")
    print(f"ref_unitizer_language: {ref_unitizer_language}")

    if not args.skip_convert:
        convert_instances_log_to_softsegmenter_hypothesis(
            instances_log=instances_log,
            hypothesis_file=hypothesis_file,
            char_level=char_level,
        )

    print_debug_units(
        yaml_file=yaml_file,
        reference_file=source_sentences_file,
        hypothesis_file=hypothesis_file,
        ref_unitizer_language=ref_unitizer_language,
        sample_idx=0,
        head=60,
    )

    resegment_inline(
        yaml_file=yaml_file,
        ref_sentences_file=source_sentences_file,
        hypothesis_file=hypothesis_file,
        char_level=char_level,
        output_folder=soft_output_folder,
        bleu_tokenizer=bleu_tokenizer,
        offset_delays=args.offset_delays,
        ref_unitizer_language=ref_unitizer_language,
    )

    instances_json = os.path.join(soft_output_folder, "instances.resegmented.json")
    enrich_resegmented_instances_with_doc_info(
        yaml_file=yaml_file,
        instances_json=instances_json,
    )

    if not args.skip_plot:
        ending_offset_delay(
            instances=instances_json,
            out_png=os.path.join(
                soft_output_folder, "last_delay_minus_recording_end.png"
            ),
            out_csv=os.path.join(
                soft_output_folder, "last_delay_minus_recording_end.csv"
            ),
        )

    print("Softsegmenter branch finished.")


if __name__ == "__main__":
    main()
