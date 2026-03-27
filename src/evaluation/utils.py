import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import nagisa
# import torch
# from qwen_asr.core.transformers_backend import (
#     Qwen3ASRConfig,
#     Qwen3ASRForConditionalGeneration,
#     Qwen3ASRProcessor,
# )
# from transformers import AutoConfig, AutoModel, AutoProcessor

# from .utils import (
#     AudioLike,
#     ensure_list,
#     normalize_audios,
# )


class Qwen3ForceAlignTokenizer():
    def __init__(self):
        ko_dict_path = os.path.join(os.path.dirname(__file__), "assets", "korean_dict_jieba.dict")
        ko_scores = {}
        with open(ko_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word = line.split()[0]
                ko_scores[word] = 1.0
        self.ko_score = ko_scores
        self.ko_tokenizer = None

    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E
            or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
        )

    def tokenize_chinese_mixed(self, text: str) -> List[str]:
        tokens: List[str] = []
        current_latin: List[str] = []

        def flush_latin():
            nonlocal current_latin
            if current_latin:
                token = "".join(current_latin)
                cleaned = self.clean_token(token)
                if cleaned:
                    tokens.append(cleaned)
                current_latin = []

        for ch in text:
            if self.is_cjk_char(ch):
                flush_latin()
                tokens.append(ch)
            else:
                if self.is_kept_char(ch):
                    current_latin.append(ch)
                else:
                    flush_latin()

        flush_latin()

        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        words = nagisa.tagging(text).words
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, ko_tokenizer, text: str) -> List[str]:
        raw_tokens = ko_tokenizer.tokenize(text)
        tokens: List[str] = []
        for w in raw_tokens:
            w_clean = self.clean_token(w)
            if w_clean:
                tokens.append(w_clean)
        return tokens

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    # ------------------------------------------------------------------
    # 带字符偏移的版本：每个 token 附带 (start_char, end_char) 在原文中的位置
    # 用于「整段 tokenize + 字符跨度」与 ASR units 对齐
    # ------------------------------------------------------------------

    def split_segment_with_chinese_with_offsets(self, seg: str, base: int = 0) -> List[Tuple[str, int, int]]:
        """与 split_segment_with_chinese 行为一致，但每个 token 带 (start_char, end_char)。"""
        out: List[Tuple[str, int, int]] = []
        buf: List[str] = []
        buf_start = base
        pos = base

        def flush_buf():
            nonlocal buf, buf_start
            if buf:
                token = "".join(buf)
                cleaned = self.clean_token(token)
                if cleaned:
                    out.append((cleaned, buf_start, pos))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                cleaned = self.clean_token(ch)
                if cleaned:
                    out.append((cleaned, pos, pos + 1))
                pos += 1
            else:
                if not buf:
                    buf_start = pos
                buf.append(ch)
                pos += 1
        flush_buf()
        return out

    def tokenize_space_lang_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """与 tokenize_space_lang 行为一致，但每个 token 带 (start_char, end_char) 在 text 中的绝对坐标。"""
        result: List[Tuple[str, int, int]] = []
        pos = 0
        for seg in text.split():
            idx = text.find(seg, pos)
            if idx < 0:
                idx = pos
            start = idx
            for tok, a, b in self.split_segment_with_chinese_with_offsets(seg, 0):
                result.append((tok, start + a, start + b))
            pos = idx + len(seg)
        return result

    def encode_timestamp_with_offsets(self, text: str, language: str) -> List[Tuple[str, int, int]]:
        """
        与 encode_timestamp 行为完全一致，但每个 token 附带 (start_char, end_char) 在 text 中的字符跨度。
        用于「整段 tokenize + 字符跨度」确定 segment 对应的 unit 区间，避免按句分词导致边界不一致。
        """
        language = language.lower()
        if language == "japanese":
            word_list = self.tokenize_japanese(text)
            pos = 0
            out = []
            for w in word_list:
                idx = text.find(w, pos)
                if idx < 0:
                    idx = pos
                out.append((w, idx, idx + len(w)))
                pos = idx + len(w)
            return out
        if language == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer
                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
            pos = 0
            out = []
            for w in word_list:
                idx = text.find(w, pos)
                if idx < 0:
                    idx = pos
                out.append((w, idx, idx + len(w)))
                pos = idx + len(w)
            return out
        return self.tokenize_space_lang_with_offsets(text)

    def encode_timestamp(self, text: str, language: str) -> List[str]:
        language = language.lower()

        if language.lower() == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language.lower() == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer
                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
        else:
            word_list = self.tokenize_space_lang(text)
        
        return word_list
    

qwen_tok = Qwen3ForceAlignTokenizer()


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _qwen_units(text: str, language: str) -> List[str]:
    return qwen_tok.encode_timestamp(_norm(text), language)
