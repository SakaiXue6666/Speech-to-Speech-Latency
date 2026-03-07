import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
    