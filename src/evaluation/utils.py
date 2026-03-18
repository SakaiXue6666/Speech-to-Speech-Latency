from pipeline_qwen3_forcealign_tokenizer2 import Qwen3ForceAlignTokenizer

qwen_tok = Qwen3ForceAlignTokenizer()


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _qwen_units(text: str, language: str) -> List[str]:
    return qwen_tok.encode_timestamp(_norm(text), language)
