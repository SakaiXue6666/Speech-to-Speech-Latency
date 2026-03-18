"""ASR module entrypoints with lazy imports."""

from .qwen_char_spans import add_char_spans_for_dir, add_char_spans_to_asr_json


def step1_asr(*args, **kwargs):
    from .qwen_transformers import step1_asr as impl

    return impl(*args, **kwargs)


def step1_asr_vllm(*args, **kwargs):
    from .qwen_vllm import step1_asr_vllm as impl

    return impl(*args, **kwargs)


__all__ = ["step1_asr", "step1_asr_vllm", "add_char_spans_for_dir", "add_char_spans_to_asr_json"]
