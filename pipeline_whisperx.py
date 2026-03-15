# -*- coding: utf-8 -*-
"""
WhisperX 作为 ASR + 词级对齐，输出与 pipeline_qwen3_asr2 相同的 JSON 格式，
以便与 SEGALE 句子级对齐 + pipeline_longyaal 的 unit 索引衔接、计算 latency。

流程对应关系：
- Qwen：ASR 得到 time_stamps（每个一条 {text, start_time, end_time}），
  longyaal 用 time_stamps 拼成 asr_units，SEGALE 只记录对齐句在 asr text 上的起止，
  即可对应到 unit 区间算 latency。
- WhisperX：transcribe + align 得到 word_segments（词/字级 start/end），
  这里把 word_segments 转成与 Qwen 相同的 time_stamps 格式并写出同一结构的 JSON，
  下游 _load_asr_units / _match_tgt_units_for_seg 无需改动即可复用。

- 每个 unit 对应 ASR 全文的索引起止（char_start/char_end）：
  Qwen 侧用 pipeline_main_new.add_char_spans_to_asr_json 通过 tokenizer 一致 + 顺序匹配得到。
  WhisperX 侧 full_text 即由同一批 word_segments 按序拼接，可直接用累积偏移量算出每个 unit
  在原句（ASR text）上的 char_start/char_end，无需再做子串查找，输出与 add_char_spans 同构。

使用前请安装 WhisperX（例如在项目根目录）：
  pip install -e whisperX
  # 或 pip install whisperx

示例：
  python pipeline_whisperx.py --manifest data/output_qwen_livetranslate/manifest.jsonl --out_dir data/output_whisperx_asr --tgt_language zh
"""

import argparse
import gc
import json
import os
import sys

# 若把 whisperX 放在项目下且未安装，可取消下面注释并改 WHISPERX_PARENT 为实际路径
# WHISPERX_PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisperX")
# if os.path.isdir(WHISPERX_PARENT) and WHISPERX_PARENT not in sys.path:
#     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


# 无空格语言（如中文）用 "" 拼接，有空格用 " " 拼接
LANGUAGES_WITHOUT_SPACES = {"zh", "ja", "zh-cn", "zh-tw"}


def _normalize_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    if code in ("chinese", "zh", "ch"):
        return "zh"
    return code


def run_whisperx_single(
    tgt_wav_path: str,
    tgt_language: str = "zh",
    device: str = "cuda",
    model_name: str = "large-v3",
    batch_size: int = 16,
    chunk_size: int = 30,
    align_model_name: str = None,
    model_dir: str = None,
):
    """
    对单条 tgt 音频跑 WhisperX：transcribe + align，返回与 Qwen ASR 兼容的结构。
    返回: dict 含 keys: text, time_stamps, language
    """
    import whisperx
    from whisperx import load_audio, load_model, load_align_model, align

    lang = _normalize_lang(tgt_language)
    audio = load_audio(tgt_wav_path)

    # 1) ASR
    model = load_model(
        model_name,
        device=device,
        download_root=model_dir,
        language=lang,
    )
    result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size)
    language = result.get("language") or lang
    segments = result.get("segments") or []
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Align（得到 word_segments）
    if not segments:
        full_text = ""
        word_segments = []
        char_spans = []
    else:
        align_model, align_metadata = load_align_model(
            language,
            device,
            model_name=align_model_name,
            model_dir=model_dir,
        )
        result = align(
            segments,
            align_model,
            align_metadata,
            tgt_wav_path,
            device,
        )
        word_segments = result.get("word_segments") or []
        full_text, char_spans = _build_full_text_and_char_spans(word_segments, language)
        del align_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) 转成与 Qwen 一致的 time_stamps，并带上每个 unit 在 ASR 全文的索引起止（与 add_char_spans 同构）
    time_stamps = []
    for i, w in enumerate(word_segments):
        ts = {
            "text": w.get("word", ""),
            "start_time": round(float(w.get("start", 0)), 3),
            "end_time": round(float(w.get("end", 0)), 3),
        }
        if i < len(char_spans):
            ts["char_start"], ts["char_end"] = char_spans[i]
        else:
            ts["char_start"] = ts["char_end"] = -1
        time_stamps.append(ts)

    return {
        "text": full_text,
        "time_stamps": time_stamps,
        "language": language,
    }


def _build_full_text(word_segments: list, language: str) -> str:
    """与 WhisperX 对齐一致：无空格语言用 '' 拼接，否则用 ' ' 拼接。"""
    lang = _normalize_lang(language)
    words = [w.get("word", "") for w in word_segments]
    if lang in LANGUAGES_WITHOUT_SPACES:
        return "".join(words)
    return " ".join(words)


def _build_full_text_and_char_spans(
    word_segments: list, language: str
) -> tuple:
    """
    在拼接 full_text 的同时，算出每个 unit 在 full_text 中的字符索引起止，
    与 pipeline_main_new.add_char_spans_to_asr_json 的 char_start/char_end 语义一致
    （即「原句」= 本段 ASR 全文，左闭右开）。
    返回: (full_text, list of (char_start, char_end))
    """
    lang = _normalize_lang(language)
    no_space = lang in LANGUAGES_WITHOUT_SPACES
    parts = []
    spans = []
    pos = 0
    for w in word_segments:
        word = w.get("word", "")
        char_start = pos
        if no_space:
            parts.append(word)
            pos += len(word)
        else:
            if parts:
                parts.append(" ")
                pos += 1
            parts.append(word)
            pos += len(word)
        spans.append((char_start, pos))
    full_text = "".join(parts)
    return full_text, spans


def run_tgt_asr_from_manifest(
    manifest_path: str,
    out_dir: str = "data/output_whisperx_asr",
    tgt_language: str = "Chinese",
    device: str = "cuda",
    model_name: str = "large-v3",
    batch_size: int = 16,
    chunk_size: int = 30,
    align_model_name: str = None,
    model_dir: str = None,
) -> None:
    """
    读 manifest.jsonl（每行含 "tgt" 与 "src"），对每条 tgt wav 跑 WhisperX ASR+align，
    写出 out_dir/{basename}_asr.json，格式与 pipeline_qwen3_asr2 一致，便于 longyaal 复用。
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    lang = _normalize_lang(tgt_language)

    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(lines)}] 跳过：无效 JSON - {e}")
            continue

        tgt_wav = rec.get("tgt")
        if not tgt_wav:
            print(f"[{i+1}/{len(lines)}] 跳过：无 tgt 路径")
            continue
        tgt_wav = os.path.normpath(tgt_wav)
        if not os.path.isfile(tgt_wav):
            print(f"[{i+1}/{len(lines)}] 跳过：文件不存在 {tgt_wav}")
            continue

        print(f"[{i+1}/{len(lines)}] WhisperX ASR: {tgt_wav}")

        try:
            out = run_whisperx_single(
                tgt_wav_path=tgt_wav,
                tgt_language=lang,
                device=device,
                model_name=model_name,
                batch_size=batch_size,
                chunk_size=chunk_size,
                align_model_name=align_model_name,
                model_dir=model_dir,
            )
        except Exception as e:
            print(f"  错误: {e}")
            continue

        # 与 Qwen 输出字段一致
        ts_list = out["time_stamps"]
        payload = {
            "src": rec.get("src", ""),
            "tgt": tgt_wav,
            "text": out["text"],
            "time_stamps": ts_list,
        }
        if ts_list:
            payload["prediction_length"] = round(ts_list[-1]["end_time"], 2)
        else:
            payload["prediction_length"] = 0.0

        base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
        out_path = os.path.join(out_dir, base_name + "_asr.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {out_path}")

'''
python pipeline_whisperx.py --manifest data2/.../manifest.jsonl --out_dir data2/.../output_whisperx_asr --tgt_language zh
'''

def main():
    parser = argparse.ArgumentParser(description="WhisperX ASR + align，输出与 Qwen ASR 同构 JSON，供 SEGALE/longyaal 用")
    parser.add_argument("--manifest", type=str, default="data/output_qwen_livetranslate/manifest.jsonl", help="manifest.jsonl 路径")
    parser.add_argument("--out_dir", type=str, default="data/output_whisperx_asr", help="输出目录")
    parser.add_argument("--tgt_language", type=str, default="Chinese", help="目标语种，如 Chinese / English")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper 模型名")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=30, help="VAD 合并后单段最大时长(秒)。默认 30；显存不足可改 15 或 10；整段很短可保持 30")
    parser.add_argument("--align_model", type=str, default=None, help="强制对齐模型，默认按语言自动选")
    parser.add_argument("--model_dir", type=str, default=None, help="模型缓存目录")
    args = parser.parse_args()

    run_tgt_asr_from_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        tgt_language=args.tgt_language,
        device=args.device,
        model_name=args.model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        align_model_name=args.align_model or None,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
