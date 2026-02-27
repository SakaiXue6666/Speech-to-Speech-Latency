# -*- coding: utf-8 -*-
"""
从 *_asr.json + gold_segments.yaml + ref_sentences (zh.txt) 生成 instances.log，
供 main_new.py 的 load_hypothesis / resegment 使用。SEGALE 格式后续再做。
"""
import argparse
import json
import os
import yaml

'''
python build_instances_log.py `
  --yaml_file data/ACL.ACLdev2023.en-xx.gold_segments.yaml `
  --ref_sentences_file data/ACL.6060.dev.en-xx.zh.txt `
  --asr_dir data/output_qwen_asr `
  --output_file data/output_qwen_asr/instances.log
'''
# ref_segments.yaml 有多个 src 文档
# references.txt 有多个 ref 文档；行和ref_segments.yaml的行一一绑定
# instances.log 有多个 hyp 文档

def get_segmentation_order(segmentation):
    """与 main_new.get_segmentation_order 一致：按首次出现顺序的 wav 列表。"""
    order = []
    for seg in segmentation:
        wav = seg["wav"]
        if not order or order[-1] != wav:
            order.append(wav)
    return order


def text_to_delays_with_punct(text: str, ts: list) -> tuple:
    """
    为 text（带标点/空格）的每个字符分配一个时间戳（ms）。
    适用于任何语言：先把 ts 展平到字符级，再用双指针匹配 text。

    步骤：
      1. 展平 ts → char_times: [(char, start_time_ms), ...]
         多字符 token（如英文词 "hello"）会拆成逐字符，时间在 token 区间内线性插值。
      2. 双指针遍历 text：
         - 匹配 char_times[j] → 用该字符的时间，j += 1
         - 不匹配（标点/空格等） → 用上一次匹配到的时间

    返回 (delays_ms, elapsed_ms)，长度均为 len(text)。
    """
    if not ts:
        return [], []

    # 步骤 1：展平 ts 到字符级
    char_times = []
    for token in ts:
        t_text = token["text"]
        start_ms = int(round(token["start_time"] * 1000))
        end_ms = int(round(token.get("end_time", token["start_time"]) * 1000))
        n = len(t_text)
        for k, c in enumerate(t_text):
            t = start_ms + (end_ms - start_ms) * k // n if n > 1 else start_ms
            char_times.append((c, t))

    # 步骤 2：双指针匹配
    delays_ms = []
    j = 0
    last_time = char_times[0][1]
    for ch in text:
        if j < len(char_times) and char_times[j][0] == ch:
            last_time = char_times[j][1]
            j += 1
        delays_ms.append(last_time)

    elapsed = list(delays_ms)
    return delays_ms, elapsed


def build_instances_log(
    yaml_file: str,
    ref_sentences_file: str,
    asr_dir: str,
    output_file: str,
    asr_suffix: str = "_tgt_asr.json",
    char_level_with_punct: bool = False,
) -> None:
    """
    按 segmentation_order 顺序，为每个 wav 生成一条 instance，写入 output_file（instances.log）。
    char_level_with_punct=True 时：prediction 用 ASR 的 text（带标点），delays/elapsed 用
    text_to_delays_with_punct 对齐（标点取上一 token 时间），保证 len(delays)==len(prediction)。
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        segmentation = yaml.load(f, Loader=yaml.CLoader)

    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        ref_sentences = [line.strip() for line in f if line.strip()]

    assert len(segmentation) == len(ref_sentences), (
        f"yaml 段数 {len(segmentation)} != ref 行数 {len(ref_sentences)}"
    )

    order = get_segmentation_order(segmentation)
    instances = []

    for idx, wav in enumerate(order):
        # 该 wav 对应的 ref 句（按段顺序拼接成一段参考文本）
        ref_indices = [i for i, seg in enumerate(segmentation) if seg["wav"] == wav]
        reference = " ".join(ref_sentences[i] for i in ref_indices)

        # 源时长：该 wav 最后一段的 offset+duration（yaml 里是秒），转 ms
        wav_segs = [seg for seg in segmentation if seg["wav"] == wav]
        last = wav_segs[-1]
        source_length_ms = int((last["offset"] + last["duration"]) * 1000)

        # 对应 _asr.json：asr_dir / {wav_basename}_tgt_asr.json
        wav_base = os.path.splitext(wav)[0]
        asr_path = os.path.join(asr_dir, wav_base + asr_suffix)

        if not os.path.isfile(asr_path):
            print(f"跳过 {wav}: 未找到 {asr_path}")
            continue

        with open(asr_path, "r", encoding="utf-8") as f:
            asr = json.load(f)

        # 优先用 time_stamps_session_sec（相对 src 开始的会话时间），否则 time_stamps（wav 相对）
        ts = asr.get("time_stamps_session_sec") or asr.get("time_stamps")
        if not ts:
            print(f"跳过 {wav}: {asr_path} 无 time_stamps / time_stamps_session_sec")
            continue

        if char_level_with_punct:
            prediction = (asr.get("text", "") or "").strip() or " ".join(t["text"] for t in ts)
            delays, elapsed = text_to_delays_with_punct(prediction, ts)
        else:
            prediction = " ".join(t["text"] for t in ts)
            delays = [int(round(t["start_time"] * 1000)) for t in ts]
            elapsed = list(delays)

        src_path = asr.get("src", "")
        if not src_path:
            src_path = os.path.join("data", wav)

        rec = {
            "index": len(instances),
            "source": src_path,
            "reference": reference,
            "prediction": prediction,
            "delays": delays,
            "elapsed": elapsed,
            "source_length": source_length_ms,
        }
        instances.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in instances:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"已写入 {output_file}，共 {len(instances)} 条（按 segmentation_order）。")


def main():
    parser = argparse.ArgumentParser(description="从 _asr.json + yaml + ref 生成 instances.log")
    parser.add_argument("--yaml_file", type=str, default="data/ACL.ACLdev2023.en-xx.gold_segments.yaml")
    parser.add_argument("--ref_sentences_file", type=str, default="data/ACL.6060.dev.en-xx.zh.txt")
    parser.add_argument("--asr_dir", type=str, default="data/output_qwen_asr")
    parser.add_argument("--output_file", type=str, default="data/output_qwen_asr/instances.log")
    parser.add_argument("--asr_suffix", type=str, default="_tgt_asr.json")
    parser.add_argument(
        "--char_level_with_punct",
        action="store_true",
        help="prediction 用 ASR 的 text（带标点），标点时间取上一 token，len(delays)=len(prediction)",
    )
    args = parser.parse_args()

    build_instances_log(
        yaml_file=args.yaml_file,
        ref_sentences_file=args.ref_sentences_file,
        asr_dir=args.asr_dir,
        output_file=args.output_file,
        asr_suffix=args.asr_suffix,
        char_level_with_punct=args.char_level_with_punct,
    )


if __name__ == "__main__":
    main()
