# -*- coding: utf-8 -*-
"""
从 *_asr.json + gold_segments.yaml + ref_sentences (zh.txt) 生成 instances.log，
供 main_new.py 的 load_hypothesis / resegment 使用。SEGALE 格式后续再做。

--s2s 模式：生成 speech-to-speech 格式（prediction 为 tgt wav 路径，含 durations/intervals 等）。
默认模式：生成 speech-to-text 格式（prediction 为文本）。
"""
import argparse
import json
import os
import wave
import yaml

# ----------------------------------------------------------------------------
# longyaal 输入格式：
# - ref_segments.yaml 有多个 src 文档
# - references.txt 有多个 ref 文档；行和ref_segments.yaml的行一一绑定
# - instances.log 有多个 hyp 文档 （这个文件要得到这个！）


# （这个文件要得到这个！）
# instances.log (s2s) 格式：{
# - index: 音频索引（与 ref_segments.yaml "wav" 绑定）

# - prediction: tgt wav 路径
# - delays: 每个输出单元的 起始时间_ms
# - durations: 每个输出单元的 时长_ms
# - elapsed: 每个输出单元的 计算耗时相关时间（ms）？？？
# - intervals: 每个输出单元的 [起始时间_ms, 时长_ms]

# - prediction_offset: 第一个输出单元相对源音频起点的时间（ms），即整段预测的起始偏移；表示从 xxx ms 处开始有输出。
# - prediction_length: tgt wav 的总时长（s）

# - source_length: src1 wav 的总时长（s）
# - reference: src1 wav 路径 ？？？
# - source: src1 wav 路径
# }
# ...
# { - source: srcN wav 路径, ... }


# 输入文件：
# - 1. ref_segments.yaml: 有多个 src 文档
# -- [{duration, offset, wav1}, {duration, offset, wav1}, ..., {duration, offset, wavN}]

# - 2. asr_dir/: 有多个 asr 文档
# -- asr1.json: {src1, tgt, text, timestamps: {text, start_time, end_time}}
# -- ...
# -- asrN.json: {srcN, tgt, text, timestamps: {text, start_time, end_time}}


# 1. instances.log 的 source: 填 ref_segments.yaml 的 wav 路径，还是填 asr.json 的 src 路径？
# - ref_segments.yaml 的 wav 的值是： <src_wav_path>
# - asr.json 的 src 的值是： .../<src_wav_path>

# 2. source_length 怎么求？
# 3. prediction_length 怎么求？ -- 需要用到timeline !

# ----------------------------------------------------------------------------

# def text__with_punct(text: str, ts: list) -> tuple:
#     """
#     备用函数
#     """
#     if not ts:
#         return [], []

#     # 步骤 1：展平 ts 到字符级
#     char_times = []
#     for token in ts:
#         t_text = token["text"]
#         start_ms = int(round(token["start_time"] * 1000))
#         end_ms = int(round(token.get("end_time", token["start_time"]) * 1000))
#         n = len(t_text)
#         for k, c in enumerate(t_text):
#             t = start_ms + (end_ms - start_ms) * k // n if n > 1 else start_ms
#             char_times.append((c, t))

#     # 步骤 2：双指针匹配
#     delays_ms = []
#     j = 0
#     last_time = char_times[0][1]
#     for ch in text:
#         if j < len(char_times) and char_times[j][0] == ch:
#             last_time = char_times[j][1]
#             j += 1
#         delays_ms.append(last_time)

#     elapsed = list(delays_ms)
#     return delays_ms, elapsed


def _get_wav_duration_sec(wav_path: str) -> float:
    """读 wav 文件时长（秒）。失败返回 0.0。"""
    try:
        with wave.open(wav_path, "r") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _wav_time_to_session_sec(wav_sec: float, timeline: list, t0: float) -> float:
    """把「无 gap 的 wav 内时间」映射为「考虑 gap 的会话相对时间」(秒)。"""
    for t in timeline:
        o = t["offset_sec"]
        d = t["duration_sec"]
        if o <= wav_sec < o + d:
            heard = t.get("heard_start") or t.get("write_before") or t.get("play_timestamp_before") or t.get("receive_timestamp")
            return (heard - t0) + (wav_sec - o)
    last = timeline[-1]
    heard_last = last.get("heard_start") or last.get("write_before") or last.get("play_timestamp_before") or last.get("receive_timestamp")
    return (heard_last - t0) + max(0.0, wav_sec - last["offset_sec"])


def _asr_to_session_timestamps(asr_data: dict, use_first_src_send_for_t0: bool = True):
    """
    若 asr 含 tgt_timeline 且文件存在，用 timeline 把 time_stamps_no_gap 映射为会话相对时间（秒）。
    返回 (session_sec_list, prediction_length_sec) 或 (None, None) 表示用 wav 相对即可。
    """
    # timeline 路径
    tgt_timeline_path = asr_data.get("tgt_timeline")
    if not tgt_timeline_path or not os.path.isfile(os.path.normpath(tgt_timeline_path)):
        return None, None
    
    # time_stamls_no_gap
    ts_list = asr_data.get("time_stamps_no_gap") or asr_data.get("time_stamps") or []
    if not ts_list:
        return None, None
    
    # 加载 timeline
    with open(os.path.normpath(tgt_timeline_path), "r", encoding="utf-8") as f:
        tl_data = json.load(f)
    timeline = tl_data.get("timeline") or []
    
    if not timeline:
        return None, 
    # timeline 不为空
    # t0：默认优先 src send，否则用第一个 tgt receive
    if use_first_src_send_for_t0 and tl_data.get("first_send_timestamp") is not None:
        t0 = float(tl_data["first_send_timestamp"])
    else:
        t0 = timeline[0].get("receive_timestamp")
    # 只对 start_time 做 wav→session 映射；end_time = start_time + duration，避免映射后 end < start
    session_sec = []
    # for t in ts_list:
    #     start_s = _wav_time_to_session_sec(t["start_time"], timeline, t0)
    #     dur_s = max(0.0, t["end_time"] - t["start_time"])
    #     session_sec.append({
    #         "text": t["text"], 
    #         "start_time": start_s, 
    #         "end_time": start_s + dur_s
    #     })
    # 单调修正：映射后可能 start[i+1] < end[i]，强制后移保证不回退
    # for j in range(1, len(session_sec)):
    #     prev_end = session_sec[j - 1]["end_time"]
    #     if session_sec[j]["start_time"] < prev_end:
    #         dur_j = session_sec[j]["end_time"] - session_sec[j]["start_time"]
    #         session_sec[j]["start_time"] = prev_end
    #         session_sec[j]["end_time"] = prev_end + dur_j

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
    prev_heard_end = 0.0  # 上一个 token/片段 “听完”的会话相对时间（秒）

    for t in ts_list:
        mapped_start = _wav_time_to_session_sec(t["start_time"], timeline, t0)
        dur_s = max(0.0, float(t["end_time"]) - float(t["start_time"]))

        # 单调修正（代码2风格）：下一段开始 = max(映射出来的开始, 上一段听完)
        heard_start = max(float(mapped_start), float(prev_heard_end))
        heard_end = heard_start + dur_s
        prev_heard_end = heard_end
        
        session_sec.append({
            "text": t["text"],
            "start_time": heard_start,
            "end_time": heard_end,
        })
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

    last_end = session_sec[-1]["end_time"] if session_sec else 0.0
    return session_sec, last_end


def _build_instances_log_s2s(ref_segments_yaml: str, asr_dir: str, output_file: str, out_dir: str) -> None:
    """
    从 ref_segments.yaml + asr_dir 下的 *_asr.json 生成 s2s 格式 instances.log。
    yaml 的 wav 与 asr 的 src 按 basename 匹配；每条记录对应一个 source（src wav）。
    """
    # 1. 读 yaml，按出现顺序收集不重复的 wav（每个 wav 对应一个 source / 一行 instances.log）
    with open(ref_segments_yaml, "r", encoding="utf-8") as f:
        segments = yaml.safe_load(f)
    if not segments:
        segments = []
    unique_wavs = []
    seen = set()
    for seg in segments:
        w = seg.get("wav")
        if w is not None and w not in seen:
            seen.add(w)
            unique_wavs.append(w)

    # 2. 扫描 asr_dir，按 src 的 basename 建表，便于和 yaml 的 wav 匹配
    asr_by_src = {}
    asr_dir = os.path.normpath(asr_dir)
    for fn in os.listdir(asr_dir):
        if not fn.endswith("_asr.json"):
            continue
        path = os.path.join(asr_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        src = data.get("src")
        if not src:
            continue
        key = os.path.basename(os.path.normpath(src))
        asr_by_src[key] = data

    # 3. 对每个 yaml 中的 wav 找对应 asr，拼一条 s2s 记录
    instances = []
    for idx, yaml_wav in enumerate(unique_wavs):
        key = os.path.basename(os.path.normpath(yaml_wav))
        asr = asr_by_src.get(key)
        if asr is None:
            continue

        # 时间戳：若有 tgt_timeline 则做 wav→session 映射（build 里做），否则用 asr 的 wav 相对
        session_ts, session_pred_len = _asr_to_session_timestamps(asr, use_first_src_send_for_t0=True)
        if session_ts is not None:
            ts = session_ts
            prediction_length_sec = session_pred_len

            # 👇 如果指定了输出目录，就把映射后的 asr 存进去
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

                out = dict(asr)  # 复制一份
                out["time_stamps"] = session_ts
                out["prediction_length"] = float(session_pred_len or 0.0)

                tgt_wav = asr.get("tgt") or ""
                base_name = os.path.basename(os.path.splitext(tgt_wav)[0])
                out_path = os.path.join(out_dir, base_name + "_asr.json")

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

                print(f"  已保存: {out_path}")

        else:
            ts = asr.get("time_stamps") or asr.get("time_stamps_no_gap") or []
            prediction_length_sec = asr.get("prediction_length")
            if prediction_length_sec is None and ts:
                prediction_length_sec = ts[-1].get("end_time", ts[-1]["start_time"])
            prediction_length_sec = float(prediction_length_sec or 0.0)

        # delays (ms)
        delays = [int(round(t["start_time"] * 1000)) for t in ts]

        # durations (ms)
        durations = [int(round((t.get("end_time", t["start_time"]) - t["start_time"]) * 1000)) for t in ts]

        # intervals
        intervals = [[d, dur] for d, dur in zip(delays, durations)]

        # prediction_offset
        prediction_offset = delays[0] if delays else 0

        # prediction_length：会话相对时长（秒），供 longyaal 等用
        prediction_length = prediction_length_sec

        tgt_path = asr.get("tgt") or ""

        # source_length
        src_path = os.path.normpath(asr.get("src", ""))
        source_length_sec = _get_wav_duration_sec(src_path)

        prediction_text = asr.get("text") or ""

        rec = {
            "index": idx,
            "prediction": tgt_path,
            "delays": delays,
            "durations": durations,
            "elapsed": [],
            "intervals": intervals,
            "prediction_offset": prediction_offset,
            "prediction_length": prediction_length,
            "source_length": source_length_sec,
            "reference": src_path,
            "source": src_path,
            "prediction_text": prediction_text,
        }
        instances.append(rec)

    # 4. 写出 JSONL（行间用 \n 分隔，最后一行后不写 \n，避免多出一行）
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, rec in enumerate(instances):
            if i > 0:
                f.write("\n")  # 👈 行间用 \n 分隔，最后一行后不写 \n，避免多出一行
            f.write(json.dumps(rec, ensure_ascii=False))

'''
python build_instances_log4.py
'''

def main():
    s2s = True
    yaml_file = "data/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    asr_dir = "data/output_qwen_asr3"
    output_file = "data/output_qwen_asr3/instances.log"
    out_dir = "data/output_qwen_asr3"
    if s2s and yaml_file and asr_dir and output_file:
        _build_instances_log_s2s(yaml_file, asr_dir, output_file, out_dir)
        return
    if not s2s or not yaml_file or not asr_dir or not output_file:
        pass


if __name__ == "__main__":
    main()