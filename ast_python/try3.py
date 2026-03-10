import json
from pathlib import Path
from statistics import mean


def load_timeline(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_timeline(data: dict) -> dict:
    timeline = data.get("timeline", [])
    first_send_timestamp = data.get("first_send_timestamp")

    if not timeline:
        return {
            "num_chunks": 0,
            "content_sec": 0.0,
            "heard_sec": 0.0,
            "silence_sec": 0.0,
            "first_chunk_delay_sec": None,
            "avg_chunk_sec": 0.0,
            "avg_receive_gap_sec": 0.0,
            "avg_gap_before_sec": 0.0,
            "max_gap_before_sec": 0.0,
        }

    content_sec = sum(x.get("duration_sec", 0.0) for x in timeline)

    first_heard = timeline[0].get("heard_start")
    last_heard = timeline[-1].get("heard_end")
    heard_sec = (last_heard - first_send_timestamp) if (
        first_send_timestamp is not None and last_heard is not None
    ) else 0.0

    silence_sec = heard_sec - content_sec if heard_sec else 0.0

    receive_gaps = [x.get("receive_gap_sec", 0.0) for x in timeline[1:]]
    gap_befores = [x.get("gap_before_sec", 0.0) for x in timeline]

    first_chunk_delay_sec = None
    if first_send_timestamp is not None and first_heard is not None:
        first_chunk_delay_sec = first_heard - first_send_timestamp

    return {
        "num_chunks": len(timeline),
        "content_sec": content_sec,
        "heard_sec": heard_sec,
        "silence_sec": silence_sec,
        "first_chunk_delay_sec": first_chunk_delay_sec,
        "avg_chunk_sec": mean(x.get("duration_sec", 0.0) for x in timeline),
        "avg_receive_gap_sec": mean(receive_gaps) if receive_gaps else 0.0,
        "avg_gap_before_sec": mean(gap_befores) if gap_befores else 0.0,
        "max_gap_before_sec": max(gap_befores) if gap_befores else 0.0,
    }


def print_compare(name1: str, s1: dict, name2: str, s2: dict):
    rows = [
        ("num_chunks", "块数"),
        ("content_sec", "内容时长(s)"),
        ("heard_sec", "真实听到时长(s)"),
        ("silence_sec", "补静音时长(s)"),
        ("first_chunk_delay_sec", "首包听到延迟(s)"),
        ("avg_chunk_sec", "平均块内容时长(s)"),
        ("avg_receive_gap_sec", "平均接收间隔(s)"),
        ("avg_gap_before_sec", "平均补静音(s)"),
        ("max_gap_before_sec", "最大补静音(s)"),
    ]

    print(f"{'指标':<20} | {name1:^18} | {name2:^18} | {'差值(1-2)':^18}")
    print("-" * 85)

    for key, label in rows:
        v1 = s1.get(key)
        v2 = s2.get(key)

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v1 - v2
            print(f"{label:<20} | {v1:>18.3f} | {v2:>18.3f} | {diff:>18.3f}")
        else:
            print(f"{label:<20} | {str(v1):>18} | {str(v2):>18} | {'-':>18}")


def print_first_n_chunks(data1: dict, data2: dict, n: int = 20):
    t1 = data1.get("timeline", [])
    t2 = data2.get("timeline", [])

    print("\n前 N 个 chunk 对比：")
    print(f"{'idx':<6} | {'100ms gap':>12} | {'200ms gap':>12} | {'100ms dur':>12} | {'200ms dur':>12}")
    print("-" * 65)

    for i in range(min(n, len(t1), len(t2))):
        g1 = t1[i].get("gap_before_sec", 0.0)
        g2 = t2[i].get("gap_before_sec", 0.0)
        d1 = t1[i].get("duration_sec", 0.0)
        d2 = t2[i].get("duration_sec", 0.0)
        print(f"{i:<6} | {g1:>12.3f} | {g2:>12.3f} | {d1:>12.3f} | {d2:>12.3f}")


if __name__ == "__main__":
    path_100 = r"data/output_volcengine_wav2_100ms/translate_audio_00001_timeline.json"
    path_200 = r"data/output_volcengine_wav2_200ms/translate_audio_00001_timeline.json"

    data_100 = load_timeline(path_100)
    data_200 = load_timeline(path_200)

    sum_100 = summarize_timeline(data_100)
    sum_200 = summarize_timeline(data_200)

    print_compare("100ms", sum_100, "200ms", sum_200)

    print_first_n_chunks(data_100, data_200, n=20)