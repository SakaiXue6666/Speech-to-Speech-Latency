import argparse
import json
from pathlib import Path
import wave
import numpy as np


def read_wav_int16(path: str):
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("仅支持 16-bit WAV")
    x = np.frombuffer(data, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x, sr


def read_pcm_int16(path: str, channels=1):
    x = np.fromfile(path, dtype=np.int16)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return x


def spectral_flatness(mag):
    eps = 1e-12
    gmean = np.exp(np.mean(np.log(mag + eps)))
    amean = np.mean(mag + eps)
    return float(gmean / amean)


def smoothness_score(x):
    if len(x) < 3:
        return 1e9
    diff = np.abs(np.diff(x.astype(np.int32))).mean()
    amp = np.abs(x.astype(np.int32)).mean() + 1.0
    return float(diff / amp)


def analyze_noise_features(x_int16, sr):
    x = x_int16.astype(np.float32) / 32768.0
    if len(x) < 1024:
        raise ValueError("音频太短")

    rms = float(np.sqrt(np.mean(x * x)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)).astype(np.float32))))

    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1 / sr)

    flat = spectral_flatness(spec)
    total_e = np.sum(spec ** 2) + 1e-12
    hf_e = np.sum(spec[freqs >= 4000] ** 2)
    hf_ratio = float(hf_e / total_e)
    clip_ratio = float(np.mean(np.abs(x_int16) >= 32000))

    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_flatness": flat,
        "hf_ratio_4k": hf_ratio,
        "clip_ratio": clip_ratio,
    }


def judge_noise(feat):
    score = 0
    reasons = []
    if feat["spectral_flatness"] > 0.35:
        score += 1
        reasons.append("谱平坦度偏高（更像噪声）")
    if feat["zcr"] > 0.15:
        score += 1
        reasons.append("过零率偏高（高频噪声明显）")
    if feat["hf_ratio_4k"] > 0.55:
        score += 1
        reasons.append("4kHz以上高频能量占比过高")
    if feat["clip_ratio"] > 0.01:
        score += 1
        reasons.append("削顶比例偏高（失真风险）")
    if feat["rms"] < 0.003:
        score += 1
        reasons.append("整体能量过低（近静音）")
    return score >= 3, score, reasons


def detect_audio_health(path, fmt="wav", sample_rate=24000, channels=1):
    if fmt == "wav":
        x, sr = read_wav_int16(path)
    else:
        x = read_pcm_int16(path, channels=channels)
        sr = sample_rate

    feat = analyze_noise_features(x, sr)
    likely_noise, score, reasons = judge_noise(feat)

    # 端序诊断：比较 byteswap 前后的平滑度
    x_be = x.byteswap().view(x.dtype)
    smooth_le = smoothness_score(x)
    smooth_be = smoothness_score(x_be)
    endian_hint = "little-endian 更合理" if smooth_le <= smooth_be else "big-endian 更合理（建议 byteswap）"

    suggest = []
    if "big-endian" in endian_hint:
        suggest.append("播放前对 PCM 做 16-bit byteswap")
    if feat["hf_ratio_4k"] > 0.7 and feat["zcr"] > 0.2:
        suggest.append("检查 target_audio.rate 是否与播放 rate 一致（16000/24000）")
    if feat["clip_ratio"] > 0.01:
        suggest.append("检查是否重复放大或解码参数错误")
    if likely_noise and not suggest:
        suggest.append("先做端序 A/B 测试，再做采样率 A/B 测试")

    return {
        "likely_noise": likely_noise,
        "noise_score": score,
        "sample_rate": sr,
        "num_samples": int(len(x)),
        "features": feat,
        "reasons": reasons,
        "endian_hint": endian_hint,
        "suggestions": suggest,
    }


def main():
    parser = argparse.ArgumentParser(description="检测 wav/pcm 是否疑似噪声")
    parser.add_argument("path", help="音频路径")
    parser.add_argument("--format", choices=["wav", "pcm"], default=None, help="不传则按文件后缀自动判断")
    parser.add_argument("--sample-rate", type=int, default=24000, help="pcm 时必填参考采样率")
    parser.add_argument("--channels", type=int, default=1, help="pcm 声道数")
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        suffix = Path(args.path).suffix.lower()
        if suffix == ".wav":
            fmt = "wav"
        elif suffix == ".pcm":
            fmt = "pcm"
        else:
            raise ValueError("无法从后缀判断格式，请显式传 --format wav 或 --format pcm")

    result = detect_audio_health(
        path=args.path,
        fmt=fmt,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

'''
python ast_python/try.py ast_python/output2/translate_audio_00001.wav
python ast_python/try.py ast_python/output/translate_audio_00001.pcm
python ast_python/try.py ast_python/output/translate_audio_00001.wav
'''

if __name__ == "__main__":
    main()