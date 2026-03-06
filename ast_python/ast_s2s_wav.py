# -*- coding: utf-8 -*-
"""
AST 同声传译 S2S（Speech-to-Speech）脚本：从 wav 流式发送，得到 tgt wav、timeline、manifest。
与 qwen_livetranslate_wav3.py 的输出格式对齐：wav、timeline.json、manifest.jsonl。
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import io
import numpy as np
import soundfile as sf
import websockets
from websockets import Headers

current_dir = os.path.dirname(os.path.abspath(__file__))
protogen_dir = os.path.join(current_dir, "python_protogen")
if protogen_dir not in sys.path:
    sys.path.append(protogen_dir)

from products.understanding.ast.ast_service_pb2 import TranslateRequest, TranslateResponse
from common.events_pb2 import Type


'''
依赖: pip install grpcio websockets soundfile "protobuf>=6.31" av
（av 用于将 ogg_opus 解码为 PCM 再保存 wav，与官方 demo 一致，避免 pcm 格式/采样率不一致导致慢速和噪音）

python ast_python/ast_s2s_wav.py data/input/acl_6060_dev/2022.acl-long.268.wav --out-dir data/output_volcengine_wav --src-lang en --tgt-lang zh
'''

# 显式设置 key：取消下面两行注释并填入；也可在运行前设置环境变量
os.environ["AST_APP_KEY"] = "2476390117"
os.environ["AST_ACCESS_KEY"] = "J0QUKxRJb9j32MYmWOgoQ7d-n_jLI5Uk"

# ---------------------------------------------------------------------------
# 配置与数据结构
# ---------------------------------------------------------------------------
@dataclass
class Config:
    ws_url: str = "wss://openspeech.bytedance.com/api/v4/ast/v2/translate"
    app_key: str = ""
    access_key: str = ""
    resource_id: str = "volc.service_type.10053"


@dataclass
class Audio:
    format: str = None
    rate: int = None
    bits: Optional[int] = None
    channel: Optional[int] = None
    binary_data: Optional[bytes] = None


# 源音频：16k, 16bit, mono wav/pcm（与文档一致）
INPUT_RATE = 16000
INPUT_CHUNK = 3200  # 约 200ms @ 16k
# 目标音频：与官方 demo 一致用 ogg_opus，解码后保存 wav（避免 pcm 格式/采样率不符导致慢速和噪音）
TARGET_AUDIO_FORMAT = "ogg_opus"  # 可选 "pcm"；推荐 ogg_opus
TARGET_RATE = 24000
TARGET_BYTES_PER_SAMPLE = 2  # 16bit，仅 pcm 时用


def decode_ogg_opus_to_pcm(opus_bytes: bytes):
    """将 ogg_opus 字节流解码为 16bit 单声道 PCM，返回 (samples_1d, sample_rate)。"""
    import av
    buf = io.BytesIO(opus_bytes)
    container = av.open(buf, format="ogg")
    stream = container.streams.audio[0]
    rate = stream.sample_rate
    out = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()  # (channels, samples), float 或 int
        if arr.dtype.kind == "f":
            arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
        elif arr.dtype != np.int16:
            arr = arr.astype(np.int16)
        if arr.shape[0] > 1:
            arr = arr.mean(axis=0).astype(np.int16)
        else:
            arr = arr[0]
        out.append(arr)
    container.close()
    if not out:
        return np.array([], dtype=np.int16), rate
    return np.concatenate(out), rate


def build_headers(conf: Config, conn_id: str) -> Headers:
    return Headers({
        "X-Api-App-Key": conf.app_key,
        "X-Api-Access-Key": conf.access_key,
        "X-Api-Resource-Id": conf.resource_id,
        "X-Api-Connect-Id": conn_id,
    })


def build_start_request(session_id: str, source_language: str = "zh", target_language: str = "en"):
    """建联：mode=s2s，目标音频与官方 demo 一致用 ogg_opus，解码后保存 wav。"""
    req = TranslateRequest()
    req.request_meta.SessionID = session_id
    req.event = Type.StartSession
    req.user.uid = "ast_s2s_wav"
    req.user.did = "ast_s2s_wav"
    req.source_audio.format = "wav"
    req.source_audio.rate = INPUT_RATE
    req.source_audio.bits = 16
    req.source_audio.channel = 1
    req.target_audio.format = TARGET_AUDIO_FORMAT
    req.target_audio.rate = TARGET_RATE
    req.request.mode = "s2s"
    req.request.source_language = source_language
    req.request.target_language = target_language
    return req


def build_task_request(session_id: str, audio_chunk: bytes):
    req = TranslateRequest()
    req.request_meta.SessionID = session_id
    req.event = Type.TaskRequest
    req.source_audio.format = "wav"
    req.source_audio.rate = INPUT_RATE
    req.source_audio.bits = 16
    req.source_audio.channel = 1
    req.source_audio.binary_data = audio_chunk
    return req


def build_finish_request(session_id: str):
    req = TranslateRequest()
    req.request_meta.SessionID = session_id
    req.event = Type.FinishSession
    req.source_audio.format = "wav"
    req.source_audio.rate = INPUT_RATE
    req.source_audio.bits = 16
    req.source_audio.channel = 1
    return req


async def read_wav_as_chunks(audio_path: str, chunk_samples: int = INPUT_CHUNK) -> List[bytes]:
    """读取 wav，重采样到 16k 单声道，按 chunk 返回 bytes（仅 PCM，无头）。"""
    data, sr = sf.read(audio_path, dtype="int16", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True).astype(np.int16)
    data = data[:, 0]
    if sr != INPUT_RATE:
        n = len(data)
        new_n = int(n * INPUT_RATE / sr)
        data = np.interp(
            np.linspace(0, n - 1, new_n),
            np.arange(n),
            data.astype(np.float64),
        ).astype(np.int16)
    chunks = []
    for i in range(0, len(data), chunk_samples):
        chunk = data[i : i + chunk_samples]
        if chunk.size > 0:
            chunks.append(chunk.tobytes())
    return chunks


async def run_s2s(
    conf: Config,
    audio_path: str,
    out_dir: str,
    source_language: str = "zh",
    target_language: str = "en",
    manifest_src_path: Optional[str] = None,
    save_tgt_wav_path: Optional[str] = None,
    save_timeline_path: Optional[str] = None,
):
    """
    执行 S2S：发送 wav 流 → 收集 TTS 音频与时间线 → 保存 wav、timeline、manifest。
    """
    manifest_src_path = manifest_src_path or audio_path
    base = Path(audio_path).stem
    if save_tgt_wav_path is None:
        save_tgt_wav_path = os.path.join(out_dir, f"{base}_tgt.wav")
    if save_timeline_path is None:
        save_timeline_path = os.path.join(out_dir, f"{base}_timeline.json")

    tgt_audio_chunks: List[bytes] = []
    tgt_timeline: List[dict] = []
    sentence_timeline: List[dict] = []
    first_send_timestamp: Optional[float] = None
    current_offset_sec = 0.0
    current_sentence_start_ms: Optional[int] = None
    chunk_id_counter = 0

    try:
        audio_chunks = await read_wav_as_chunks(audio_path)
    except Exception as e:
        logging.error(f"读取音频失败: {e}")
        return

    conn_id = str(uuid.uuid4())
    headers = build_headers(conf, conn_id)
    try:
        conn = await websockets.connect(
            conf.ws_url,
            additional_headers=headers,
            max_size=1000000000,
            ping_interval=None,
        )
        log_id = conn.response.headers.get("X-Tt-Logid")
        logging.info(f"已连接, X-Tt-Logid={log_id}")
    except Exception as e:
        logging.error(f"连接失败: {e}")
        if "401" in str(e):
            logging.error("HTTP 401 表示鉴权失败，请检查：1) AST_APP_KEY / AST_ACCESS_KEY 是否在代码或环境中正确设置；2) 是否使用火山引擎控制台提供的 APP ID 与 Access Token；3) Token 是否过期。")
        return

    session_id = str(uuid.uuid4())
    await conn.send(build_start_request(session_id, source_language, target_language).SerializeToString())

    resp = TranslateResponse()
    resp.ParseFromString(await conn.recv())
    if resp.event != Type.SessionStarted:
        logging.error(f"建联失败: event={resp.event}, message={getattr(resp.response_meta, 'Message', '')}")
        await conn.close()
        return
    logging.info("Session 已启动")

    async def send_audio():
        nonlocal first_send_timestamp
        for i, chunk in enumerate(audio_chunks):
            if first_send_timestamp is None:
                first_send_timestamp = time.time()
            await conn.send(build_task_request(session_id, chunk).SerializeToString())
            await asyncio.sleep(chunk.__len__() / (INPUT_RATE * 2))  # 按 16bit 换算时长
        await conn.send(build_finish_request(session_id).SerializeToString())
        logging.info("已发送 FinishSession")

    sender = asyncio.create_task(send_audio())

    try:
        while True:
            raw = await conn.recv()
            resp = TranslateResponse()
            resp.ParseFromString(raw)
            ev = resp.event

            if ev == Type.SessionFailed or ev == Type.SessionCanceled:
                logging.error(f"Session 异常: event={ev}, message={getattr(resp.response_meta, 'Message', '')}")
                break
            if ev == Type.SessionFinished:
                break

            if ev == Type.TTSSentenceStart:
                current_sentence_start_ms = resp.start_time if resp.start_time else None

            if ev == Type.TTSResponse and resp.data:
                receive_ts = time.time()
                # ogg_opus 时无法从字节长直接算时长，先记 0，保存时用解码后总时长均分
                if TARGET_AUDIO_FORMAT == "pcm":
                    duration_sec = len(resp.data) / (TARGET_BYTES_PER_SAMPLE * TARGET_RATE)
                else:
                    duration_sec = 0.0
                tgt_timeline.append({
                    "chunk_id": chunk_id_counter,
                    "offset_sec": round(current_offset_sec, 6),
                    "duration_sec": round(duration_sec, 6),
                    "receive_timestamp": round(receive_ts, 6),
                    "write_before": None,
                    "write_after": None,
                    "heard_start": None,
                    "heard_end": None,
                    "output_time_sec": None,
                })
                chunk_id_counter += 1
                current_offset_sec += duration_sec
                tgt_audio_chunks.append(bytes(resp.data))

            if ev == Type.TTSSentenceEnd:
                # 服务端 start_time/end_time 为毫秒，若模型考虑句间 gap 会体现在此
                st_ms = getattr(resp, "start_time", None) or current_sentence_start_ms
                et_ms = getattr(resp, "end_time", None)
                if st_ms is not None or et_ms is not None:
                    sentence_timeline.append({"start_time_ms": st_ms, "end_time_ms": et_ms})
                current_sentence_start_ms = None

            if ev == Type.SourceSubtitleEnd or ev == Type.TranslationSubtitleEnd:
                if resp.text:
                    logging.info(f"[字幕] {resp.text[:80]}...")
    except Exception as e:
        logging.error(f"接收异常: {e}")
    finally:
        await sender
        await conn.close()

    # ---------- 保存 wav ----------
    os.makedirs(out_dir, exist_ok=True)
    if tgt_audio_chunks:
        raw = b"".join(tgt_audio_chunks)
        if TARGET_AUDIO_FORMAT == "ogg_opus":
            try:
                samples, sample_rate = decode_ogg_opus_to_pcm(raw)
                sf.write(save_tgt_wav_path, samples, sample_rate, subtype="PCM_16")
                # 用解码后的总时长均分到各 chunk，修正 timeline 的 duration_sec / offset_sec
                total_duration_sec = len(samples) / sample_rate
                n = len(tgt_timeline)
                if n > 0:
                    chunk_dur = total_duration_sec / n
                    for i, ent in enumerate(tgt_timeline):
                        ent["duration_sec"] = round(chunk_dur, 6)
                        ent["offset_sec"] = round(i * chunk_dur, 6)
                logging.info(f"已保存 tgt wav (ogg_opus→PCM, {sample_rate}Hz): {save_tgt_wav_path}")
            except Exception as e:
                logging.error(f"ogg_opus 解码失败: {e}，请检查是否已安装 av (pip install av)")
                raise
        else:
            samples = np.frombuffer(raw, dtype=np.int16)
            sf.write(save_tgt_wav_path, samples, TARGET_RATE, subtype="PCM_16")
            logging.info(f"已保存 tgt wav: {save_tgt_wav_path}")
    else:
        logging.warning("未收到 TTS 数据，未写入 wav")

    # ---------- 模拟播放队列，计算 heard_start / heard_end / output_time_sec ----------
    # 与 qwen_livetranslate_wav3.py 的播放线程逻辑一致：
    #   heard_start = max(receive_timestamp, prev_heard_end)
    #   heard_end   = heard_start + duration_sec
    #   output_time_sec = heard_start - first_send_timestamp
    # 连续到达的块会排队；有 gap（缓冲区空了）则以到达时间为准。
    if tgt_timeline and first_send_timestamp is not None:
        prev_heard_end = 0.0
        for ent in tgt_timeline:
            recv_ts = ent["receive_timestamp"]
            dur = ent["duration_sec"]
            heard_start = max(recv_ts, prev_heard_end)
            heard_end = heard_start + dur
            prev_heard_end = heard_end

            ent["heard_start"] = round(heard_start, 6)
            ent["heard_end"] = round(heard_end, 6)
            ent["write_before"] = round(heard_start, 6)
            ent["write_after"] = round(heard_end, 6)
            ent["output_time_sec"] = round(heard_start - first_send_timestamp, 6)

    # ---------- 保存 timeline ----------
    payload = {
        "first_send_timestamp": first_send_timestamp,
        "timeline": tgt_timeline,
        "sentence_timeline": sentence_timeline,
    }
    with open(save_timeline_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info(f"已保存 timeline: {save_timeline_path}")

    # ---------- 追加 manifest（格式与 qwen_livetranslate_wav3 一致：使用传入的路径，不转绝对路径）----------
    if manifest_src_path and tgt_audio_chunks:
        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        record = {
            "src": manifest_src_path,
            "tgt": save_tgt_wav_path,
            "tgt_timeline": save_timeline_path,
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"已追加 manifest: {manifest_path}")


async def main():

    import argparse

    p = argparse.ArgumentParser(description="AST S2S: wav → tgt wav + timeline + manifest")
    p.add_argument("audio_path", help="输入 wav 路径")
    p.add_argument("--out-dir", default="output_ast_s2s", help="输出目录")
    p.add_argument("--src-lang", default="zh", help="源语言")
    p.add_argument("--tgt-lang", default="en", help="目标语言")
    p.add_argument("--app-key", default=os.environ.get("AST_APP_KEY", ""), help="APP Key 或环境变量 AST_APP_KEY")
    p.add_argument("--access-key", default=os.environ.get("AST_ACCESS_KEY", ""), help="Access Key 或环境变量 AST_ACCESS_KEY")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    conf = Config(app_key=args.app_key, access_key=args.access_key)
    if not conf.app_key or not conf.access_key:
        logging.error("请设置 --app-key 与 --access-key，或环境变量 AST_APP_KEY、AST_ACCESS_KEY")
        return

    await run_s2s(
        conf,
        args.audio_path,
        args.out_dir,
        source_language=args.src_lang,
        target_language=args.tgt_lang,
    )


if __name__ == "__main__":
    asyncio.run(main())
