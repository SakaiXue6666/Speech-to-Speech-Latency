import asyncio
import audioop
import contextlib
import io
import uuid
import os
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Optional, List, Callable, Tuple
import websockets
from websockets import Headers
import sys
import time
import json
from google.protobuf.json_format import MessageToDict
from websockets.legacy.exceptions import InvalidStatusCode
import pyaudio
import wave as wf
import numpy as np

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算 python_protogen 目录的路径
protogen_dir = os.path.join(current_dir, "python_protogen")

# 只添加一次 python_protogen 目录
sys.path.append(protogen_dir)

# 现在可以直接导入所有模块
from products.understanding.ast.ast_service_pb2 import TranslateRequest, ReqParams, TranslateResponse
from common.events_pb2 import Type

# 根据 try.py 检测结果固定为 True：服务端 PCM 更像 big-endian，播放前需 byteswap
PCM_BYTESWAP = True

# Configuration
@dataclass
class Config:
    ws_url: str
    app_key: str
    access_key: str
    resource_id: str
    # Add other config fields as needed


@dataclass
class Audio:
    format: str = None
    rate: int = None
    bits: Optional[int] = None
    channel: Optional[int] = None
    binary_data: Optional[bytes] = None


@dataclass
class TranslateRequestData:
    session_id: str
    event: str
    source_audio: Optional[Audio] = None
    target_audio: Optional[Audio] = None
    mode: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None


@dataclass
class TranslateResponseData:
    event: str
    session_id: str
    sequence: int
    text: str
    data: bytes
    spk_chg: bool
    message: str = None


async def read_audio_chunks(audio_path: str, chunk_size: int) -> List[bytes]:
    """Read 16k/16bit/mono WAV PCM chunks."""
    chunks = []
    with wf.open(audio_path, "rb") as wav_in:
        if wav_in.getnchannels() != 1 or wav_in.getsampwidth() != 2 or wav_in.getframerate() != 16000:
            raise ValueError("输入必须是 16kHz/16bit/单声道 WAV")
        frames_per_chunk = chunk_size // 2  # 16-bit
        while True:
            chunk = wav_in.readframes(frames_per_chunk)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks


async def send_request(ws, request: TranslateRequestData):
    """Send request to WebSocket server"""
    # Implement your actual protocol serialization here
    request_data = TranslateRequest()
    request_data.request_meta.SessionID = request.session_id
    if request.event == "Type_StartSession":
        request_data.event = Type.StartSession
    elif request.event == "Type_TaskRequest":
        request_data.event = Type.TaskRequest
    elif request.event == "Type_FinishSession":
        request_data.event = Type.FinishSession
    request_data.user.uid = "ast_py_client"
    request_data.user.did = "ast_py_client"
    request_data.source_audio.format = "wav"
    request_data.source_audio.codec = "raw"
    request_data.source_audio.rate = 16000
    request_data.source_audio.bits = 16
    request_data.source_audio.channel = 1
    if request.source_audio.binary_data:
        request_data.source_audio.binary_data = request.source_audio.binary_data
    request_data.target_audio.format = "pcm"
    request_data.target_audio.rate = 24000
    request_data.target_audio.bits = 16
    request_data.target_audio.channel = 1
    request_data.request.mode = "s2s"
    request_data.request.source_language = "en"
    request_data.request.target_language = "zh"
    await ws.send(request_data.SerializeToString())  # Replace with your actual serialization


async def receive_message(ws) -> TranslateResponseData:
    """Receive and parse response from server"""
    response = await ws.recv()
    # Implement your actual protocol deserialization here
    # This is a placeholder - adapt to your actual response format
    Response_data = TranslateResponse()
    Response_data.ParseFromString(response)

    response_text = Response_data.text # Extract from actual response
    if Response_data.event == Type.UsageResponse:
        # 将 protobuf 消息转换为字典
        response_dict = MessageToDict(Response_data)
        # 以 JSON 格式打印，设置缩进和确保 ASCII 不转义
        #print("Response content (event=154):")
        #print(json.dumps(response_dict, indent=2, ensure_ascii=False))
        response_text = json.dumps(response_dict, indent=2, ensure_ascii=False)

    return TranslateResponseData(
        event=Response_data.event,  # Extract from actual response
        session_id=Response_data.response_meta.SessionID,  # Extract from actual response
        sequence=Response_data.response_meta.Sequence,  # Extract from actual response
        text=response_text,
        data=Response_data.data,  # Extract from actual response
        spk_chg=Response_data.spk_chg,  # Extract from actual response
        message= Response_data.response_meta.Message
    )


async def build_http_headers(conf: Config, conn_id: str) -> Headers:
    """Build WebSocket connection headers from config"""
    headers = Headers({
        "X-Api-App-Key": conf.app_key,
        "X-Api-Access-Key": conf.access_key,
        "X-Api-Resource-Id": conf.resource_id,
        "X-Api-Connect-Id": conn_id
    })
    return headers


def decode_ogg_chunk_to_pcm(opus_chunk: bytes):
    """Decode a single ogg_opus chunk to int16 mono PCM bytes."""
    try:
        import av
    except Exception:
        return b""
    try:
        container = av.open(io.BytesIO(opus_chunk), format="ogg")
        stream = container.streams.audio[0]
        out = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.dtype.kind == "f":
                arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
            elif arr.dtype != np.int16:
                arr = arr.astype(np.int16)
            if arr.ndim > 1:
                arr = arr.mean(axis=0).astype(np.int16)
            out.append(arr)
        container.close()
        if not out:
            return b""
        return np.concatenate(out).tobytes()
    except Exception:
        return b""


def _smoothness_score_i16(arr_i16: np.ndarray) -> float:
    if arr_i16.size < 3:
        return 1e9
    diff = np.abs(np.diff(arr_i16.astype(np.int32))).mean()
    amp = np.abs(arr_i16.astype(np.int32)).mean() + 1.0
    zcr = np.mean(np.abs(np.diff(np.signbit(arr_i16)).astype(np.float32)))
    # 越小越像语音（更平滑、过零率更低）
    return float(diff / amp + zcr * 2.0)


def _pcm_decoder_candidates(raw: bytes) -> List[Tuple[str, Callable[[bytes], bytes]]]:
    cands = []

    # int16 mono/stereo little/big
    def mk_i16_decoder(dtype_str: str, channels: int):
        def _decode(b: bytes):
            if len(b) < 2:
                return b""
            b2 = b[: len(b) - (len(b) % 2)]
            x = np.frombuffer(b2, dtype=np.dtype(dtype_str))
            if channels == 2 and x.size >= 2:
                x = x[: x.size - (x.size % 2)].reshape(-1, 2).mean(axis=1).astype(np.int16)
            else:
                x = x.astype(np.int16)
            return x.tobytes()
        return _decode

    cands.append(("i16-le-mono", mk_i16_decoder("<i2", 1)))
    cands.append(("i16-be-mono", mk_i16_decoder(">i2", 1)))
    cands.append(("i16-le-stereo", mk_i16_decoder("<i2", 2)))
    cands.append(("i16-be-stereo", mk_i16_decoder(">i2", 2)))

    # float32 mono/stereo little/big
    def mk_f32_decoder(dtype_str: str, channels: int):
        def _decode(b: bytes):
            if len(b) < 4:
                return b""
            b4 = b[: len(b) - (len(b) % 4)]
            x = np.frombuffer(b4, dtype=np.dtype(dtype_str))
            if channels == 2 and x.size >= 2:
                x = x[: x.size - (x.size % 2)].reshape(-1, 2).mean(axis=1)
            x = np.clip(x, -1.0, 1.0)
            x = (x * 32767.0).astype(np.int16)
            return x.tobytes()
        return _decode

    cands.append(("f32-le-mono", mk_f32_decoder("<f4", 1)))
    cands.append(("f32-be-mono", mk_f32_decoder(">f4", 1)))
    cands.append(("f32-le-stereo", mk_f32_decoder("<f4", 2)))
    cands.append(("f32-be-stereo", mk_f32_decoder(">f4", 2)))

    return cands


def select_best_pcm_decoder(first_chunk: bytes) -> Tuple[str, Callable[[bytes], bytes]]:
    best_name = "raw-passthrough"
    best_decoder = lambda b: b
    best_score = 1e18

    for name, decoder in _pcm_decoder_candidates(first_chunk):
        pcm = decoder(first_chunk)
        if len(pcm) < 320:
            continue
        arr = np.frombuffer(pcm, dtype=np.int16)
        score = _smoothness_score_i16(arr)
        if score < best_score:
            best_score = score
            best_name = name
            best_decoder = decoder
    return best_name, best_decoder


def render_pcm_with_gaps(timeline, pcm_chunks, rate=24000, t0=None):
    """按 heard_start 渲染输出音频：gap 自动填静音。
    t0 默认取首个 heard_start；传 first_send_timestamp 时可从第一个 src 发送开始计时。
    """
    if not timeline or not pcm_chunks:
        return np.array([], dtype=np.int16)

    if t0 is None:
        t0 = timeline[0]["heard_start"]
    rendered_parts = []
    write_cursor = 0

    for ent, chunk in zip(timeline, pcm_chunks):
        samples = np.frombuffer(chunk, dtype=np.int16)
        target_pos = max(0, int(round((ent["heard_start"] - t0) * rate)))
        if target_pos > write_cursor:
            silence_len = target_pos - write_cursor
            rendered_parts.append(np.zeros(silence_len, dtype=np.int16))
            write_cursor += silence_len
        rendered_parts.append(samples)
        write_cursor += len(samples)

    return np.concatenate(rendered_parts) if rendered_parts else np.array([], dtype=np.int16)

async def translate_v4(conf: Config,audio_path: str, n: int, out_dir: str = "output"):
    """Main translation function"""
    # Read audio chunks
    try:
        audio_chunks = await read_audio_chunks(audio_path, 6400)  # 200ms chunks @ 16kHz/16bit/mono
    except Exception as e:
        logging.error(f"Read audio chunks from file: {e}")
        return

    # Connect to server
    try:
        conn_id = str(uuid.uuid4())
        headers = await build_http_headers(conf, conn_id)
        response_headers = None
        conn = await websockets.connect(
            conf.ws_url,
            additional_headers=headers,
            max_size = 1000000000,
            ping_interval = None
        )
        logging.info(f"Connected to server (logg id={conn.response.headers.get('X-Tt-Logid')}   {n})")
        log_id = conn.response.headers.get('X-Tt-Logid')
    except Exception as e:
        logging.error(f"Connect: {e}")
        logging.error(f"Response headers: {e.response.body}")
        logging.error(f"Response logid: {e.args[0].headers['X-Tt-Logid']}")
        return

    session_id = str(uuid.uuid4())

    # Start session
    start_request = TranslateRequestData(
        session_id=session_id,
        event="Type_StartSession",
        source_audio=Audio(format="wav", rate=16000, bits=16, channel=1),
        target_audio=Audio(format="pcm", rate=24000, bits=16, channel=1),
        mode="s2s",
        source_language="zh",
        target_language="en"
    )

    try:
        await send_request(conn, start_request)
        resp = await receive_message(conn)
        if resp.event != Type.SessionStarted:
            logging.error(f"Unexpected response logid: {log_id}")
            logging.error(f"Unexpected response: {resp.event}")
            logging.error(f"Unexpected response message: {resp.message}")
            await conn.close()
            return
        logging.info(f"Session (ID={session_id}) started.")
    except Exception as e:
        logging.error(f"Start session: {e}")
        await conn.close()
        return

    # Send audio chunks
    first_send_timestamp = None

    async def send_audio_chunks():
        nonlocal first_send_timestamp
        try:
            for i, chunk in enumerate(audio_chunks):
                logging.info(f"Sending chunk: {len(chunk)}")
                if first_send_timestamp is None:
                    first_send_timestamp = time.time()
                chunk_request = TranslateRequestData(
                    session_id=session_id,
                    event="Type_TaskRequest",
                    source_audio=Audio(binary_data=chunk)
                )
                await send_request(conn, chunk_request)
                await asyncio.sleep(len(chunk) / (16000 * 2))
            # Send finish session
            finish_request = TranslateRequestData(
                session_id=session_id,
                event="Type_FinishSession",
                source_audio = Audio()
            )
            await send_request(conn, finish_request)
            logging.info("FinishSession request is sent.")
        except Exception as e:
            logging.error(f"Error sending chunks: {e}")

    # Start sender task
    sender_task = asyncio.create_task(send_audio_chunks())

    # Receive responses
    recv_audio = bytearray()
    recv_text = []
    timeline = []
    pcm_chunks = []
    chunk_id = 0
    prev_heard_end = 0.0
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )
    session_failed = False
    audio_mode = "unknown"  # unknown/pcm/ogg
    pcm_decoder_name = None
    pcm_decoder: Optional[Callable[[bytes], bytes]] = None

    try:
        while True:
            #logging.info("Waiting for message...")
            resp = await receive_message(conn) ## ✌

            logging.info(
                f"Receive message (event={resp.event}, session_id={resp.session_id}): "
                f"seq: {resp.sequence}, text:{resp.text}, audio data length:{len(resp.data)}, spk_chg: {resp.spk_chg}"
            )
            if resp.event == Type.SessionFailed or resp.event == Type.SessionCanceled:
                logging.error(f"Session failed, message: {resp.message} logid: {log_id} event: {resp.event} message: {resp.message}")
                session_failed = True
                break

            if resp.event == Type.SessionFinished:
                break
            if resp.event == Type.TTSResponse and resp.data:
                pcm_bytes = resp.data
                if audio_mode == "unknown":
                    audio_mode = "ogg" if pcm_bytes.startswith(b"OggS") else "pcm"
                    logging.info(f"Detected TTS audio mode: {audio_mode}")
                if audio_mode == "ogg":
                    pcm_bytes = decode_ogg_chunk_to_pcm(resp.data)
                    if not pcm_bytes:
                        continue
                else:
                    if pcm_decoder is None:
                        pcm_decoder_name, pcm_decoder = select_best_pcm_decoder(resp.data)
                        logging.info(f"Selected PCM decoder: {pcm_decoder_name}")
                    pcm_bytes = pcm_decoder(resp.data) if pcm_decoder else resp.data
                if len(pcm_bytes) % 2 == 1:
                    pcm_bytes = pcm_bytes[:-1]
                if pcm_bytes:
                    # 仅在原始 int16 PCM 通路时使用 byteswap；自动解码器若已按端序处理则不再重复 swap
                    if audio_mode == "pcm" and PCM_BYTESWAP and pcm_decoder_name in ("i16-le-mono", "i16-le-stereo"):
                        pcm_bytes = audioop.byteswap(pcm_bytes, 2)

                    receive_ts = time.time()
                    duration_sec = len(pcm_bytes) / (2 * 24000)
                    heard_start = max(receive_ts, prev_heard_end)
                    heard_end = heard_start + duration_sec
                    prev_heard_end = heard_end

                    timeline.append({
                        "chunk_id": chunk_id,
                        "receive_timestamp": float(receive_ts),
                        "duration_sec": float(duration_sec),
                        "heard_start": float(heard_start),
                        "heard_end": float(heard_end),
                        "write_before": float(heard_start),
                        "write_after": float(heard_end),
                        "output_time_sec": float(heard_start - first_send_timestamp) if first_send_timestamp else None,
                        "receive_time_sec": float(receive_ts - first_send_timestamp) if first_send_timestamp else None,
                    })
                    pcm_chunks.append(pcm_bytes)
                    chunk_id += 1
                    await asyncio.to_thread(stream.write, pcm_bytes)
                    recv_audio.extend(pcm_bytes)
            if resp.event != Type.UsageResponse and resp.text:
                recv_text.append(resp.text)
    except Exception as e:
        logging.error(f"Receive message error: {e}")
    finally:
        if session_failed and not sender_task.done():
            sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
        await conn.close()
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            pass

    # Save results
    if recv_audio:
        os.makedirs(out_dir, exist_ok=True)
        output_path = Path(out_dir) / f"translate_audio_{n:05}.wav"
        # timeline_path = Path(out_dir) / f"translate_audio_{n:05}_timeline.json"
        try:
            # 1) 原始实时播放顺序落盘（不补 gap）
            with open(str(output_path), "wb") as f:
                with wf.open(f, "wb") as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(24000)
                    wav_out.writeframes(bytes(recv_audio))

            # 2) 按真实 heard_start 渲染，gap 补静音；时间零点对齐到第一个 src 发送时刻
            t0 = first_send_timestamp or 0.0

            # 先补齐 timeline（和下面保存 json 一样的逻辑），确保每条都有 heard_start
            prev_heard_end_ts = t0
            for ent in timeline:
                if ent.get("heard_start") is None:
                    ent["heard_start"] = float(max(ent["receive_timestamp"], prev_heard_end_ts))
                    ent["heard_end"] = float(ent["heard_start"] + ent["duration_sec"])
                prev_heard_end_ts = ent["heard_end"]

            # 按 heard_start 时间轴渲染：gap 填静音，overlap 紧接
            write_cursor = 0  # 当前已写到的采样点位置
            rendered_parts: List[np.ndarray] = []
            rate = 24000

            for i, chunk_bytes in enumerate(pcm_chunks):
                chunk_samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                ent = timeline[i] if i < len(timeline) else None

                if ent is not None and ent.get("heard_start") is not None:
                    target_pos = int(round((ent["heard_start"] - t0) * rate))
                else:
                    target_pos = write_cursor

                if target_pos > write_cursor:
                    silence_len = target_pos - write_cursor
                    rendered_parts.append(np.zeros(silence_len, dtype=np.int16))
                    write_cursor += silence_len

                rendered_parts.append(chunk_samples)
                write_cursor += len(chunk_samples)

            if rendered_parts:
                rendered = np.concatenate(rendered_parts)
            else:
                rendered = np.array([], dtype=np.int16)
            gap_output_path = Path(out_dir) / f"translate_audio_{n:05}_gap.wav"
            with open(str(gap_output_path), "wb") as f:
                with wf.open(f, "wb") as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(24000)
                    wav_out.writeframes(rendered.tobytes())
            logging.info(f"[INFO] 已保存 tgt 音频（含 gap 静音）: {gap_output_path}")

            # payload = {
            #     "first_send_timestamp": first_send_timestamp,
            #     "timeline": timeline,
            # }
            # with open(str(timeline_path), "w", encoding="utf-8") as tf:
            #     json.dump(payload, tf, ensure_ascii=False, indent=2)

            logging.info(f"Session finished, audio is saved as: {output_path}")
            logging.info(f"Session finished, gap-audio is saved as: {gap_output_path}")
            # logging.info(f"Session finished, timeline is saved as: {timeline_path}")
            logging.info(f"Session finished, text is: {' '.join(recv_text)}")

            # 保存 manifest.jsonl（像 qwen_livetranslate_wav4 一样）
            manifest_path = Path(out_dir) / "manifest.jsonl"
            manifest_record = {
                "src": audio_path,
                "tgt": str(gap_output_path),
            }
            with open(str(manifest_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            logging.info(f"[INFO] 已追加记录到: {manifest_path}")
        except Exception as e:
            logging.error(f"Save audio file: {e}")
    else:
        logging.error("Session finished, no audio data is received.")

'''
pip install imageio-ffmpeg
python ast_python/ast_demo_pcm2.py
'''

# Example usage
async def main():
    conf = Config(ws_url="wss://openspeech.bytedance.com/api/v4/ast/v2/translate",
                   app_key="2476390117",
                   access_key="J0QUKxRJb9j32MYmWOgoQ7d-n_jLI5Uk",
                  resource_id="volc.service_type.10053")
    start = time.time()
    # task = asyncio.create_task(translate_v4(conf, "ast_python/test_audio.wav", 1, "ast_python/output2"))
    task = asyncio.create_task(translate_v4(conf, "data/input/acl_6060_dev/2022.acl-long.268.wav", 1, "data/output_volcengine_wav"))
    
    await  task
    end = time.time()
    logging.info(f"Total time: {end - start:.6f} 秒")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
