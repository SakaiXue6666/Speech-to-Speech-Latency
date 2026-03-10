import asyncio
import subprocess
import uuid
import os
import io
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Optional, List
import websockets
from websockets import Headers
import sys
import time
import json
from google.protobuf.json_format import MessageToDict
from websockets.legacy.exceptions import InvalidStatusCode
import numpy as np
import wave as wf
import pyaudio

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算 python_protogen 目录的路径
protogen_dir = os.path.join(current_dir, "python_protogen")

# 只添加一次 python_protogen 目录
sys.path.append(protogen_dir)

# 现在可以直接导入所有模块
from products.understanding.ast.ast_service_pb2 import TranslateRequest, ReqParams, TranslateResponse
from common.events_pb2 import Type

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
    """Read audio file in chunks"""
    chunks = []
    with open(audio_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
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
    request_data.source_audio.rate = 16000
    request_data.source_audio.bits = 16
    request_data.source_audio.channel = 1
    if request.source_audio.binary_data:
        request_data.source_audio.binary_data = request.source_audio.binary_data
    request_data.target_audio.format = "ogg_opus"
    request_data.target_audio.rate = 24000
    request_data.request.mode = "s2s"
    request_data.request.source_language = "zh"
    request_data.request.target_language = "en"
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


async def translate_v4(conf: Config,audio_path: str, n: int, out_dir: str = "output"):
    """Main translation function"""
    # Read audio chunks
    try:
        audio_chunks = await read_audio_chunks(audio_path, 3200)  # 100ms chunks
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
        target_audio=Audio(format="ogg_opus", rate=24000),
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

    # Send audio chunks
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
                await asyncio.sleep(0.1)  # 100ms delay
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

    # Initialize audio playback
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )

    # Receive responses
    recv_audio = bytearray()
    recv_text = []
    timeline = []
    chunk_id = 0
    prev_heard_end = 0.0
    first_send_timestamp = None  # 会在发送开始时设置

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
                raise Exception("Session faild")
                break

            if resp.event == Type.SessionFinished:
                break
            if resp.event == Type.TTSResponse and resp.data:
                # 记录接收时间戳
                receive_ts = time.time()
                
                # 用 len(resp.data) 计算估算时长
                # Opus 24kHz 压缩比通常 6:1 到 20:1，根据数据大小动态调整
                data_size_kb = len(resp.data) / 1024.0
                if data_size_kb < 2:  # 小块，假设高压缩比
                    bytes_per_sec = 3000.0  # ~15:1 压缩比
                elif data_size_kb < 5:  # 中等块
                    bytes_per_sec = 4000.0  # ~12:1 压缩比  
                else:  # 大块，假设低压缩比
                    bytes_per_sec = 6000.0  # ~8:1 压缩比
                
                estimated_duration_sec = len(resp.data) / bytes_per_sec
                
                # 使用接收时间和duration估计真实听到时间
                heard_start = max(receive_ts, prev_heard_end)
                heard_end = heard_start + estimated_duration_sec
                prev_heard_end = heard_end
                
                # 记录 timeline
                timeline.append({
                    "chunk_id": chunk_id,
                    "receive_timestamp": float(receive_ts),
                    "estimated_duration_sec": float(estimated_duration_sec),
                    "data_length": len(resp.data),
                    "heard_start": float(heard_start),
                    "heard_end": float(heard_end),
                    "output_time_sec": float(heard_start - first_send_timestamp) if first_send_timestamp else None,
                    "receive_time_sec": float(receive_ts - first_send_timestamp) if first_send_timestamp else None,
                })
                
                # 可选：实时播放（需要PCM解码，影响性能）
                # 如果不需要实时播放，可以注释掉下面这部分以提高性能
                # pcm_bytes = decode_ogg_chunk_to_pcm(resp.data)
                # if pcm_bytes:
                #     await asyncio.to_thread(stream.write, pcm_bytes)
                
                recv_audio.extend(resp.data)
                chunk_id += 1
                
            if resp.event != Type.UsageResponse and resp.text:
                recv_text.append(resp.text)
    except Exception as e:
        logging.error(f"Receive message error: {e}")
    finally:
        await sender_task  # Ensure sender completes
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
        opus_output_path = Path(out_dir) / f"translate_audio_{n:05}.opus"
        try:
            # 1) 保存原始 opus 文件
            with open(opus_output_path, 'wb') as f:
                f.write(recv_audio)
            logging.info(f"[INFO] 已保存原始 opus 音频: {opus_output_path}")

            # 2) 用 ffmpeg 转成 wav
            wav_output_path = opus_output_path.with_suffix(".wav")
            try:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = "ffmpeg"
                subprocess.run([
                    ffmpeg_exe, "-y", "-i", str(opus_output_path), 
                    "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", str(wav_output_path)
                ], check=True, capture_output=True)
                logging.info(f"[INFO] 已转码为 wav: {wav_output_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logging.warning(f"opus -> wav 转换失败: {e}")
                wav_output_path = None

            # 3) 根据 timeline 补齐 gap（使用估算的时长信息）
            if timeline and first_send_timestamp and wav_output_path and wav_output_path.exists():
                t0 = first_send_timestamp
                
                # 读取完整的WAV文件
                with wf.open(str(wav_output_path), "rb") as wav_in:
                    sample_rate = wav_in.getframerate()
                    channels = wav_in.getnchannels()
                    frames = wav_in.readframes(wav_in.getnframes())
                    all_samples = np.frombuffer(frames, dtype=np.int16)
                    if channels == 2:
                        all_samples = all_samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # 使用 timeline 中的估算时长来分割音频
                rendered_parts = []
                write_cursor = 0
                
                for ent in timeline:
                    # 根据估算的时长计算这块音频应该有多少采样点
                    estimated_samples = int(round(ent["estimated_duration_sec"] * sample_rate))
                    
                    # 从完整音频中取出对应长度的片段
                    chunk_audio = all_samples[write_cursor:write_cursor + estimated_samples]
                    
                    # 计算在时间轴上的目标位置
                    target_pos = int(round((ent["heard_start"] - t0) * sample_rate))
                    
                    # 补齐 gap
                    if target_pos > write_cursor:
                        silence_len = target_pos - write_cursor
                        rendered_parts.append(np.zeros(silence_len, dtype=np.int16))
                        write_cursor += silence_len
                    
                    # 添加音频块
                    rendered_parts.append(chunk_audio)
                    write_cursor += len(chunk_audio)
                
                if rendered_parts:
                    rendered = np.concatenate(rendered_parts)
                    gap_output_path = Path(out_dir) / f"translate_audio_{n:05}_gap.wav"
                    with open(str(gap_output_path), "wb") as f:
                        with wf.open(f, "wb") as wav_out:
                            wav_out.setnchannels(1)
                            wav_out.setsampwidth(2)
                            wav_out.setframerate(sample_rate)
                            wav_out.writeframes(rendered.tobytes())
                    logging.info(f"[INFO] 已保存 gap 补齐音频: {gap_output_path}")

            logging.info(f"Session finished, opus is saved as: {opus_output_path}")
            if wav_output_path:
                logging.info(f"Session finished, wav is saved as: {wav_output_path}")
            logging.info(f"Session finished, text is: {' '.join(recv_text)}")

            # 保存 manifest.jsonl
            manifest_path = Path(out_dir) / "manifest.jsonl"
            manifest_record = {
                "src": audio_path,
                "tgt": str(gap_output_path) if 'gap_output_path' in locals() and gap_output_path.exists() else str(wav_output_path or opus_output_path),
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
python ast_python/ast_demo.py
'''

# Example usage
async def main():
    conf = Config(ws_url="wss://openspeech.bytedance.com/api/v4/ast/v2/translate",
                   app_key="2476390117",
                   access_key="J0QUKxRJb9j32MYmWOgoQ7d-n_jLI5Uk",
                  resource_id="volc.service_type.10053")
    start = time.time()
    task = asyncio.create_task(translate_v4(conf, "data/input/acl_6060_dev/2022.acl-long.268.wav", 1, "ast_python/output_demo_268"))
    
    await  task
    end = time.time()
    logging.info(f"Total time: {end - start:.6f} 秒")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())