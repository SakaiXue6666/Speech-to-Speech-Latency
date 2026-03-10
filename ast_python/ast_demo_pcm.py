import asyncio
import subprocess
import uuid
import os
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

import wave

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

# +++++++++++++++++++++++++++++++++++++++++\
def detect_audio_signature(data: bytes) -> str:
    if not data:
        return "empty"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF":
        return "wav"
    return "raw_or_unknown"


def save_pcm_as_wav(pcm_bytes: bytes, wav_path: Path, sample_rate: int = 24000, sample_width: int = 2, channels: int = 1):
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)  # 2=int16, 4=int32
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
# +++++++++++++++++++++++++++++++++++++++++/


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
    # request_data.source_audio.format = "wav"
    # request_data.source_audio.rate = 16000
    # request_data.source_audio.bits = 16
    # request_data.source_audio.channel = 1
    # if request.source_audio.binary_data:
    #     request_data.source_audio.binary_data = request.source_audio.binary_data
    # request_data.target_audio.format = "pcm"
    # request_data.target_audio.rate = 24000

    # 保留你送 wav 的逻辑，但不要写死 target/source 配置覆盖外部参数
    if request.source_audio:
        request_data.source_audio.format = request.source_audio.format or "wav"
        request_data.source_audio.rate = request.source_audio.rate or 16000
        request_data.source_audio.bits = request.source_audio.bits or 16
        request_data.source_audio.channel = request.source_audio.channel or 1
        if request.source_audio.binary_data:
            request_data.source_audio.binary_data = request.source_audio.binary_data
    else:
        request_data.source_audio.format = "wav"
        request_data.source_audio.rate = 16000
        request_data.source_audio.bits = 16
        request_data.source_audio.channel = 1

    if request.target_audio:
        request_data.target_audio.format = request.target_audio.format or "pcm"
        request_data.target_audio.rate = request.target_audio.rate or 24000
        if request.target_audio.bits is not None:
            request_data.target_audio.bits = request.target_audio.bits
        if request.target_audio.channel is not None:
            request_data.target_audio.channel = request.target_audio.channel
    else:
        request_data.target_audio.format = "pcm_s16le"
        request_data.target_audio.rate = 24000

    # request_data.request.mode = "s2s"
    # request_data.request.source_language = "zh"
    # request_data.request.target_language = "en"
    request_data.request.mode = request.mode or "s2s"
    request_data.request.source_language = request.source_language or "zh"
    request_data.request.target_language = request.target_language or "en"

    # 关键日志：确认你真正发出去的是啥
    if request.event == "Type_StartSession":
        logging.info(
            "SEND StartSession | "
            f"src=({request_data.source_audio.format}, {request_data.source_audio.rate}, "
            f"{request_data.source_audio.bits}, {request_data.source_audio.channel}) | "
            f"tgt=({request_data.target_audio.format}, {request_data.target_audio.rate}, "
            f"{request_data.target_audio.bits}, {request_data.target_audio.channel}) | "
            f"mode={request_data.request.mode} | "
            f"{request_data.request.source_language}->{request_data.request.target_language}"
        )

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
        target_audio=Audio(format="pcm_s16le", rate=24000),
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
    async def send_audio_chunks():
        try:
            for i, chunk in enumerate(audio_chunks):
                # logging.info(f"Sending chunk: {len(chunk)}")
                # chunk_request = TranslateRequestData(
                #     session_id=session_id,
                #     event="Type_TaskRequest",
                #     source_audio=Audio(binary_data=chunk)
                # )
                chunk_request = TranslateRequestData(
                    session_id=session_id,
                    event="Type_TaskRequest",
                    source_audio=Audio(
                        format="wav",
                        rate=16000,
                        bits=16,
                        channel=1,
                        binary_data=chunk
                    ),
                    target_audio=Audio(format="pcm_s16le", rate=24000),
                    mode="s2s",
                    source_language="zh",
                    target_language="en"
                )
                await send_request(conn, chunk_request)
                await asyncio.sleep(0.1)  # 100ms delay
            # Send finish session
            # finish_request = TranslateRequestData(
            #     session_id=session_id,
            #     event="Type_FinishSession",
            #     source_audio = Audio()
            # )
            finish_request = TranslateRequestData(
                session_id=session_id,
                event="Type_FinishSession",
                source_audio=Audio(format="wav", rate=16000, bits=16, channel=1),
                target_audio=Audio(format="pcm_s16le", rate=24000),
                mode="s2s",
                source_language="zh",
                target_language="en"
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
    first_audio_logged = False

    try:
        while True:
            #logging.info("Waiting for message...")
            resp = await receive_message(conn) ## ✌

            # logging.info(
            #     f"Receive message (event={resp.event}, session_id={resp.session_id}): "
            #     f"seq: {resp.sequence}, text:{resp.text}, audio data length:{len(resp.data)}, spk_chg: {resp.spk_chg}"
            # )
            if resp.event == Type.SessionFailed or resp.event == Type.SessionCanceled:
                logging.error(f"Session failed, message: {resp.message} logid: {log_id} event: {resp.event} message: {resp.message}")
                raise Exception("Session faild")

            if resp.event == Type.SessionFinished:
                break
            if resp.event != Type.UsageResponse:  # Skip append UsageResponse message to recv_text
                if resp.data and not first_audio_logged:
                    first_audio_logged = True
                    sig = detect_audio_signature(resp.data)
                    logging.info(f"FIRST AUDIO signature={sig}, first16={resp.data[:16]}")
                    if sig == "ogg":
                        logging.warning("You requested PCM, but first packet looks like OGG.")
                    elif sig == "wav":
                        logging.warning("You requested PCM, but first packet looks like WAV.")
                    else:
                        logging.info("First packet looks like raw/unknown bytes, could be PCM.")
                
                if resp.data:
                    chunk_ms = len(resp.data) / (24000 * 1 * (16 // 8)) * 1000.0
                    logging.info(f"bytes={len(resp.data)}, duration_ms={chunk_ms:.2f}")
                
                recv_audio.extend(resp.data)  ## ✌
                recv_text.append(resp.text)
    except Exception as e:
        logging.error(f"Receive message error: {e}")
    finally:
        await sender_task  # Ensure sender completes
        await conn.close()

    # Save results
    if recv_audio:
        # os.makedirs(out_dir, exist_ok=True)
        # output_path = Path(out_dir) / f"translate_audio_{n:05}.opus"
        # try:
        #     with open(output_path, 'wb') as f:
        #         f.write(recv_audio)
        #     logging.info(f"Session finished, audio is saved as: {output_path}")
        #     logging.info(f"Session finished, text is: {' '.join(recv_text)}")
        #     # 再转一份 wav 便于直接播放（优先用 imageio_ffmpeg 自带的 ffmpeg，免装系统 ffmpeg）
        #     wav_path = output_path.with_suffix(".wav")
        #     try:
        #         try:
        #             import imageio_ffmpeg
        #             _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        #         except Exception:
        #             _ffmpeg = "ffmpeg"
        #         subprocess.run(
        #             [_ffmpeg, "-y", "-i", str(output_path), str(wav_path)],
        #             check=True,
        #             capture_output=True,
        #         )
        #         logging.info(f"Session finished, wav is saved as: {wav_path}")
        #     except (subprocess.CalledProcessError, FileNotFoundError) as e:
        #         logging.warning(f"opus -> wav failed. pip install imageio-ffmpeg or install ffmpeg: {e}")
        # except Exception as e:
        #     logging.error(f"Save audio file: {e}")
        os.makedirs(out_dir, exist_ok=True)
        sig = detect_audio_signature(recv_audio)
        logging.info(f"FINAL AUDIO signature={sig}, total_bytes={len(recv_audio)}")

        # 1) 永远先把原始返回落盘，方便排查
        raw_path = Path(out_dir) / f"translate_audio_{n:05}.bin"
        with open(raw_path, 'wb') as f:
            f.write(recv_audio)
        logging.info(f"Raw bytes saved as: {raw_path}")

        # 2) 如果看起来像 OGG，就按 OGG 保存
        if sig == "ogg":
            opus_path = Path(out_dir) / f"translate_audio_{n:05}.opus"
            with open(opus_path, 'wb') as f:
                f.write(recv_audio)
            logging.warning(f"Requested PCM but got OGG-like data. Saved as: {opus_path}")

            wav_path = opus_path.with_suffix(".wav")
            try:
                try:
                    import imageio_ffmpeg
                    _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    _ffmpeg = "ffmpeg"
                subprocess.run(
                    [_ffmpeg, "-y", "-i", str(opus_path), str(wav_path)],
                    check=True,
                    capture_output=True,
                )
                logging.info(f"Converted wav is saved as: {wav_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logging.warning(f"opus -> wav failed: {e}")

        # 3) 如果看起来像 WAV，就按 WAV 保存
        elif sig == "wav":
            wav_path = Path(out_dir) / f"translate_audio_{n:05}.wav"
            with open(wav_path, 'wb') as f:
                f.write(recv_audio)
            logging.warning(f"Requested PCM but got WAV-like data. Saved as: {wav_path}")

        # 4) 否则按 PCM 处理，并给两份 wav 猜测版
        else:
            pcm_path = Path(out_dir) / f"translate_audio_{n:05}.pcm"
            with open(pcm_path, 'wb') as f:
                f.write(recv_audio)
            logging.info(f"PCM/raw bytes saved as: {pcm_path}")

            # 假设 24k / mono / int16
            wav16_path = Path(out_dir) / f"translate_audio_{n:05}_24k_s16.wav"
            try:
                save_pcm_as_wav(
                    recv_audio,
                    wav16_path,
                    sample_rate=24000,
                    sample_width=2,
                    channels=1,
                )
                logging.info(f"Wrapped PCM as WAV (16-bit guess): {wav16_path}")
            except Exception as e:
                logging.warning(f"Wrap PCM -> WAV(16-bit) failed: {e}")

            # 若长度可被4整除，再给一份 32-bit 猜测版

        logging.info(f"Session finished, text is: {' '.join(recv_text)}")
    else:
        logging.error("Session finished, no audio data is received.")

'''
pip install imageio-ffmpeg
python ast_python/ast_demo_pcm.py
'''

# Example usage
async def main():
    conf = Config(ws_url="wss://openspeech.bytedance.com/api/v4/ast/v2/translate",
                   app_key="2476390117",
                   access_key="J0QUKxRJb9j32MYmWOgoQ7d-n_jLI5Uk",
                  resource_id="volc.service_type.10053")
    start = time.time()
    task = asyncio.create_task(translate_v4(conf, "ast_python/test_audio.wav", 1, "ast_python/output"))
    
    await  task
    end = time.time()
    logging.info(f"Total time: {end - start:.6f} 秒")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())