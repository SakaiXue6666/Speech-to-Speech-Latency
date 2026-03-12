import asyncio
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
import wave
from google.protobuf.json_format import MessageToDict
from websockets.legacy.exceptions import InvalidStatusCode

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
    start_time: int = None
    end_time: int = None


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
    request_data.target_audio.format = "pcm"
    request_data.target_audio.rate = 16000
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
        message=Response_data.response_meta.Message,
        start_time=Response_data.start_time,
        end_time=Response_data.end_time
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
        target_audio=Audio(format="ogg_opus", rate=24000),
        mode="s2s",
        source_language="en",
        target_language="zh"
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
                logging.info(f"Sending chunk: {len(chunk)}")
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

    # Receive responses
    recv_audio = bytearray()
    recv_text = []
    sample_rate = 16000
    bytes_per_sample = 2  # 16-bit mono
    audio_cursor = 0.0  # current end of placed audio (in seconds)
    wall_clock_start = time.time()

    try:
        while True:
            #logging.info("Waiting for message...")
            resp = await receive_message(conn)

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
            if resp.event == Type.TTSResponse:
                wall_time = time.time() - wall_clock_start
                chunk_duration = len(resp.data) / (sample_rate * bytes_per_sample)
                if wall_time > audio_cursor:
                    # Gap: insert silence from audio_cursor to wall_time
                    gap = wall_time - audio_cursor
                    silence_bytes = int(gap * sample_rate) * bytes_per_sample
                    recv_audio.extend(b'\x00' * silence_bytes)
                    audio_cursor = wall_time + chunk_duration
                else:
                    # No gap: place immediately after previous audio
                    audio_cursor += chunk_duration
                recv_audio.extend(resp.data)
            if resp.event == Type.TTSSentenceEnd:
                elapsed = time.time() - wall_clock_start
                audio_duration_s = len(recv_audio) / (sample_rate * bytes_per_sample)
                logging.warning(f"TTSSentenceEnd: wall_clock={elapsed:8.3f}s, audio_dur={audio_duration_s:8.3f}s, audio_cursor={audio_cursor:8.3f}s")
    except Exception as e:
        logging.error(f"Receive message error: {e}")
    finally:
        await sender_task  # Ensure sender completes
        await conn.close()

    # Save results
    if recv_audio:
        os.makedirs(out_dir, exist_ok=True)
        output_path = Path(out_dir) / f"translate_audio_{n:05}.wav"
        try:
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(16000)
                wf.writeframes(recv_audio)
            logging.info(f"Session finished, audio is saved as: {output_path}")
            logging.info(f"Session finished, text is: {' '.join(recv_text)}")
        except Exception as e:
            logging.error(f"Save audio file: {e}")
    else:
        logging.error("Session finished, no audio data is received.")


# Example usage
async def main():
    conf = Config(ws_url="wss://openspeech.bytedance.com/api/v4/ast/v2/translate",
                   app_key="2476390117",
                   access_key="J0QUKxRJb9j32MYmWOgoQ7d-n_jLI5Uk",
                  resource_id="volc.service_type.10053")
    # if len(sys.argv) < 2:
    #     print(f"Usage: python {sys.argv[0]} <audio_file>")
    #     sys.exit(1)
    # audio_path = sys.argv[1]
    audio_path = "ast_python/input.wav"   # 输入路径
    out_dir = "ast_python/output_inputWAV_siqi"              # 输出目录

    start = time.time()
    task = asyncio.create_task(translate_v4(conf, audio_path, 1, out_dir))
    await task
    end = time.time()
    logging.warning(f"Total time: {end - start:.6f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())