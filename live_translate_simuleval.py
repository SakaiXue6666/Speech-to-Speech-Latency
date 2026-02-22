from typing import Optional
import argparse
import asyncio
import base64
import contextlib
import json
import os
import threading
import time
import wave
from array import array
from pathlib import Path
from queue import Empty, Queue

import websockets
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import WriteAction, ReadAction


os.environ["DASHSCOPE_API_KEY"] = "sk-34274543fd8e4e8e863a96b1293d1f58"


API_URL_MAP = {
    "intl": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
    "cn": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
}


class QwenLiveTranslateModel:
    """
    Minimal qwen3 livetranslate wrapper in the same role as original TTSModel.
    - feed_audio_chunk(): send source audio
    - synthesize(): pull one translated chunk
    """

    def __init__(self, api_key: str, region: str, model: str, source_language: str, target_language: str, voice: str):
        if not api_key:
            raise ValueError("Missing API key. Set --api-key or DASHSCOPE_API_KEY")
        if region not in API_URL_MAP:
            raise ValueError(f"Unsupported region: {region}")

        self.api_key = api_key
        self.ws_url = f"{API_URL_MAP[region]}?model={model}"
        self.source_language = source_language
        self.target_language = target_language
        self.voice = voice

        self._loop = None
        self._thread = None
        self._connected = threading.Event()
        self._stopped = threading.Event()
        self._response_done = threading.Event()
        self._error_queue: Queue[str] = Queue(maxsize=1)
        self._send_queue: Queue[bytes | None] = Queue()
        self._recv_queue: Queue[bytes] = Queue()

        self._start()

    def _start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._connected.wait(timeout=0.2):
                self._raise_if_error()
                return
            self._raise_if_error()
        raise RuntimeError(f"Timeout connecting to qwen3-livetranslate ({self.ws_url})")

    @staticmethod
    def _float_to_pcm16_bytes(samples) -> bytes:
        pcm = array("h")
        for x in samples:
            if isinstance(x, float):
                v = int(max(-1.0, min(1.0, x)) * 32767)
            else:
                v = int(x)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            pcm.append(v)
        return pcm.tobytes()

    @staticmethod
    def _pcm16_bytes_to_float(pcm: bytes):
        a = array("h")
        a.frombytes(pcm)
        return [s / 32768.0 for s in a]

    def feed_audio_chunk(self, samples):
        self._raise_if_error()
        pcm = self._float_to_pcm16_bytes(samples)
        if pcm:
            self._send_queue.put(pcm)

    def finish_input(self):
        self._send_queue.put(None)

    def synthesize(self):
        """Return one translated audio chunk (samples, fs). If no chunk, return ([], 24000)."""
        self._raise_if_error()
        try:
            pcm = self._recv_queue.get_nowait()
        except Empty:
            return [], 24000
        return self._pcm16_bytes_to_float(pcm), 24000

    def is_done(self) -> bool:
        return self._response_done.is_set() and self._recv_queue.empty()

    def close(self):
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._send_queue.put(None)
        if self._loop and (not self._loop.is_closed()):
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread:
            self._thread.join(timeout=5)

    def _raise_if_error(self):
        try:
            msg = self._error_queue.get_nowait()
        except Empty:
            return
        raise RuntimeError(msg)

    def _push_error(self, msg: str):
        if not self._error_queue.full():
            self._error_queue.put(msg)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self._push_error(f"qwen realtime loop crashed: {e}")
            self._connected.set()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                with contextlib.suppress(Exception):
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _main(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with websockets.connect(
            self.ws_url,
            additional_headers=headers,
            # Avoid false disconnects on unstable links while server is processing long chunks.
            ping_interval=30,
            ping_timeout=120,
            close_timeout=10,
        ) as ws:
            await self._send_session_update(ws)
            self._connected.set()

            sender = asyncio.create_task(self._sender(ws))
            receiver = asyncio.create_task(self._receiver(ws))
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_EXCEPTION)

            for task in done:
                exc = task.exception()
                if exc:
                    self._push_error(str(exc))

            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _send_session_update(self, ws):
        config = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.voice,
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "input_audio_transcription": {
                    "model": "qwen3-asr-flash-realtime",
                    "language": self.source_language,
                },
                "translation": {"language": self.target_language},
            },
        }
        await ws.send(json.dumps(config, ensure_ascii=False))

    async def _sender(self, ws):
        while not self._stopped.is_set():
            pcm = await asyncio.get_running_loop().run_in_executor(None, self._send_queue.get)
            if pcm is None:
                return
            event = {
                "event_id": f"event_{int(time.time() * 1000)}",
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("utf-8"),
            }
            await ws.send(json.dumps(event))

    async def _receiver(self, ws):
        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "response.audio.delta":
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    self._recv_queue.put(base64.b64decode(audio_b64))

            elif event_type == "response.done":
                self._response_done.set()
                if self._send_queue.empty():
                    return

            elif event_type == "error":
                self._push_error(f"Server error: {json.dumps(event, ensure_ascii=False)}")
                return


@entrypoint
class QwenSpeechCounter(SpeechToSpeechAgent):
    """
    Minimal agent in original template style.
    - wait_seconds: initial read-only warm-up
    - after that: stream source audio to qwen and emit translated audio chunks
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_seconds = args.wait_seconds
        self.min_chunk_ms = args.min_chunk_ms

        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.qwen_model = QwenLiveTranslateModel(
            api_key=api_key,
            region=args.region,
            model=args.model,
            source_language=args.source_language,
            target_language=args.target_language,
            voice=args.voice,
        )

        self.sent_source_samples = 0
        self.sent_finished = False
        self.wrote_final = False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-seconds", default=1, type=int)
        parser.add_argument("--min-chunk-ms", default=200, type=int)
        parser.add_argument("--api-key", default="", type=str)
        parser.add_argument("--region", default="intl", choices=["intl", "cn"])
        parser.add_argument("--model", default="qwen3-livetranslate-flash-realtime")
        parser.add_argument("--source-language", default="en")
        parser.add_argument("--target-language", default="zh")
        parser.add_argument("--voice", default="Cherry")

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = round(len(states.source) / states.source_sample_rate)

        if not states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()

        total_samples = len(states.source)
        if total_samples < self.sent_source_samples:
            self.sent_source_samples = 0
            self.sent_finished = False
            self.wrote_final = False

        min_chunk_samples = max(1, int(states.source_sample_rate * self.min_chunk_ms / 1000))
        new_samples = total_samples - self.sent_source_samples

        if new_samples >= min_chunk_samples:
            chunk = states.source[self.sent_source_samples:total_samples]
            self.qwen_model.feed_audio_chunk(chunk)
            self.sent_source_samples = total_samples

        if states.source_finished and not self.sent_finished:
            if self.sent_source_samples < total_samples:
                chunk = states.source[self.sent_source_samples:total_samples]
                self.qwen_model.feed_audio_chunk(chunk)
                self.sent_source_samples = total_samples
            self.qwen_model.finish_input()
            self.sent_finished = True

        samples, fs = self.qwen_model.synthesize()
        if samples:
            is_final = states.source_finished and self.qwen_model.is_done()
            if is_final:
                self.wrote_final = True
            return WriteAction(
                SpeechSegment(content=samples, sample_rate=fs, finished=is_final),
                finished=is_final,
            )

        if states.source_finished and self.qwen_model.is_done() and not self.wrote_final:
            self.wrote_final = True
            return WriteAction(
                SpeechSegment(content=[], sample_rate=24000, finished=True),
                finished=True,
            )

        return ReadAction()

    def __del__(self):
        if hasattr(self, "qwen_model") and self.qwen_model is not None:
            self.qwen_model.close()


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        width = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    if channels != 1 or sample_rate != 16000 or width != 2:
        raise ValueError(
            "Input wav must be 16kHz mono 16-bit PCM. "
            f"Got channels={channels}, sample_rate={sample_rate}, sample_width={width}."
        )
    return pcm


def _write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _parse_main_args():
    p = argparse.ArgumentParser(description="Direct run for qwen3 live speech-to-speech translation")
    p.add_argument("--input-wav", default="data/2022.acl-long.268.wav")
    p.add_argument("--output-wav", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--region", default="intl", choices=["intl", "cn"])
    p.add_argument("--model", default="qwen3-livetranslate-flash-realtime")
    p.add_argument("--source-language", default="en")
    p.add_argument("--target-language", default="zh")
    p.add_argument("--voice", default="Cherry")
    p.add_argument("--chunk-ms", default=100, type=int)
    p.add_argument("--wait-timeout", default=90.0, type=float)
    return p.parse_args()


def _main_direct():
    args = _parse_main_args()
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set --api-key or DASHSCOPE_API_KEY")

    input_wav = Path(args.input_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"Input wav not found: {input_wav}")

    output_wav = Path(args.output_wav) if args.output_wav else input_wav.with_name(
        f"{input_wav.stem}.simuleval.translated.{args.target_language}.wav"
    )

    pcm = _read_wav_pcm16(input_wav)

    model = QwenLiveTranslateModel(
        api_key=api_key,
        region=args.region,
        model=args.model,
        source_language=args.source_language,
        target_language=args.target_language,
        voice=args.voice,
    )

    out_chunks = []
    chunk_ms = max(10, args.chunk_ms)
    chunk_bytes = int(16000 * chunk_ms / 1000) * 2
    deadline = time.time() + args.wait_timeout

    try:
        for i in range(0, len(pcm), chunk_bytes):
            part = pcm[i : i + chunk_bytes]
            model.feed_audio_chunk(model._pcm16_bytes_to_float(part))
            time.sleep(chunk_ms / 1000.0)

        model.finish_input()

        while time.time() < deadline:
            samples, _ = model.synthesize()
            if samples:
                out_chunks.append(model._float_to_pcm16_bytes(samples))
            elif model.is_done():
                break
            else:
                time.sleep(0.01)
    finally:
        model.close()

    if not out_chunks:
        raise RuntimeError("No translated audio returned. Try increasing --wait-timeout.")

    _write_wav_pcm16(output_wav, b"".join(out_chunks), sample_rate=24000)
    print(f"Saved translated speech: {output_wav}")


if __name__ == "__main__":
    _main_direct()
