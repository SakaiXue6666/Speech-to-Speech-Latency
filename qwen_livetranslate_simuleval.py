"""
将 Qwen 实时语音翻译 API 适配为 SimulEval 的 SpeechToSpeechAgent。

整体架构:
=========

  SimulEval (主线程)                  QwenRealtimeBackend (后台线程)
  ┌──────────────────┐                ┌────────────────────────────┐
  │                  │  push_audio    │                            │
  │  policy() ──────►├───────────────►│  input_audio_queue         │
  │                  │                │       │                    │
  │                  │                │       ▼                    │
  │                  │                │  _sender_loop ──► WebSocket│
  │                  │                │                    (Qwen)  │
  │                  │  try_pop_audio │                    │       │
  │  policy() ◄──────├◄───────────────│  output_audio_queue│       │
  │                  │                │       ▲            │       │
  │                  │                │  _receiver_loop ◄──┘       │
  └──────────────────┘                └────────────────────────────┘

核心难点:
  - SimulEval 的 policy() 是同步函数，但 Qwen API 是异步 WebSocket。
    因此用一个后台线程 (QwenRealtimeBackend) 运行 asyncio 事件循环，
    通过线程安全队列与 policy() 通信。
  - SimulEval 会以极快的速度把源音频全部喂完，但 Qwen 是实时 API，
    需要按真实速率接收音频。因此在 policy() 中加入"实时节拍控制"：
    每推送 N 秒源音频后，等待 N 秒真实时间，期间收集翻译结果。

运行命令:
  conda run --no-capture-output -n s2s_latency simuleval \\
      --agent qwen_livetranslate_simuleval.py \\
      --source data/source_list.txt \\
      --target data/target_list.txt \\
      --source-type speech --target-type speech \\
      --output data/output \\
      --target-language zh --print-text \\
      --source-segment-size 1000
"""

import asyncio
import base64
import json
import os
import queue
import threading
import time
import traceback
from typing import Optional

import numpy as np
import websockets
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.data.segments import SpeechSegment
from simuleval.utils import entrypoint


'''
conda run --no-capture-output -n s2s_latency simuleval --agent qwen_livetranslate_simuleval.py --source data/source_list.txt --target data/target_list.txt --source-type speech --target-type speech --output data/output --target-language zh --print-text --source-segment-size 1000
'''

os.environ["DASHSCOPE_API_KEY"] = "sk-34274543fd8e4e8e863a96b1293d1f58"


# =============================================================================
#  第一部分: QwenRealtimeBackend — 后台线程，负责与 Qwen WebSocket API 通信
# =============================================================================

class QwenRealtimeBackend:
    """
    后台线程适配器：在独立线程中运行 asyncio 事件循环和 WebSocket 连接。

    提供三个线程安全队列供 SimulEval Agent（主线程）交互:
      - input_audio_queue:  主线程 → 后台线程（待发送的源音频 PCM16 字节）
      - output_audio_queue: 后台线程 → 主线程（已翻译的音频采样 float32 列表）
      - output_text_queue:  后台线程 → 主线程（翻译文本，调试用）

    生命周期: start() → push_audio_bytes() ... → mark_input_finished() → stop()
    """

    def __init__(
        self,
        api_key: str,
        *,
        target_language: str = "en",     # 翻译目标语言
        voice: str = "Cherry",           # TTS 合成声音
        audio_enabled: bool = True,      # 是否同时输出音频（False 则仅文本）
        input_rate: int = 16000,         # 输入音频采样率（Qwen 要求 16kHz）
        output_rate: int = 24000,        # 输出音频采样率（Qwen 返回 24kHz）
        model_name: str = "qwen3-livetranslate-flash-realtime",
        final_wait_seconds: float = 2.0, # 输入结束后等待服务端生成剩余输出的秒数
    ):
        if not api_key:
            raise ValueError("API key cannot be empty.")

        self.api_key = api_key
        self.target_language = target_language
        self.voice = voice
        self.audio_enabled = audio_enabled
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.final_wait_seconds = final_wait_seconds
        self.api_url = (
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
            f"?model={model_name}"
        )

        # 线程同步事件
        self._thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()    # asyncio 循环已启动
        self._connected = threading.Event()     # WebSocket 已连接
        self._stop_requested = threading.Event() # 请求停止后台线程
        self._reset_queues_and_flags()

    def _reset_queues_and_flags(self):
        """重置所有队列和状态标志（每次 start() 前调用）。"""
        self.input_audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self.output_audio_queue: "queue.Queue[tuple[list[float], int]]" = queue.Queue()
        self.output_text_queue: "queue.Queue[str]" = queue.Queue()
        self._input_finished = False   # 源音频是否已全部推入
        self.server_done = False       # 服务端是否已完成所有响应
        self.error: Optional[BaseException] = None
        self.error_traceback: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self):
        """启动后台线程，建立 WebSocket 连接。"""
        if self._thread and self._thread.is_alive():
            return
        self._reset_queues_and_flags()
        self._loop_ready.clear()
        self._connected.clear()
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        # 等待后台线程的 asyncio 循环就绪
        self._loop_ready.wait(timeout=10)

    def stop(self, timeout: float = 5.0):
        """请求后台线程停止并等待其退出。"""
        self._stop_requested.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def push_audio_bytes(self, pcm16_bytes: bytes):
        """主线程调用：将一段 PCM16 音频字节推入发送队列。"""
        if pcm16_bytes:
            self.input_audio_queue.put(pcm16_bytes)

    def mark_input_finished(self):
        """主线程调用：标记源音频已全部推入，无更多输入。"""
        self._input_finished = True

    def try_pop_audio(self) -> Optional[tuple[list[float], int]]:
        """主线程调用：非阻塞地取出一个翻译音频块。返回 (采样列表, 采样率) 或 None。"""
        try:
            return self.output_audio_queue.get_nowait()
        except queue.Empty:
            return None

    def try_pop_text(self) -> Optional[str]:
        """主线程调用：非阻塞地取出一段翻译文本。"""
        try:
            return self.output_text_queue.get_nowait()
        except queue.Empty:
            return None

    def has_pending_output(self) -> bool:
        """主线程调用：检查输出音频队列是否还有数据。"""
        return not self.output_audio_queue.empty()

    # ---- 以下方法在后台线程中运行 ----

    def _thread_main(self):
        """后台线程入口：运行 asyncio 事件循环。"""
        try:
            asyncio.run(self._run())
        except BaseException as e:
            self.error = e
            self.error_traceback = traceback.format_exc()
            self.server_done = True

    async def _run(self):
        """
        后台线程的主协程：
        1. 连接 WebSocket
        2. 发送会话配置
        3. 并发运行 sender（发送音频）和 receiver（接收翻译结果）
        4. 任意一方出错或完成后清理退出
        """
        self._loop_ready.set()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        ws = None
        try:
            ws = await websockets.connect(self.api_url, additional_headers=headers)
            self._connected.set()
            await self._configure_session(ws)

            # 并发运行发送和接收两个协程
            sender_task = asyncio.create_task(self._sender_loop(ws))
            receiver_task = asyncio.create_task(self._receiver_loop(ws))

            # 等待任一任务完成或异常
            done, pending = await asyncio.wait(
                {sender_task, receiver_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except websockets.exceptions.ConnectionClosed:
            pass  # 正常关闭
        except BaseException as e:
            self.error = e
            self.error_traceback = traceback.format_exc()
        finally:
            self._connected.clear()
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            self.server_done = True

    async def _configure_session(self, ws):
        """发送会话配置：目标语言、TTS 声音、音频格式等。"""
        config = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                # ["text", "audio"] 同时返回文本和音频；["text"] 仅返回文本
                "modalities": ["text", "audio"] if self.audio_enabled else ["text"],
                **({"voice": self.voice} if self.audio_enabled and self.voice else {}),
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "translation": {"language": self.target_language},
            },
        }
        await ws.send(json.dumps(config))

    async def _sender_loop(self, ws):
        """
        发送协程：从 input_audio_queue 取出音频 → base64 编码 → 通过 WebSocket 发送。

        当所有输入音频发送完毕后，等待 final_wait_seconds 秒（让服务端生成剩余翻译），
        然后关闭 WebSocket。
        """
        input_drained_at: Optional[float] = None

        while not self._stop_requested.is_set():
            # 取出队列中所有待发送的音频块并逐个发送
            sent_any = False
            while True:
                try:
                    chunk = self.input_audio_queue.get_nowait()
                except queue.Empty:
                    break
                event = {
                    "event_id": f"event_{int(time.time() * 1000)}",
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }
                await ws.send(json.dumps(event))
                sent_any = True

            # 如果刚发送了新数据，重置等待计时器
            if sent_any:
                input_drained_at = None

            # 输入已结束且队列为空 → 开始倒计时关闭
            if self._input_finished and self.input_audio_queue.empty():
                if input_drained_at is None:
                    input_drained_at = time.monotonic()
                elif (time.monotonic() - input_drained_at) >= self.final_wait_seconds:
                    await ws.close()
                    return

            await asyncio.sleep(0.01)

        try:
            await ws.close()
        except Exception:
            pass

    async def _receiver_loop(self, ws):
        """
        接收协程：监听 WebSocket 消息，根据事件类型分发处理：
          - response.audio.delta:    翻译音频增量 → 解码为 float32 采样 → 放入输出队列
          - response.audio_transcript.done: 翻译文本 → 放入文本队列
          - response.text.done:      翻译文本（纯文本模式）
          - response.done:           一轮响应结束
        """
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "response.audio.delta" and self.audio_enabled:
                # 收到一小段翻译音频（base64 编码的 PCM16 字节）
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    pcm_bytes = base64.b64decode(audio_b64)
                    # PCM16 → float32（归一化到 [-1.0, 1.0]），供 SimulEval 写 WAV
                    samples = (
                        np.frombuffer(pcm_bytes, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    ).tolist()
                    if samples:
                        self.output_audio_queue.put((samples, self.output_rate))

            elif event_type == "response.audio_transcript.done":
                text = event.get("transcript", "")
                if text:
                    self.output_text_queue.put(text)

            elif event_type == "response.text.done":
                text = event.get("text", "")
                if text:
                    self.output_text_queue.put(text)

            elif event_type == "response.done":
                self.server_done = True


# =============================================================================
#  第二部分: QwenLiveTranslateSimulEvalAgent — SimulEval Agent 主体
# =============================================================================

@entrypoint
class QwenLiveTranslateSimulEvalAgent(SpeechToSpeechAgent):
    """
    SimulEval Agent：将 Qwen 实时语音翻译封装为 SimulEval 可调用的 Agent。

    SimulEval 框架的工作流程:
      1. SimulEval 加载源音频，按 --source-segment-size 分成小块
      2. 每次调用 policy()，Agent 决定：
         - ReadAction()  → "我需要更多源音频"（SimulEval 再喂一块）
         - WriteAction(SpeechSegment) → "这是翻译结果"（SimulEval 记录延迟）
      3. 循环直到 Agent 返回 finished=True 的 WriteAction
      4. SimulEval 计算延迟指标（LAAL、AL、DAL 等）

    每个 WriteAction 的延迟 = 当时已消费的源音频总量（毫秒）。
    因此 ReadAction 越多 → 消费越多源音频 → 延迟越大。
    """

    def __init__(self, args):
        # 注意：必须在 super().__init__() 之前初始化这些属性，
        # 因为父类构造函数会调用 self.reset()
        self.backend: Optional[QwenRealtimeBackend] = None
        self._reset_instance_state()

        super().__init__(args)

        # 获取 API Key（优先用命令行参数，其次用环境变量）
        api_key = getattr(args, "api_key", None) or os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Set --api-key or env var {args.api_key_env}."
            )

        self._api_key = api_key
        self._args = args

    def _reset_instance_state(self):
        """重置实例级别的状态（每个测试样本之间调用）。"""
        self.sent_source_samples = 0   # 已推送到 backend 的源音频采样数
        self.input_finish_sent = False  # 是否已通知 backend 输入结束
        self.finished_emitted = False   # 是否已发出 finished=True 的 WriteAction

    def _create_backend(self) -> QwenRealtimeBackend:
        """根据命令行参数创建一个新的 backend 实例。"""
        return QwenRealtimeBackend(
            api_key=self._api_key,
            target_language=self._args.target_language,
            voice=self._args.voice,
            audio_enabled=(not self._args.text_only),
            input_rate=self._args.qwen_input_rate,
            output_rate=self._args.qwen_output_rate,
            final_wait_seconds=self._args.final_wait_seconds,
        )

    @staticmethod
    def add_args(parser):
        """注册命令行参数（SimulEval 框架会自动调用）。"""
        parser.add_argument("--api-key", type=str, default=None,
                            help="Dashscope API Key（也可通过环境变量设置）")
        parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY",
                            help="API Key 的环境变量名")
        parser.add_argument("--target-language", type=str, default="en",
                            help="翻译目标语言（zh/en/ja/ko/...）")
        parser.add_argument("--voice", type=str, default="Cherry",
                            help="TTS 合成声音（Cherry/Nofish/...）")
        parser.add_argument("--text-only", action="store_true",
                            help="仅输出翻译文本，不生成语音")
        parser.add_argument("--print-text", action="store_true",
                            help="在终端打印翻译文本")
        parser.add_argument("--qwen-input-rate", type=int, default=16000,
                            help="Qwen API 要求的输入采样率")
        parser.add_argument("--qwen-output-rate", type=int, default=24000,
                            help="Qwen API 输出的采样率")
        parser.add_argument("--final-wait-seconds", type=float, default=10.0,
                            help="源音频结束后等待 API 生成剩余翻译的秒数")

    def reset(self):
        """SimulEval 在处理不同测试样本之间调用：停止旧 backend，重置状态。"""
        super().reset()
        if self.backend is not None:
            self.backend.stop()
            self.backend = None
        self._reset_instance_state()

    # ---- 音频输出辅助方法 ----

    def _drain_all_audio(self) -> list:
        """一次性取出输出队列中所有音频块，合并为一个采样列表。"""
        collected = []
        while True:
            out = self.backend.try_pop_audio()
            if out is None:
                break
            collected.extend(out[0])
        return collected

    def _make_write_action(self, samples: list, states: AgentStates):
        """用累积的音频采样构造一个 WriteAction 返回给 SimulEval。"""
        finished = self._should_finish_after_output(states)
        if finished:
            self.finished_emitted = True
        return WriteAction(
            SpeechSegment(
                content=samples,
                sample_rate=self.backend.output_rate,
                finished=finished,
            ),
            finished=finished,
        )

    # ---- 核心方法: policy() ----

    def policy(self, states: Optional[AgentStates] = None):
        """
        SimulEval 反复调用此方法来驱动翻译流程。

        整体流程:
          1. 懒启动 backend（首次调用时创建 WebSocket 连接）
          2. 把 states.source 中新增的源音频转为 PCM16 → 推入 backend
          3. 实时节拍控制：等待与新推送音频等长的真实时间，
             期间持续收集翻译输出，攒够后一次性返回 WriteAction
          4. 如果源音频未结束 → ReadAction（请求更多输入）
          5. 如果源音频已结束 → 阻塞轮询，直到 backend 生成完所有翻译
        """
        if states is None:
            states = self.states

        # 第 1 步：懒启动 backend（reset() 后首次调用会重新创建）
        if self.backend is None:
            self.backend = self._create_backend()
            self.backend.start()

        self._raise_if_backend_failed()
        self._drain_text_messages()

        # 第 2 步：将 states.source 中新增的采样推送到 backend
        # （states.source 是累积的，每次 ReadAction 后会追加新数据）
        new_samples_count = 0
        if states.source_sample_rate and len(states.source) > self.sent_source_samples:
            new_source = states.source[self.sent_source_samples:]
            pcm16_bytes = self._source_to_pcm16_bytes(
                new_source,
                src_rate=states.source_sample_rate,
                dst_rate=self.backend.input_rate,
            )
            if pcm16_bytes:
                self.backend.push_audio_bytes(pcm16_bytes)
            new_samples_count = len(states.source) - self.sent_source_samples
            self.sent_source_samples = len(states.source)

        # 通知 backend 输入已结束
        if states.source_finished and not self.input_finish_sent:
            self.backend.mark_input_finished()
            self.input_finish_sent = True

        # 第 3 步：实时节拍控制
        # 推送了 N 秒的源音频后，等待 N 秒真实时间。
        # 在等待期间每 50ms 检查一次翻译输出，把所有到达的音频累积起来。
        # 等待结束后，将累积的音频作为一个 WriteAction 返回。
        # 这样做的目的：
        #   a) 让 Qwen API 以接近实时的速率接收音频
        #   b) 减少 WriteAction 数量，避免 SimulEval 在输出 WAV 中插入过多静音
        if new_samples_count > 0 and states.source_sample_rate:
            wait_secs = new_samples_count / states.source_sample_rate
            deadline = time.monotonic() + wait_secs
            accumulated = []
            while time.monotonic() < deadline:
                self._raise_if_backend_failed()
                self._drain_text_messages()
                accumulated.extend(self._drain_all_audio())
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(0.05, remaining))

            # 等待结束后再取一次，确保不遗漏
            accumulated.extend(self._drain_all_audio())
            if accumulated:
                return self._make_write_action(accumulated, states)

        # 检查是否有遗留的输出（上一轮未取完的）
        leftover = self._drain_all_audio()
        if leftover:
            return self._make_write_action(leftover, states)

        # 第 4 步：源音频未结束 → 请求更多输入
        if not states.source_finished:
            return ReadAction()

        # 第 5 步：源音频已结束，阻塞等待 backend 生成剩余翻译
        # 每次最多等 1 秒，取出所有累积音频后返回 WriteAction
        while not self.finished_emitted:
            self._raise_if_backend_failed()
            self._drain_text_messages()

            accumulated = []
            poll_deadline = time.monotonic() + 1.0
            while time.monotonic() < poll_deadline:
                accumulated.extend(self._drain_all_audio())
                if (
                    self.backend.server_done
                    and not self.backend.has_pending_output()
                ):
                    break
                time.sleep(0.05)

            accumulated.extend(self._drain_all_audio())

            if accumulated:
                return self._make_write_action(accumulated, states)

            # backend 完全结束且无更多输出 → 发送空的终止段
            if (
                self.backend.server_done
                and not self.backend.has_pending_output()
            ):
                self.finished_emitted = True
                return WriteAction(
                    SpeechSegment(
                        content=[],
                        sample_rate=self.backend.output_rate,
                        finished=True,
                    ),
                    finished=True,
                )

        return ReadAction()

    # ---- 辅助方法 ----

    def _raise_if_backend_failed(self):
        """检查 backend 是否出错，有则抛出异常。"""
        if self.backend.error is not None:
            msg = f"Qwen realtime backend failed: {self.backend.error}"
            if self.backend.error_traceback:
                msg += f"\n{self.backend.error_traceback}"
            raise RuntimeError(msg)

    def _drain_text_messages(self):
        """如果启用了 --print-text，将翻译文本打印到终端。"""
        if not self._args.print_text:
            return
        while True:
            text = self.backend.try_pop_text()
            if text is None:
                break
            print(f"[qwen text] {text}", flush=True)

    def _should_finish_after_output(self, states: AgentStates) -> bool:
        """判断当前是否应该标记 finished=True（所有输入输出都已完成）。"""
        return (
            states.source_finished
            and self.backend.server_done
            and not self.backend.has_pending_output()
        )

    @staticmethod
    def _source_to_pcm16_bytes(source_chunk, src_rate: int, dst_rate: int) -> bytes:
        """
        将 SimulEval 提供的源音频采样转换为 PCM16 字节。

        处理步骤:
          1. 多通道 → 单通道（取平均值）
          2. 浮点数 → int16（归一化缩放）
          3. 如果采样率不匹配 → 线性插值重采样
        """
        if src_rate <= 0:
            return b""

        audio = np.asarray(source_chunk)
        if audio.size == 0:
            return b""

        # 多通道取平均变单通道
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # 浮点归一化为 int16
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        else:
            audio = np.clip(audio, -32768, 32767).astype(np.int16)

        # 采样率不匹配时进行重采样（线性插值）
        if src_rate != dst_rate and audio.size > 0:
            src_len = audio.shape[0]
            dst_len = int(src_len * dst_rate / src_rate)
            if dst_len <= 0:
                return b""
            x_src = np.linspace(0, src_len - 1, src_len)
            x_dst = np.linspace(0, src_len - 1, dst_len)
            audio = np.interp(x_dst, x_src, audio).astype(np.int16)

        return audio.tobytes()
