import os
import time
import base64
import asyncio
import json
import websockets
import pyaudio
import queue
import threading
import traceback

import numpy as np
import soundfile as sf


# [INFO] 翻译文本完成。
# [INFO] 翻译文本: 我是你爸爸，我是你爸爸。你是谁？

# [INFO] 一轮响应完成。
# [INFO] Token 使用情况: {
#   "total_tokens": 147,
#   "input_tokens": 92,
#   "output_tokens": 55,
#   "input_tokens_details": {
#     "text_tokens": 30,
#     "audio_tokens": 62
#   },
#   "output_tokens_details": {
#     "text_tokens": 15,
#     "audio_tokens": 40
#   }
# }


# [INFO] 翻译文本完成。
# [INFO] 翻译文本: 你好，你叫什么名字？

# [INFO] 一轮响应完成。
# [INFO] Token 使用情况: {
#   "total_tokens": 838,
#   "input_tokens": 599,
#   "output_tokens": 239,
#   "input_tokens_details": {
#     "text_tokens": 276,
#     "audio_tokens": 323
#   },
#   "output_tokens_details": {
#     "text_tokens": 62,
#     "audio_tokens": 177
#   }
# }

os.environ["DASHSCOPE_API_KEY"] = "sk-34274543fd8e4e8e863a96b1293d1f58"

class LiveTranslateClient:
    def __init__(self, api_key: str, target_language: str = "en", voice: str | None = "Cherry", *, audio_enabled: bool = True, 
    manifest_src_path: str | None = None, save_tgt_wav_path: str | None = None, save_timeline_path: str | None = None):
        if not api_key:
            raise ValueError("API key cannot be empty.")
            
        self.api_key = api_key
        self.target_language = target_language
        self.audio_enabled = audio_enabled
        self.voice = voice if audio_enabled else "Cherry"
        self.ws = None
        self.api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-livetranslate-flash-realtime"
        
        # 音频输入配置 (来自麦克风)
        self.input_rate = 16000
        self.input_chunk = 3200  # 1600
        # self.input_format = pyaudio.paInt16
        # self.input_channels = 1
        
        # 音频输出配置 (用于播放)
        self.output_rate = 24000
        self.output_chunk = 2400
        self.output_format = pyaudio.paInt16
        self.output_channels = 1
        
        # 状态管理
        self.is_connected = False
        self.audio_player_thread = None
        self.audio_playback_queue = queue.Queue()  # [tgt] 音频播放队列
        self.pyaudio_instance = pyaudio.PyAudio()

        # ===========================================================\
        # ⭐ [tgt] 保存生成的翻译音频与时间线：每段 delta 的 (offset_sec, duration_sec, receive_timestamp)
        self.tgt_audio_chunks = []  # 收集所有 response.audio.delta 的 PCM，最后拼成 wav
        self.tgt_timeline = []      # 每段在 tgt 音频时间轴上的 offset、时长、接收物理时间
        self.manifest_src_path = manifest_src_path           # 源 wav 路径，用于写入 manifest
        self.save_tgt_wav_path = save_tgt_wav_path   # 若设置，close() 时将 tgt 音频写入该路径
        self.save_timeline_path = save_timeline_path # 若设置，close() 时将时间线写入该 json 路径
        # ⭐ [src] 以「发送」为参考：第一次发送 src 音频的墙钟时间（用于 tgt 物理时间 = first_send_timestamp + 偏移）
        self.first_send_timestamp = None

        self._chunk_id = 0
        self._timeline_by_id = {}  # chunk_id -> entry
        self._timeline_lock = threading.Lock()
        # ===========================================================/

    # ------------------------------------------------------------
    # 连接
    # Client -> Server，用到 configure_session 的 session.update
    async def connect(self):
        """建立到翻译服务的 WebSocket 连接。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            # 等待连接
            self.ws = await websockets.connect(self.api_url, additional_headers=headers)
            # 连接成功
            self.is_connected = True
            # 等待配置会话
            print(f"成功连接到服务端: {self.api_url}")
            # 配置会话成功
            await self.configure_session()
        except Exception as e:
            # 失败
            print(f"连接失败: {e}")
            self.is_connected = False
            raise

    # Client -> Server: session.update
    async def configure_session(self):
        """配置翻译会话，设置目标语言、声音等。"""
        config = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                # 'modalities': ["text", "audio"]: 同时返回翻译文本和合成音频（推荐）；["text"]: 仅返回翻译文本。
                "modalities": ["text", "audio"] if self.audio_enabled else ["text"],
                **({"voice": self.voice} if self.audio_enabled and self.voice else {}),
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                # 'input_audio_transcription' 配置源语言识别。
                # 设置 'model' 为 'qwen3-asr-flash-realtime' 可同时输出源语言识别结果。
                # "input_audio_transcription": {
                #     "model": "qwen3-asr-flash-realtime",
                #     "language": "zh"  # 源语言，默认 'en'
                # },
                "translation": {
                    "language": self.target_language
                }
            }
        }
        print(f"发送会话配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        await self.ws.send(json.dumps(config))

    # ------------------------------------------------------------
    # Client -> Server: input_audio_buffer.append
    async def send_audio_chunk(self, audio_data: bytes):
        """将音频数据块编码并发送到服务端。"""
        if not self.is_connected:
            return

        # ⭐ 以「发送」为参考：记录第一次发送 src 的墙钟时间（只记一次）
        # ✅ 把基准点绑定到“第一次真正 append 发送”
        if self.first_send_timestamp is None:
            self.first_send_timestamp = time.time()
            
        event = {
            "event_id": f"event_{int(time.time() * 1000)}",  # 事件ID
            "type": "input_audio_buffer.append",  # 事件类型：把后面带的音频追加到输入缓冲区
            "audio": base64.b64encode(audio_data).decode()  # 音频数据
        }
        await self.ws.send(json.dumps(event))  # 发送事件

    # Client -> Server: input_image_buffer.append
    async def send_image_frame(self, image_bytes: bytes, *, event_id: str | None = None):
        #将图像数据发送到服务端
        if not self.is_connected:
            return

        if not image_bytes:
            raise ValueError("image_bytes 不能为空")

        # 编码为 Base64
        image_b64 = base64.b64encode(image_bytes).decode()

        event = {
            "event_id": event_id or f"event_{int(time.time() * 1000)}",
            "type": "input_image_buffer.append",
            "image": image_b64,
        }

        await self.ws.send(json.dumps(event))

    # ------------------------------------------------------------
    # 播放
    def _audio_player_task(self):
        """在子线程里循环从 audio_playback_queue 取数据并写入声卡播放"""
        stream = self.pyaudio_instance.open(
            format=self.output_format,
            channels=self.output_channels,
            rate=self.output_rate,
            output=True,
            frames_per_buffer=self.output_chunk,
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
        prev_heard_end = 0.0  # 上一块"听众听完"的墙钟时刻
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/
        try:
            while self.is_connected or not self.audio_playback_queue.empty():
                try:
                    item = self.audio_playback_queue.get(timeout=0.1)
                    if item is None:
                        break
                    chunk_id, audio_chunk = item
                    t_before = time.time()
                    try:
                        stream.write(bytes(audio_chunk))
                    except OSError:
                        break
                    t_after = time.time()
                    with self._timeline_lock:
                        ent = self._timeline_by_id.get(chunk_id)
                        if ent is not None:
                            # ent["play_timestamp_before"] = float(t_before)
                            # ent["play_timestamp_after"] = float(t_after)
                            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
                            ent["write_before"] = float(t_before)
                            ent["write_after"] = float(t_after)

                            dur = ent["duration_sec"]
                            # 👈 heard_start = 听众真正开始听到这块的时刻
                            # 连续播放时：上一块听完后紧接着听这块
                            # gap 后：缓冲区空了，t_before 就是真实开始时刻
                            heard_start = max(t_before, prev_heard_end)
                            heard_end = heard_start + dur
                            prev_heard_end = heard_end

                            ent["heard_start"] = float(heard_start)
                            ent["heard_end"] = float(heard_end)
                            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/
    
                            if self.first_send_timestamp is not None:
                                # output_time_sec = float(t_before - self.first_send_timestamp)
                                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
                                output_time_sec = float(heard_start - self.first_send_timestamp)
                                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/
                                ent["output_time_sec"] = output_time_sec
                                print(f"👉 output_time_sec: {output_time_sec}")
                    self.audio_playback_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            try:
                stream.stop_stream()
            except OSError:
                pass
            try:
                stream.close()
            except OSError:
                pass

    def start_audio_player(self):
        """启动音频播放线程（仅当启用音频输出时）。"""
        if not self.audio_enabled:
            return
        if self.audio_player_thread is None or not self.audio_player_thread.is_alive():
            self.audio_player_thread = threading.Thread(target=self._audio_player_task, daemon=True)
            self.audio_player_thread.start()

    # ------------------------------------------------------------
    # Server -> Client
    # response.audio.delta: [tgt] 翻译结果语音的一小段（流式）
    # response.done: 本轮响应结束
    # conversation.item.input_audio_transcription.text： [src] 识别进行中的中间结果（开asr时）
    # conversation.item.input_audio_transcription.completed： [src] 这一段音频的最终识别结果（开asr时）
    # response.audio_transcript.done: [tgt] 本段翻译的最终文本（开音频时）
    # response.text.done: [tgt] 本段翻译的最终文本（仅文本时）
    async def handle_server_messages(self, on_text_received):
        """循环处理来自服务端的消息。"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                # response.audio.delta: [tgt] 翻译结果语音的一小段（流式）
                if event_type == "response.audio.delta" and self.audio_enabled:
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)

                        # ===========================================================\
                        # ⭐ 记录接收物理时间与在 tgt 音频时间轴上的 offset，供后续保存与 latency 分析
                        # receive_ts = time.time()
                        # duration_sec = len(audio_data) / (2 * self.output_rate)  # 16bit mono，每样本 2 字节
                        # offset_sec = (self.tgt_timeline[-1][0] + self.tgt_timeline[-1][1]) if self.tgt_timeline else 0  # 当前块在最终 tgt wav 中的起始时间（秒）
                        # # 注意：相邻 delta 的 receive_timestamp 间隔可能小于 duration_sec——服务端流式推送多块时，
                        # # 多块可在很短时间内连续到达，故「收到时间间隔」与「每块内容时长」无必然关系。

                        # # offset_src：这块在「最终 tgt 音频」里从第几秒开始
                        # # duration_src：这块在「最终 tgt 音频」音频里占多少秒
                        # # receive_ts：收到这块时的物理时间
                        # # 若改成“考虑时间 gap”（例如按收到间隔插静音，让 tgt 时间轴 = 真实经过时间），并定义 offset 为“相对会话起点 t0 的秒”，则可以做到 offset = receive_ts - t0
                        # self.tgt_timeline.append((offset_sec, duration_sec, receive_ts))

                        # self.tgt_audio_chunks.append(audio_data)  # 把 一小段翻译结果语音 放入 tgt 音频
                        # # ===========================================================/

                        # # 收到一块就 put 进队列，播放线程按顺序 get 后连续 stream.write()，中间不插静音
                        # # 生成的 tgt 音频时间轴 = 把所有 delta 的 PCM 按顺序接在一起，没有显式“空闲时间”
                        # # 每块在“最终 tgt 音频”里的位置，只由前面所有块的 PCM 总时长决定，和“收到的时间间隔”无关
                        # self.audio_playback_queue.put(audio_data)  # 把 一小段翻译结果语音 放入音频播放队列

                        receive_ts = time.time()
                        # duration_sec = len(audio_data) / (2 * self.output_rate)
                        duration_sec = len(audio_data) / (2 * self.output_rate * self.output_channels)
                        offset_sec = (self.tgt_timeline[-1]["offset_sec"] + self.tgt_timeline[-1]["duration_sec"]) if self.tgt_timeline else 0.0

                        chunk_id = self._chunk_id
                        self._chunk_id += 1

                        entry = {
                            "chunk_id": chunk_id,
                            "offset_sec": float(offset_sec),
                            "duration_sec": float(duration_sec),
                            "receive_timestamp": float(receive_ts),
                            # 播放时再补
                            # "play_timestamp_before": None,
                            # "play_timestamp_after": None,
                            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
                            "write_before": None,
                            "write_after": None,
                            "heard_start": None,
                            "heard_end": None,
                            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/
                            "output_time_sec": None,
                        }

                        with self._timeline_lock:
                            self.tgt_timeline.append(entry)
                            self._timeline_by_id[chunk_id] = entry

                        self.tgt_audio_chunks.append(audio_data)
                        self.audio_playback_queue.put((chunk_id, audio_data))

                # response.done: 本轮响应结束
                # Server 把这一段输入（你之前发的音频/文本）处理完了：识别、翻译、以及若开启了音频则 TTS 也播完了
                # elif event_type == "response.done":
                #     print("\n[INFO] 一轮响应完成。")
                #     usage = event.get("response", {}).get("usage", {})
                #     if usage:
                #         print(f"[INFO] Token 使用情况: {json.dumps(usage, indent=2, ensure_ascii=False)}")

                # 处理源语言识别结果（需启用 input_audio_transcription.model）
                # # conversation.item.input_audio_transcription.text： [src] 识别进行中的中间结果（开asr时）
                # # 流式 ASR，说一点就返回一点，stash 可能是“当前已识别出的、还可能被修正”的文本，会随着后续音频更新
                # elif event_type == "conversation.item.input_audio_transcription.text":
                #     stash = event.get("stash", "")  # 待确认的识别文本
                #     print(f"[识别中] {stash}")
                #
                # # conversation.item.input_audio_transcription.completed： [src] 这一段音频的最终识别结果（开asr时）
                # # 这一段输入已经处理完，transcript 是这一段对应的最终、完整源语言文本，不再变
                # elif event_type == "conversation.item.input_audio_transcription.completed":
                #     transcript = event.get("transcript", "")  # 完整识别结果
                #     print(f"[源语言] {transcript}")

                # response.audio_transcript.done: [tgt] 本段翻译的最终文本（开音频时）
                elif event_type == "response.audio_transcript.done":
                    print("\n[INFO] 翻译文本完成。")
                    text = event.get("transcript", "")
                    if text:
                        print(f"[INFO] 翻译文本: {text}")

                # response.text.done: [tgt] 本段翻译的最终文本（仅文本时）
                elif event_type == "response.text.done":
                    print("\n[INFO] 翻译文本完成。")
                    text = event.get("text", "")
                    if text:
                        print(f"[INFO] 翻译文本: {text}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[WARNING] 连接已关闭: {e}")
            self.is_connected = False
        except Exception as e:
            print(f"[ERROR] 消息处理时发生未知错误: {e}")
            traceback.print_exc()
            self.is_connected = False

    # ------------------------------------------------------------
    # wav
    # Client -> Server，用到 send_audio_chunk 的 input_audio_buffer.append
    async def start_wav_streaming(self, audio_path: str = "data/2022.acl-long.268.wav"):
        """从 wav 文件读取音频并按流式方式传输到服务端。"""
        # ===========================================================\
        # ⭐ 每次调用 start_wav_streaming 开始时清空，只保存本次的 tgt 音频与时间线、src 首次发送时间
        self.tgt_audio_chunks = []
        self.tgt_timeline = []

        self.first_send_timestamp = None

        # ✅ 新增：每轮重置 chunk_id 和索引表，避免串轮
        with self._timeline_lock:
            self._chunk_id = 0
            self._timeline_by_id.clear()
        # ===========================================================/

        print(f"开始从文件流式发送音频: {audio_path}")
        # 读取音频文件；audio_data 形状为 (num_samples, num_channels)，采样率为 sample_rate
        audio_data, sample_rate = sf.read(audio_path, dtype="int16", always_2d=True)
        # 如果是多通道音频，转换为单通道（取平均值）
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1, keepdims=True).astype(np.int16)
        # 转换为一维数组
        audio_data = audio_data[:, 0]

        # 如果音频采样率与输入采样率不匹配，进行重采样
        if sample_rate != self.input_rate:
            # 简单线性插值重采样（可替换为更高质量的重采样算法）
            src_len = audio_data.shape[0]  # 原始采样点数
            dst_len = int(src_len * self.input_rate / sample_rate)  # 目标采样点数
            x_src = np.linspace(0, src_len - 1, src_len)
            x_dst = np.linspace(0, src_len - 1, dst_len)
            # 线性插值重采样，并确保输出为 int16 类型
            audio_data = np.interp(x_dst, x_src, audio_data).astype(np.int16)

        chunk_size = self.input_chunk
        total_samples = audio_data.shape[0]
        idx = 0

                
        # 按chunk发送音频数据，并根据实际发送的样本数控制发送速率，模拟实时流式传输
        while self.is_connected and idx < total_samples:
            # 获取当前chunk的音频数据
            chunk = audio_data[idx: idx + chunk_size]
            if chunk.size == 0:
                break
            # ===========================================================\
            # # ⭐ 以「发送」为参考：记录第一次发送 src 的墙钟时间（只记一次）
            # if self.first_send_timestamp is None:
            #     self.first_send_timestamp = time.time()
            # ===========================================================/
            # 将音频数据转换为字节并发送
            await self.send_audio_chunk(chunk.tobytes())  # 发送input audio chunk => Client -> Server: input_audio_buffer.append
            # 更新索引以发送下一个chunk
            idx += chunk_size
            # ✅ 根据实际发送的样本数控制发送速率，模拟实时流式传输
            await asyncio.sleep(chunk.shape[0] / self.input_rate)

        # ！！！cursor改的下面这段代码 ++++++++++++++++++++++++++++++++++++
        # # 用 PyAudio 输出流做天然限速：stream.write() 会阻塞到声卡播完这块，
        # # 效果和麦克风的 stream.read() 一样——硬件按采样率消费，无需 asyncio.sleep。
        # pace_stream = self.pyaudio_instance.open(
        #     format=pyaudio.paInt16,
        #     channels=1,
        #     rate=self.input_rate,
        #     output=True,
        #     frames_per_buffer=chunk_size,
        # )
        # loop = asyncio.get_event_loop()

        # try:
        #     while self.is_connected and idx < total_samples:
        #         chunk = audio_data[idx: idx + chunk_size]
        #         if chunk.size == 0:
        #             break
        #         chunk_bytes = chunk.tobytes()

        #         if self.first_send_timestamp is None:
        #             self.first_send_timestamp = time.time()

        #         # 阻塞式写入声卡，天然按实时速度节流（和麦克风 read 一样）
        #         await loop.run_in_executor(None, pace_stream.write, chunk_bytes)
        #         await self.send_audio_chunk(chunk_bytes)

        #         idx += chunk_size
        # finally:
        #     try:
        #         pace_stream.stop_stream()
        #     except OSError:
        #         pass
        #     try:
        #         pace_stream.close()
        #     except OSError:
        #         pass
        # +++++++++++++++++++++++++++++++++++++++++++

        # 音频已全部发送；等待服务端把最后一段翻译/语音推完，再关闭连接，否则 handle_server_messages 会一直卡在 async for message in self.ws
        wait_after_send_sec = 60.0  # 可按需调整
        print(f"[INFO] 音频发送完毕，等待 {wait_after_send_sec}s 收尾后关闭连接…")
        await asyncio.sleep(wait_after_send_sec)
        if self.ws:
            await self.ws.close()
            print("[INFO] 已关闭 WebSocket，等待收尾后退出。")

    # ------------------------------------------------------------
    # 关闭
    async def close(self):
        """优雅地关闭连接和资源。"""
        self.is_connected = False
        if self.ws:
            await self.ws.close()
            print("WebSocket 连接已关闭。")
        
        # ===========================================================\
        # ⭐ 若设置了保存路径，将本轮的 tgt 音频按真实播放时间轴渲染成 wav（gap 填静音）
        if self.save_tgt_wav_path and self.tgt_audio_chunks:
            os.makedirs(os.path.dirname(self.save_tgt_wav_path) or ".", exist_ok=True)
            t0 = self.first_send_timestamp or 0.0

            # 先补齐 timeline（和下面保存 json 一样的逻辑），确保每条都有 heard_start
            with self._timeline_lock:
                prev_heard_end_ts = t0
                for ent in self.tgt_timeline:
                    if ent.get("heard_start") is None:
                        ent["heard_start"] = float(max(ent["receive_timestamp"], prev_heard_end_ts))
                        ent["heard_end"] = float(ent["heard_start"] + ent["duration_sec"])
                    prev_heard_end_ts = ent["heard_end"]

            # 按 heard_start 时间轴渲染：gap 填静音，overlap 紧接
            write_cursor = 0  # 当前已写到的采样点位置
            rendered_parts: list[np.ndarray] = []
            rate = self.output_rate

            for i, chunk_bytes in enumerate(self.tgt_audio_chunks):
                chunk_samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                ent = self.tgt_timeline[i] if i < len(self.tgt_timeline) else None

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
                samples = np.concatenate(rendered_parts)
            else:
                samples = np.array([], dtype=np.int16)

            sf.write(self.save_tgt_wav_path, samples, rate, subtype="PCM_16")
            print(f"[INFO] 已保存 tgt 音频（含 gap 静音）: {self.save_tgt_wav_path}")

        # 保存 tgt 音频时间线 json（含 first_send_timestamp，供以「发送」为参考的物理时间）
        if self.save_timeline_path and self.tgt_timeline:
            # offset_sec = 该块在「拼接后 wav」中的起始位置（内容时间轴，无静音）
            # output_time_sec = 该块真实被收到/播放的时刻，相对首次发送的秒数（= receive_timestamp - first_send_timestamp）
            # t0 = self.first_send_timestamp
            # timeline = []
            # for o, d, t in self.tgt_timeline:
            #     entry = {"offset_sec": o, "duration_sec": d, "receive_timestamp": t}
            #     if t0 is not None:
            #         entry["output_time_sec"] = t - t0  # 真实输出时间（秒，相对会话开始）
            #     timeline.append(entry)
            # payload = {
            #     "first_send_timestamp": t0,  # 第一次发送 src 的墙钟时间；None 表示未记录
            #     "timeline": timeline,
            # }

            # ✅ NEW: 保存前补齐 output_time_sec，避免 None
            if self.first_send_timestamp is not None:
                with self._timeline_lock:
                    # prev_end = 0.0
                    # for ent in self.tgt_timeline:
                    #     # 观测到的接收时间（相对 first_send）
                    #     recv_base = ent["receive_timestamp"] - self.first_send_timestamp

                    #     # 如果没有真实播放时间，就用"单调下界"补齐
                    #     if ent.get("output_time_sec") is None:
                    #         base = max(recv_base, prev_end)
                    #         ent["output_time_sec"] = float(base)

                    #     # play_timestamp_before 没有的话，给个兜底
                    #     if ent.get("play_timestamp_before") is None:
                    #         ent["play_timestamp_before"] = float(ent["receive_timestamp"])
                    #     if ent.get("play_timestamp_after") is None:
                    #         ent["play_timestamp_after"] = float(ent["receive_timestamp"]) + ent["duration_sec"]

                    #     prev_end = ent["output_time_sec"] + ent["duration_sec"]
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
                    prev_heard_end = 0.0
                    for ent in self.tgt_timeline:
                        recv_ts = ent["receive_timestamp"]
                        dur = ent["duration_sec"]

                        if ent.get("heard_start") is None:
                            ent["heard_start"] = float(max(recv_ts, prev_heard_end))
                            ent["heard_end"] = float(ent["heard_start"] + dur)
                        if ent.get("write_before") is None:
                            ent["write_before"] = ent["heard_start"]
                        if ent.get("write_after") is None:
                            ent["write_after"] = ent["heard_end"]
                        if ent.get("output_time_sec") is None:
                            ent["output_time_sec"] = float(ent["heard_start"] - self.first_send_timestamp)

                        prev_heard_end = ent["heard_end"]
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

                    timeline_copy = list(self.tgt_timeline)  # ✅ NEW: 在锁内复制
            else:
                # 没有 first_send_timestamp 就只能原样存
                with self._timeline_lock:
                    timeline_copy = list(self.tgt_timeline)
                    
            payload = {
                "first_send_timestamp": self.first_send_timestamp,
                "timeline": timeline_copy,   # ✅ 用复制的，不直接用 self.tgt_timeline
            }
            # 保存为 json 文件
            # os.makedirs(os.path.dirname(self.save_timeline_path) or ".", exist_ok=True)
            # with open(self.save_timeline_path, "w", encoding="utf-8") as f:
            #     json.dump(payload, f, ensure_ascii=False, indent=2)
            # print(f"[INFO] 已保存 tgt 时间线: {self.save_timeline_path}")

        # 若设置了 src/tgt/timeline 路径，追加一条记录到 output_qwen_livetranslate2/manifest.jsonl
        if self.manifest_src_path and self.save_tgt_wav_path and self.save_timeline_path:
            manifest_path = os.path.join(os.path.dirname(self.save_tgt_wav_path), "manifest.jsonl")
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            record = {
                "src": self.manifest_src_path,
                "tgt": self.save_tgt_wav_path,
                # "tgt_timeline": self.save_timeline_path,
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[INFO] 已追加记录到: {manifest_path}")
        # ===========================================================/

        if self.audio_player_thread:
            self.audio_playback_queue.put(None)  # 发送结束信号
            self.audio_player_thread.join(timeout=30)  # 等播放线程把剩余块播完
            print("音频播放线程已停止。")

        self.pyaudio_instance.terminate()
        print("PyAudio 实例已释放。")


import os
import asyncio

def print_banner():
    print("=" * 60)
    print("  基于千问 qwen3-livetranslate-flash-realtime")
    print("=" * 60 + "\n")

def get_user_config():
    """
    获取用户配置：
    target language: 输出的翻译目标语言
    voice: 输出的语音合成声音
    audio_enabled: 输出是否启用音频输出
    """
    print("请选择模式:")
    print("1. 语音+文本 [默认] | 2. 仅文本")
    mode_choice = input("请输入选项 (直接回车选择语音+文本): ").strip()
    audio_enabled = (mode_choice != "2")

    if audio_enabled:
        lang_map = {
            "1": "en", "2": "zh", "3": "ru", "4": "fr", "5": "de", "6": "pt",
            "7": "es", "8": "it", "9": "ko", "10": "ja", "11": "yue"
        }
        print("请选择翻译目标语言 (音频+文本 模式):")
        print("1. 英语 | 2. 中文 | 3. 俄语 | 4. 法语 | 5. 德语 | 6. 葡萄牙语 | 7. 西班牙语 | 8. 意大利语 | 9. 韩语 | 10. 日语 | 11. 粤语")
    else:
        lang_map = {
            "1": "en", "2": "zh", "3": "ru", "4": "fr", "5": "de", "6": "pt", "7": "es", "8": "it",
            "9": "id", "10": "ko", "11": "ja", "12": "vi", "13": "th", "14": "ar",
            "15": "yue", "16": "hi", "17": "el", "18": "tr"
        }
        print("请选择翻译目标语言 (仅文本 模式):")
        print("1. 英语 | 2. 中文 | 3. 俄语 | 4. 法语 | 5. 德语 | 6. 葡萄牙语 | 7. 西班牙语 | 8. 意大利语 | 9. 印尼语 | 10. 韩语 | 11. 日语 | 12. 越南语 | 13. 泰语 | 14. 阿拉伯语 | 15. 粤语 | 16. 印地语 | 17. 希腊语 | 18. 土耳其语")

    choice = input("请输入选项 (默认取第一个): ").strip()
    target_language = lang_map.get(choice, next(iter(lang_map.values())))

    voice = None
    if audio_enabled:
        print("\n请选择语音合成声音:")
        voice_map = {"1": "Cherry", "2": "Nofish", "3": "Sunny", "4": "Jada", "5": "Dylan", "6": "Peter", "7": "Eric", "8": "Kiki"}
        print("1. Cherry (女声) [默认] | 2. Nofish (男声) | 3. 晴儿 Sunny (四川女声) | 4. 阿珍 Jada (上海女声) | 5. 晓东 Dylan (北京男声) | 6. 李彼得 Peter (天津男声) | 7. 程川 Eric (四川男声) | 8. 阿清 Kiki (粤语女声)")
        voice_choice = input("请输入选项 (直接回车选择Cherry): ").strip()
        voice = voice_map.get(voice_choice, "Cherry")
    return target_language, voice, audio_enabled

async def main(audio_path: str = "data/2022.acl-long.268.wav", out_dir: str = "data/output_qwen_livetranslate"):
    """主程序入口"""
    print_banner()
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 请设置环境变量 DASHSCOPE_API_KEY")
        print("  例如: export DASHSCOPE_API_KEY='your_api_key_here'")
        return
        
    target_language, voice, audio_enabled = get_user_config()
    print("\n配置完成:")
    print(f"  - 目标语言: {target_language}")
    if audio_enabled:
        print(f"  - 合成声音: {voice}")
    else:
        print("  - 输出模式: 仅文本")
    
    # ===========================================================\
    # ⭐ 若需保存本轮的 tgt 音频与时间线，存到 {音频所在目录}/output_qwen_livetranslate2/{文件名}_tgt.wav
    base, _ = os.path.splitext(audio_path)
    save_tgt_wav_path = os.path.join(out_dir, os.path.basename(base) + "_tgt.wav")
    save_timeline_path = os.path.join(out_dir, os.path.basename(base) + "_timeline.json")
    # ===========================================================/
    
    client = LiveTranslateClient(
        api_key=api_key,
        target_language=target_language,
        voice=voice,
        audio_enabled=audio_enabled,
        manifest_src_path=audio_path,
        save_tgt_wav_path=save_tgt_wav_path,
        save_timeline_path=save_timeline_path,
    )

    # ------------------------------------------------------------
    
    # 定义回调函数
    def on_translation_text(text):
        print(text, end="", flush=True)

    try:
        print("正在连接到翻译服务...")
        await client.connect()
        
        # 根据模式启动音频播放
        client.start_audio_player()
        
        print("\n" + "-" * 60)
        print(f"连接成功！将从 wav 文件读取音频: {audio_path}")
        print("程序将实时翻译该音频并播放结果。")
        print("-" * 60 + "\n")

        # 并发运行消息处理和 wav 音频流发送
        message_handler = asyncio.create_task(client.handle_server_messages(on_translation_text))  # 后台持续处理服务端消息
        tasks = [message_handler]
        microphone_streamer = asyncio.create_task(client.start_wav_streaming(audio_path))  # 后台持续发送 wav 音频流
        tasks.append(microphone_streamer)

        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    except Exception as e:
        print(f"\n发生严重错误: {e}")
    finally:
        print("\n正在清理资源...")
        await client.close()
        print("程序已退出。")

'''
conda activate s2s_latency
conda run --no-capture-output -n s2s_latency python -u qwen_livetranslate_wav4.py
'''

if __name__ == "__main__":
    asyncio.run(main(audio_path="data/input/acl_6060_dev/2022.acl-long.117.wav", out_dir="data/output_qwen_wav5"))
