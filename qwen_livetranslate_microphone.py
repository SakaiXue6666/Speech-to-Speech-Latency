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

import wave


'''
conda run --no-capture-output -n s2s_latency python -u qwen_livetranslate_microphone.py
'''

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
    def __init__(self, api_key: str, target_language: str = "en", voice: str | None = "Cherry", *, audio_enabled: bool = True):
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
        self.input_chunk = 1600
        self.input_format = pyaudio.paInt16
        self.input_channels = 1
        
        # 音频输出配置 (用于播放)
        self.output_rate = 24000
        self.output_chunk = 2400
        self.output_format = pyaudio.paInt16
        self.output_channels = 1
        
        # 状态管理
        self.is_connected = False
        self.audio_player_thread = None
        self.audio_playback_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()

        self.audio_bytes = bytearray()  # ⭐ 用于累积接收到的 PCM 音频数据

    async def connect(self):
        """建立到翻译服务的 WebSocket 连接。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            self.ws = await websockets.connect(self.api_url, additional_headers=headers)
            self.is_connected = True
            print(f"成功连接到服务端: {self.api_url}")
            await self.configure_session()
        except Exception as e:
            print(f"连接失败: {e}")
            self.is_connected = False
            raise

    async def configure_session(self):
        """配置翻译会话，设置目标语言、声音等。"""
        config = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                # 'modalities' 控制输出类型。
                # ["text", "audio"]: 同时返回翻译文本和合成音频（推荐）。
                # ["text"]: 仅返回翻译文本。
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

    async def send_audio_chunk(self, audio_data: bytes):
        """将音频数据块编码并发送到服务端。"""
        if not self.is_connected:
            return
            
        event = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode()
        }
        await self.ws.send(json.dumps(event))

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

    def _audio_player_task(self):
        stream = self.pyaudio_instance.open(
            format=self.output_format,
            channels=self.output_channels,
            rate=self.output_rate,
            output=True,
            frames_per_buffer=self.output_chunk,
        )
        try:
            while self.is_connected or not self.audio_playback_queue.empty():
                try:
                    audio_chunk = self.audio_playback_queue.get(timeout=0.1)
                    if audio_chunk is None: # 结束信号
                        break
                    stream.write(audio_chunk)
                    self.audio_playback_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            stream.stop_stream()
            stream.close()

    def start_audio_player(self):
        """启动音频播放线程（仅当启用音频输出时）。"""
        if not self.audio_enabled:
            return
        if self.audio_player_thread is None or not self.audio_player_thread.is_alive():
            self.audio_player_thread = threading.Thread(target=self._audio_player_task, daemon=True)
            self.audio_player_thread.start()

    async def handle_server_messages(self, on_text_received):
        """循环处理来自服务端的消息。"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "response.audio.delta" and self.audio_enabled:
                    print("[DEBUG] 收到 response.audio.delta 事件")
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        print(f"[DEBUG] 音频块入队，bytes={len(audio_data)}")
                        self.audio_playback_queue.put(audio_data)
                        self.audio_bytes.extend(audio_data)  # ⭐ 新增：累积 PCM

                elif event_type == "response.done":
                    print("\n[INFO] 一轮响应完成。")
                    usage = event.get("response", {}).get("usage", {})
                    if usage:
                        print(f"[INFO] Token 使用情况: {json.dumps(usage, indent=2, ensure_ascii=False)}")
                # 处理源语言识别结果（需启用 input_audio_transcription.model）
                # elif event_type == "conversation.item.input_audio_transcription.text":
                #     stash = event.get("stash", "")  # 待确认的识别文本
                #     print(f"[识别中] {stash}")
                # elif event_type == "conversation.item.input_audio_transcription.completed":
                #     transcript = event.get("transcript", "")  # 完整识别结果
                #     print(f"[源语言] {transcript}")
                elif event_type == "response.audio_transcript.done":
                    print("\n[INFO] 翻译文本完成。")
                    text = event.get("transcript", "")
                    if text:
                        print(f"[INFO] 翻译文本: {text}")
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

    async def start_microphone_streaming(self):
        """从麦克风捕获音频并流式传输到服务端。"""
        stream = self.pyaudio_instance.open(
            format=self.input_format,
            channels=self.input_channels,
            rate=self.input_rate,
            input=True,
            frames_per_buffer=self.input_chunk
        )
        print("麦克风已启动，请开始说话...")
        try:
            while self.is_connected:
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None, stream.read, self.input_chunk
                )
                await self.send_audio_chunk(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()

    async def close(self):
        """优雅地关闭连接和资源。"""
        self.is_connected = False
        if self.ws:
            await self.ws.close()
            print("WebSocket 连接已关闭。")
        
        if self.audio_player_thread:
            self.audio_playback_queue.put(None) # 发送结束信号
            self.audio_player_thread.join(timeout=1)
            print("音频播放线程已停止。")
            
        self.pyaudio_instance.terminate()
        print("PyAudio 实例已释放。")

        # ⭐ 程序结束前保存累积的 PCM 音频数据到 WAV 文件，便于后续分析或使用。
        if self.audio_bytes:
            with wave.open("translated_output.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)    # int16
                wf.setframerate(24000)
                wf.writeframes(bytes(self.audio_bytes))
            print("已保存音频文件: translated_output.wav")


import os
import asyncio

def print_banner():
    print("=" * 60)
    print("  基于千问 qwen3-livetranslate-flash-realtime")
    print("=" * 60 + "\n")

def get_user_config():
    """获取用户配置"""
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

async def main():
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
    
    client = LiveTranslateClient(api_key=api_key, target_language=target_language, voice=voice, audio_enabled=audio_enabled)
    
    # 定义回调函数
    def on_translation_text(text):
        print(text, end="", flush=True)

    try:
        print("正在连接到翻译服务...")
        await client.connect()
        
        # 根据模式启动音频播放
        client.start_audio_player()
        
        print("\n" + "-" * 60)
        print("连接成功！请对着麦克风说话。")
        print("程序将实时翻译您的语音并播放结果。按 Ctrl+C 退出。")
        print("-" * 60 + "\n")

        # 并发运行消息处理和麦克风录音
        message_handler = asyncio.create_task(client.handle_server_messages(on_translation_text))
        tasks = [message_handler]
        # 无论是否启用音频输出，都需要从麦克风捕获音频进行翻译
        microphone_streamer = asyncio.create_task(client.start_microphone_streaming())
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

if __name__ == "__main__":
    asyncio.run(main())
