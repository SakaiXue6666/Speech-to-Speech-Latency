import numpy as np
import wave

pcm_path = r"ast_python\output\translate_audio_00001.pcm"

with open(pcm_path, "rb") as f:
    raw = f.read()

# 方案1：按 int32 little-endian 解释
x_i32 = np.frombuffer(raw, dtype="<i4")
with wave.open("ast_python/output/debug_i32.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(4)
    wf.setframerate(24000)
    wf.writeframes(x_i32.tobytes())

# 方案2：按 float32 little-endian 解释，再转 int16
x_f32 = np.frombuffer(raw, dtype="<f4")
x_f32 = np.nan_to_num(x_f32, nan=0.0, posinf=0.0, neginf=0.0)
x_f32 = np.clip(x_f32, -1.0, 1.0)
x_i16 = (x_f32 * 32767.0).astype("<i2")

with wave.open("ast_python/output/debug_f32_to_i16.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(x_i16.tobytes())