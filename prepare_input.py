"""读取源英语转录和中文参考翻译，生成 main.py 所需的输入 JSON。"""
import json

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

src_transcript_lines = read_lines("data/ACL.6060.dev.en-xx.en.txt")
tgt_ref_lines = read_lines("data/ACL.6060.dev.en-xx.zh.txt")

payload = {
    "samples": [
        {
            "sample_id": "acl_6060",
            "src": {
                "audio_path": "data/2022.acl-long.268.wav",
                "transcript": " ".join(src_transcript_lines),
            },
            "tgt": {
                "audio_path": "data/output/wavs/0_pred.wav",
                "reference_transcript": " ".join(tgt_ref_lines),
            },
        }
    ]
}

out_path = "data/input_samples.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"已生成: {out_path}")
print(f"  src transcript: {len(src_transcript_lines)} 句, {len(payload['samples'][0]['src']['transcript'])} 字符")
print(f"  tgt reference:  {len(tgt_ref_lines)} 句, {len(payload['samples'][0]['tgt']['reference_transcript'])} 字符")
