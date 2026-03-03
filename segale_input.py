import json
import yaml
import os

def get_jsonl(src_text, tgt_ref_text, ref_segments, instances):
    with open(src_text, "r", encoding="utf-8") as f:
        src_text_lines = [line.rstrip("\n") for line in f]

    with open(tgt_ref_text, "r", encoding="utf-8") as f:
        tgt_ref_text_lines = [line.rstrip("\n") for line in f]

    with open(ref_segments, "r", encoding="utf-8") as f:
        src_info_lines = yaml.safe_load(f)

    src_for_hyp_docs = []
    ref_dicts = []
    curr_doc = None

    for i in range(len(src_info_lines)):
        ref_dict = {
            "src": src_text_lines[i],
            "tgt": tgt_ref_text_lines[i],
            "sys_id": None,
            "doc_id": src_info_lines[i]['wav'],
            "seg_id": i + 1
        }
        ref_dicts.append(ref_dict)
        # -----------------------------------------------------------
        if src_info_lines[i]['wav'] != curr_doc:
            src_for_hyp_docs.append({"doc_id": src_info_lines[i]['wav'], "src": ""})
            curr_doc = src_info_lines[i]['wav']
        src_for_hyp_docs[-1]["src"] += " " + src_text_lines[i]
    # -----------------------------------------------------------
    with open(instances, "r", encoding="utf-8") as f:
        instances_lines = [line.rstrip("\n") for line in f]
    
    hyp_dicts = []
    for j in range(len(instances_lines)):
        instance_dict = json.loads(instances_lines[j])
        hyp_text = instance_dict["prediction_text"]
        for src_for_hyp_doc in src_for_hyp_docs:
            if src_for_hyp_doc["doc_id"] == os.path.basename(str(instance_dict["source"] or "")).replace("\\", "/"):
                hyp_dict = {
                    "src": src_for_hyp_doc["src"],
                    "tgt": hyp_text,
                    "sys_id": None,
                    "doc_id": src_for_hyp_doc["doc_id"],
                    "seg_id": j + 1
                }
                hyp_dicts.append(hyp_dict)
                break

    return ref_dicts, hyp_dicts

ref_dicts, hyp_dicts = get_jsonl(
    "data/input/ACL.6060.dev.en-xx.en.txt", 
    "data/input/ACL.6060.dev.en-xx.zh.txt", 
    "data/input/ACL.ACLdev2023.en-xx.gold_segments.yaml", 
    "data/output_qwen_asr/instances.log")

# 保存 jsonl
def save_jsonl(dict_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in dict_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 输出到 data/segale/
save_jsonl(ref_dicts, "data/output_segale/ref.jsonl")
save_jsonl(hyp_dicts, "data/output_segale/hyp.jsonl")

print("saved:", "data/output_segale/ref.jsonl", "data/output_segale/hyp.jsonl")