import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple


# ================================================================
# 配置（改这里）
# ================================================================
@dataclass
class PipelineConfig:
    # 类级常量：不属于 __init__ 参数，也不随实例复制一份
    LANGUAGE_NAME_MAP: ClassVar[Dict[str, str]] = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "ja": "Japanese",
    }
    BLEU_MAP: ClassVar[Dict[str, str]] = {
        "en": "13a",
        "zh": "zh",
        "de": "13a",
        "ja": "ja-mecab",
    }

    # ── 系统标识 ──────────────────────────────────────────────────────
    model_name: str = "seed"
    src_lang: str = "en"
    tgt_lang: str = "zh"
    output_version: str = ""

    # ── 输入路径（改这里换数据集或待评估系统）────────────────────────
    src_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_dir: str = "input/acl_6060_dev_tgt_seed/en_zh"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
    output_dir: str = "output"

    # ── 运行参数 ───────────────────────────────────────────────────────
    asr_backend: str = "transformers"   # "transformers" 或 "vllm"
    batch_size: int = 2
    max_new_tokens: int = 1024
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"
    only_doc_ids: Optional[Tuple[str, ...]] = None  # 画图时仅看这几个 doc；None = 全部

    # ── 派生输出路径（随 model_name / 语言对 / output_version 变化）────
    @property
    def _output_base(self) -> str:
        return f"{self.output_dir}/{self.model_name}/{self.src_lang}_{self.tgt_lang}"

    @property
    def manifest(self) -> str:
        return f"{self._output_base}/output_asr{self.output_version}/manifest.jsonl"

    @property
    def output_dir_asr(self) -> str:
        return f"{self._output_base}/output_asr{self.output_version}"

    @property
    def output_dir_asr_enriched(self) -> str:
        return self.output_dir_asr + "_"

    @property
    def output_path_instances(self) -> str:
        return os.path.join(self.output_dir_asr_enriched, "instances.log")

    @property
    def output_dir_segale(self) -> str:
        return f"{self._output_base}/output_segale{self.output_version}"

    @property
    def segale_file(self) -> str:
        return os.path.join(self.output_dir_segale, "hyp/aligned_spacy_hyp.jsonl")

    @property
    def output_dir_evaluation(self) -> str:
        return f"{self._output_base}/output_evaluation{self.output_version}"

    def build_manifest(self, manifest_path: str = None) -> str:
        """
        从 src_dir / tgt_dir 自动配对 src/tgt wav，生成 manifest.jsonl。
        按 basename 配对，找不到对应 tgt 的 src 跳过。
        返回写出的 manifest 路径。
        """
        import json

        if manifest_path is None:
            manifest_path = self.manifest

        tgt_wavs = {
            os.path.splitext(f)[0]: os.path.join(self.tgt_dir, f)
            for f in os.listdir(self.tgt_dir)
            if f.endswith(".wav")
        }

        records = []
        for src_file in sorted(os.listdir(self.src_dir)):
            if not src_file.endswith(".wav"):
                continue
            stem = os.path.splitext(src_file)[0]
            if stem not in tgt_wavs:
                continue
            records.append({
                "src": os.path.join(self.src_dir, src_file).replace("\\", "/"),
                "tgt": tgt_wavs[stem].replace("\\", "/"),
            })

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"manifest 已生成：{manifest_path}（{len(records)} 条）")
        return manifest_path
