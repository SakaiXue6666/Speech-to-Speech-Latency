import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ================================================================
# 语言映射（一般不用改）
# ================================================================
LANGUAGE_NAME_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ja": "Japanese",
}

BLEU_MAP: Dict[str, str] = {
    "en": "13a",
    "zh": "zh",
    "de": "13a",
    "ja": "ja-mecab",
}


# ================================================================
# 路径数据类（自动生成，不用手动填）
# ================================================================
@dataclass(frozen=True)
class StepPaths:
    src_segments_yaml: str
    src_txt: str
    tgt_ref_txt: str
    manifest: str
    output_dir_asr: str
    output_dir_asr_enriched: str
    output_path_instances: str
    output_dir_segale: str
    segale_file: str
    output_dir_longyaal: str


# ================================================================
# 配置（改这里）
# ================================================================
@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "seed"
    input_version: str = ""
    output_version: str = ""
    src_lang: str = "en"
    tgt_lang: str = "zh"
    asr_backend: str = "transformers"          # "transformers" 或 "vllm"
    batch_size: int = 2
    max_new_tokens: int = 1024
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"
    only_doc_ids: Optional[Tuple[str, ...]] = ("110", "117")  # 画图时仅看这几个 doc；None 表示全部

    @property
    def tgt_language(self) -> str:
        return LANGUAGE_NAME_MAP.get(self.tgt_lang, self.tgt_lang)

    @property
    def bleu_tokenizer(self) -> str:
        return BLEU_MAP.get(self.tgt_lang, "13a")

    def _manifest_path(self) -> str:
        """manifest 默认存放在 ASR 输出目录下。"""
        base = f"output/{self.model_name}/{self.src_lang}_{self.tgt_lang}"
        return f"{base}/output_asr{self.output_version}/manifest.jsonl"

    def build_manifest(self, manifest_path: str = None) -> str:
        """
        从目录结构自动配对 src/tgt wav，生成 manifest.jsonl。
        src: input/acl_6060_dev/full_wavs/*.wav
        tgt: input/acl_6060_dev_tgt_{model_name}/{src_lang}_{tgt_lang}/*.wav
        按 basename 配对，找不到对应 tgt 的 src 跳过。
        返回写出的 manifest 路径。
        """
        import json

        src_dir = "input/acl_6060_dev/full_wavs"
        tgt_dir = f"input/acl_6060_dev_tgt_{self.model_name}/{self.src_lang}_{self.tgt_lang}"

        if manifest_path is None:
            manifest_path = self._manifest_path()

        tgt_wavs = {
            os.path.splitext(f)[0]: os.path.join(tgt_dir, f)
            for f in os.listdir(tgt_dir)
            if f.endswith(".wav")
        }

        records = []
        for src_file in sorted(os.listdir(src_dir)):
            if not src_file.endswith(".wav"):
                continue
            stem = os.path.splitext(src_file)[0]
            if stem not in tgt_wavs:
                continue
            tgt_wav = tgt_wavs[stem]
            records.append({
                "src": os.path.join(src_dir, src_file).replace("\\", "/"),
                "tgt": tgt_wav.replace("\\", "/"),
            })

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"manifest 已生成：{manifest_path}（{len(records)} 条）")
        return manifest_path

    def build_paths(self) -> StepPaths:
        base = f"output/{self.model_name}/{self.src_lang}_{self.tgt_lang}"
        asr = f"{base}/output_asr{self.output_version}"
        asr_enriched = asr + "_"
        segale = f"{base}/output_segale{self.output_version}"
        return StepPaths(
            src_segments_yaml=f"input/ACL.ACLdev2023.{self.src_lang}-xx.gold_segments.yaml",
            src_txt=f"input/acl_6060_dev/text/txt/ACL.6060.dev.{self.src_lang}-xx.{self.src_lang}.txt",
            tgt_ref_txt=f"input/acl_6060_dev/text/txt/ACL.6060.dev.{self.src_lang}-xx.{self.tgt_lang}.txt",
            manifest=self._manifest_path(),
            output_dir_asr=asr,
            output_dir_asr_enriched=asr_enriched,
            output_path_instances=os.path.join(asr_enriched, "instances.log"),
            output_dir_segale=segale,
            segale_file=os.path.join(segale, "hyp/aligned_spacy_hyp.jsonl"),
            output_dir_longyaal=f"{base}/output_longyaal{self.output_version}",
        )
