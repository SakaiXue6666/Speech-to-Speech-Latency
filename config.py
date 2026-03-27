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
    tgt_lang: str = "ja"
    asr_backend: str = "transformers"          # "transformers" 或 "vllm"
    batch_size: int = 3
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

    def build_paths(self) -> StepPaths:
        base = f"output/{self.model_name}/{self.src_lang}_{self.tgt_lang}"
        asr = f"{base}/output_asr{self.output_version}"
        asr_enriched = asr + "_"
        segale = f"{base}/output_segale{self.output_version}"
        return StepPaths(
            src_segments_yaml=f"input/ACL.ACLdev2023.{self.src_lang}-xx.gold_segments.yaml",
            src_txt=f"input/acl_6060_dev/text/txt/ACL.6060.dev.{self.src_lang}-xx.{self.src_lang}.txt",
            tgt_ref_txt=f"input/acl_6060_dev/text/txt/ACL.6060.dev.{self.src_lang}-xx.{self.tgt_lang}.txt",
            manifest=f"input/acl_6060_dev_tgt_{self.model_name}/{self.src_lang}_{self.tgt_lang}/manifest.jsonl",
            output_dir_asr=asr,
            output_dir_asr_enriched=asr_enriched,
            output_path_instances=os.path.join(asr_enriched, "instances.log"),
            output_dir_segale=segale,
            segale_file=os.path.join(segale, "hyp/aligned_spacy_hyp.jsonl"),
            output_dir_longyaal=f"{base}/output_longyaal{self.output_version}",
        )
