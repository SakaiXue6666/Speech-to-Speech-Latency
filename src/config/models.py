from dataclasses import dataclass


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


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "seed"
    input_version: str = ""
    output_version: str = ""
    src_lang: str = "en"
    tgt_lang: str = "ja"
    asr_backend: str = "transformers"
    batch_size: int = 3
    max_new_tokens: int = 1024

