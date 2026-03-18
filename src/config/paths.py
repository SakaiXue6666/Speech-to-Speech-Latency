import os

from .models import PipelineConfig, StepPaths


def build_step_paths(cfg: PipelineConfig) -> StepPaths:
    src_segments_yaml = f"data2/input/ACL.ACLdev2023.{cfg.src_lang}-xx.gold_segments.yaml"
    src_txt = f"data2/input/ACL.6060.dev.{cfg.src_lang}-xx.{cfg.src_lang}.txt"
    tgt_ref_txt = f"data2/input/ACL.6060.dev.{cfg.src_lang}-xx.{cfg.tgt_lang}.txt"

    output_dir_asr = f"data3/{cfg.model_name}/{cfg.src_lang}_{cfg.tgt_lang}/output_asr{cfg.output_version}"
    manifest = f"data3/{cfg.model_name}/{cfg.src_lang}_{cfg.tgt_lang}/output_tgt{cfg.input_version}/manifest.jsonl"
    output_dir_asr_enriched = output_dir_asr + "_"
    output_path_instances = os.path.join(output_dir_asr_enriched, "instances.log")
    output_dir_segale = f"data3/{cfg.model_name}/{cfg.src_lang}_{cfg.tgt_lang}/output_segale{cfg.output_version}"
    segale_file = os.path.join(output_dir_segale, "hyp/aligned_spacy_hyp.jsonl")
    output_dir_longyaal = f"data3/{cfg.model_name}/{cfg.src_lang}_{cfg.tgt_lang}/output_longyaal{cfg.output_version}"

    return StepPaths(
        src_segments_yaml=src_segments_yaml,
        src_txt=src_txt,
        tgt_ref_txt=tgt_ref_txt,
        manifest=manifest,
        output_dir_asr=output_dir_asr,
        output_dir_asr_enriched=output_dir_asr_enriched,
        output_path_instances=output_path_instances,
        output_dir_segale=output_dir_segale,
        segale_file=segale_file,
        output_dir_longyaal=output_dir_longyaal,
    )

