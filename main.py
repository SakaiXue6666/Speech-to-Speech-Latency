import gc
import os
import torch

from config import PipelineConfig
from src.asr import step1_asr, step1_asr_vllm
from src.asr.qwen_char_spans import add_char_spans_for_dir
from src.alignment import step2_segale
from src.evaluation import step3_longyaal
from src.intermediate.prepare_artifacts import asr_to_instances, instances_to_segale


def get_tgt_language(tgt_lang: str) -> str:
    return PipelineConfig.LANGUAGE_NAME_MAP.get(tgt_lang, tgt_lang)


def get_bleu_tokenizer(tgt_lang: str) -> str:
    return PipelineConfig.BLEU_MAP.get(tgt_lang, "13a")


def main():
    print("[DEBUG] 环境变量确认:", "HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"), "| HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
    print("[DEBUG] 开始 build_manifest...")
    cfg = PipelineConfig()
    cfg.build_manifest()   # 自动扫目录生成 manifest.jsonl
    print("[DEBUG] build_manifest 完成")

    # ================================================================
    # 想单独跑某 step：注释其他 steps 就好
    # ================================================================

    # step1 asr -------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting ASR...")
    if cfg.asr_backend == "vllm":
        step1_asr_vllm(
            manifest=cfg.manifest,
            tgt_language=get_tgt_language(cfg.tgt_lang),
            out_dir=cfg.output_dir_asr,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            gpu_memory_utilization=0.7,
        )
    else:
        step1_asr(
            manifest=cfg.manifest,
            tgt_language=get_tgt_language(cfg.tgt_lang),
            out_dir=cfg.output_dir_asr,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("ASR finished.")

    # ---------------------------------------------------------------
    # 中间过程，算 asr 的 unit 索引
    add_char_spans_for_dir(
        cfg.output_dir_asr,
        cfg.output_dir_asr_enriched,
    )
    # 中间过程，生成 instances.log
    asr_to_instances(
        s2s=True,
        yaml_file=cfg.src_segments_yaml,
        asr_dir=cfg.output_dir_asr_enriched,
        output_file=cfg.output_path_instances,
    )
    # 中间过程，生成 segale 需要的 hyp.jsonl 和 ref.jsonl
    instances_to_segale(
        src_txt=cfg.src_txt,
        tgt_ref_txt=cfg.tgt_ref_txt,
        src_segments_yaml=cfg.src_segments_yaml,
        instances=cfg.output_path_instances,
        out_dir=cfg.output_dir_segale,
    )

    # step2 segale ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Segale...")
    step2_segale(
        system_file=os.path.join(cfg.output_dir_segale, "hyp.jsonl"),
        ref_file=os.path.join(cfg.output_dir_segale, "ref.jsonl"),
        segmenter="spacy",
        task_lang=cfg.tgt_lang,
        proc_device=cfg.proc_device,
        embedding_model=cfg.embedding_model,
    )
    print("Segale finished.")

    # step3 evaluation ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Evaluation...")
    step3_longyaal(
        yaml_file=cfg.src_segments_yaml,
        source_sentences_file=cfg.src_txt,
        instances_log=cfg.output_path_instances,
        segale_file=cfg.segale_file,
        output_folder=cfg.output_dir_evaluation,
        bleu_tokenizer=get_bleu_tokenizer(cfg.tgt_lang),
    )
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
