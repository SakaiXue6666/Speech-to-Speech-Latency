import gc
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 让 HuggingFace 完全跳过网络请求，直接读本地缓存

import torch

from config import PipelineConfig
from src.asr import step1_asr, step1_asr_vllm
from src.asr.qwen_char_spans import add_char_spans_for_dir
from src.alignment import step2_segale
from src.evaluation import step3_longyaal
from src.intermediate.prepare_artifacts import asr_to_instances, instances_to_segale
from src.plot.plot import (
    ending_offset_delay,
    delay_span_vs_source_length,
    tgt_minus_src_length_histogram,
)


def main():
    cfg = PipelineConfig()
    paths = cfg.build_paths()

    # ================================================================
    # 想单独跑某 step：注释其他 steps 就好
    # ================================================================

    # step1 asr -------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting ASR...")
    if cfg.asr_backend == "vllm":
        step1_asr_vllm(
            manifest=paths.manifest,
            tgt_language=cfg.tgt_language,
            out_dir=paths.output_dir_asr,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            gpu_memory_utilization=0.7,
        )
    else:
        step1_asr(
            manifest=paths.manifest,
            tgt_language=cfg.tgt_language,
            out_dir=paths.output_dir_asr,
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
        paths.output_dir_asr,
        paths.output_dir_asr_enriched,
    )
    # 中间过程，生成 instances.log
    asr_to_instances(
        s2s=True,
        yaml_file=paths.src_segments_yaml,
        asr_dir=paths.output_dir_asr_enriched,
        output_file=paths.output_path_instances,
    )
    # 中间过程，生成 segale 需要的 hyp.jsonl 和 ref.jsonl
    instances_to_segale(
        src_txt=paths.src_txt,
        tgt_ref_txt=paths.tgt_ref_txt,
        src_segments_yaml=paths.src_segments_yaml,
        instances=paths.output_path_instances,
        out_dir=paths.output_dir_segale,
    )

    # step2 segale ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Segale...")
    step2_segale(
        system_file=os.path.join(paths.output_dir_segale, "hyp.jsonl"),
        ref_file=os.path.join(paths.output_dir_segale, "ref.jsonl"),
        segmenter="spacy",
        task_lang=cfg.tgt_lang,
        proc_device=cfg.proc_device,
        embedding_model=cfg.embedding_model,
    )
    print("Segale finished.")

    # step3 longyaal ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Longyaal...")
    step3_longyaal(
        yaml_file=paths.src_segments_yaml,
        source_sentences_file=paths.src_txt,
        instances_log=paths.output_path_instances,
        segale_file=paths.segale_file,
        output_folder=paths.output_dir_longyaal,
        bleu_tokenizer=cfg.bleu_tokenizer,
    )
    print("Longyaal finished.")

    # ending offset delay ---------------------------------------------------------------
    ending_offset_delay(
        instances=os.path.join(paths.output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(paths.output_dir_longyaal, "last_delay_minus_recording_end.png"),
        out_csv=os.path.join(paths.output_dir_longyaal, "last_delay_minus_recording_end.csv"),
    )
    delay_span_vs_source_length(
        instances=os.path.join(paths.output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(paths.output_dir_longyaal, "delay_span_vs_source_length.png"),
    )
    tgt_minus_src_length_histogram(
        instances=os.path.join(paths.output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(paths.output_dir_longyaal, "tgt_minus_src_length_histogram.png"),
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        only_doc_ids=cfg.only_doc_ids,
    )


if __name__ == "__main__":
    main()
