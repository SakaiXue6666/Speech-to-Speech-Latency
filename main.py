import argparse
import gc
import os
from datetime import datetime

import torch

from config import PipelineConfig
from src.asr import step1_asr, step1_asr_vllm
from src.asr.qwen_char_spans import add_char_spans_for_dir
from src.alignment import step2_segale
from src.evaluation import step3_longyaal
from src.intermediate.prepare_artifacts import asr_to_instances, instances_to_segale
from src.runtime import write_run_meta


def get_tgt_language(tgt_lang: str) -> str:
    return PipelineConfig.LANGUAGE_NAME_MAP.get(tgt_lang, tgt_lang)


def get_bleu_tokenizer(tgt_lang: str) -> str:
    return PipelineConfig.BLEU_MAP.get(tgt_lang, "13a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speech-to-Speech Latency Evaluation Pipeline",
    )
    p.add_argument("--src-lang", default="en", help="Source language code (default: en)")
    p.add_argument("--tgt-lang", default="ja",
                   help="Target language code: zh / de / ja (default: ja)")
    p.add_argument("--src-audio-dir", default="input/acl_6060_dev/full_wavs",
                   help="Directory containing source audio WAV files")
    p.add_argument("--tgt-audio-dir", default="input/acl_6060_dev_tgt_seed/en_ja",
                   help="Directory containing target (translated) audio WAV files")
    p.add_argument("--src-segments-yaml",
                   default="input/ACL.ACLdev2023.en-xx.gold_segments.yaml",
                   help="Source audio segmentation YAML file")
    p.add_argument("--src-txt",
                   default="input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt",
                   help="Source language transcript text file")
    p.add_argument("--tgt-ref-txt",
                   default="input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt",
                   help="Target language reference translation text file")
    p.add_argument("--output-dir", default="output",
                   help="Output directory for all results")
    return p.parse_args()


def _run_pipeline(cfg: PipelineConfig) -> None:
    """实际跑三步流水线。抽出来方便 main() 用 try/finally 包元信息写出。"""
    cfg.build_manifest()   # 自动扫目录生成 manifest.jsonl

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
    # 中间过程：in-place 给 *_asr.json 加 char span
    add_char_spans_for_dir(cfg.output_dir_asr)

    # 中间过程：生成 instances.log
    asr_to_instances(
        s2s=True,
        yaml_file=cfg.src_segments_yaml,
        asr_dir=cfg.output_dir_asr,
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
        seed=cfg.seed,
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


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        src_audio_dir=args.src_audio_dir,
        tgt_audio_dir=args.tgt_audio_dir,
        src_segments_yaml=args.src_segments_yaml,
        src_txt=args.src_txt,
        tgt_ref_txt=args.tgt_ref_txt,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    started_at = datetime.now().astimezone()
    error: BaseException | None = None
    try:
        _run_pipeline(cfg)
    except BaseException as e:
        error = e
        raise
    finally:
        status = "success" if error is None else "failed"
        meta_path = write_run_meta(
            cfg.output_dir,
            cfg,
            started_at=started_at,
            status=status,
            error=error,
        )
        print(f"[meta] wrote {meta_path}")


if __name__ == "__main__":
    main()
