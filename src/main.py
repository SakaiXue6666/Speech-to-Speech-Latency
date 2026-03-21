import gc
import os
import torch

# # cursor写的：添加 SEGALE 路径
# import sys
# # 保证 import pipeline_segale_align_new 时能找到 vecalign（SEGALE 在项目下的 SEGALE 目录）
# _script_dir = os.path.dirname(os.path.abspath(__file__))
# _segale_dir = os.path.join(_script_dir, "SEGALE")
# if os.path.isdir(_segale_dir) and _segale_dir not in sys.path:
#     sys.path.insert(0, _script_dir)
#     sys.path.insert(0, _segale_dir)

from src.asr import step1_asr
from src.asr import step1_asr_vllm
from src.alignment import step2_segale
from src.evaluation import step3_longyaal
from pipeline_plot_ending_offset_delay import (
    ending_offset_delay,
    delay_span_vs_source_length,
    tgt_minus_src_length_histogram,
)

import argparse
import json
import os
import wave
import yaml


# ============================================================================\
import json
import os
import unicodedata
from typing import Dict, List, Tuple


from src.asr.qwen_char_spans import add_char_spans_for_dir, add_char_spans_to_asr_json
from src.intermediate.prepare_artifacts import asr_to_instances, get_jsonl, instances_to_segale


def main():
    # 设置你的路径 =========================================================
    '''
    我自己的路径：
    data2/input
    - 存 src yaml, src txt, ref txt
    - acl_6060_dev/ 存 输入音频.wav

    data3/
    - seed
    -- en_de
    ---- output_tgt/ 存 输出音频.wav、输出文本.txt、manifest.jsonl
    ---- output_asr/ 存 asr 结果.json (step1)
    ---- output_asr_/ 存 asr 加上 unit 的索引.json、instances.log
    ---- output_segale/hyp/ 存 segale 的对齐结果.jsonl (step2)
    ---- output_longyaal/ 存 longyaal 的 分数.csv、ending offset的图.png ... (step2)
    -- 其他语言
    - 其他模型

    ================================================================

    pipeline_main_new.py 是主流程，会调用：

    pipeline_qwen3_asr2.py              # step1: ASR (transformers 后端)
    pipeline_qwen3_asr2vllm.py          # step1: ASR (vllm 后端, 我的windows电脑还没跑出来)
    pipeline_segale_align_new.py        # step2: SEGALE 句子级对齐
    pipeline_longyaal_new.py            # step3: LongYAAL latency 计算
    pipeline_qwen3_forcealign_tokenizer2.py  # tokenizer, longyaal 内部调用
    pipeline_plot_ending_offset_delay.py     # ending offset delay 画图

    pipeline_softsegmenter_longyaal2.py      # 可选，main没调用，单独跑: softsegmenter + longyaal

    --- 项目内子目录 ---
    SEGALE/                             # vecalign 等, pipeline_segale_align_new 依赖

    ================================================================

    spacy 安装：
    de:
    python -m pip install "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl"
    ja:
    python -m pip install ginza ja_ginza
    python -m pip install "sacrebleu[ja]"

    bleu 安装：
    ja:
    pip install "sacrebleu[ja]"

    ================================================================

    完整依赖（pip 包 + spacy 模型 + 项目内 py 文件）：

    --- pip 包 ---
    pip install torch                   # PyTorch（需匹配你的 CUDA 版本）
    pip install numpy
    pip install soundfile               # pipeline_qwen3_asr2 读 wav
    pip install qwen-asr                # Qwen3ASRModel (transformers 后端)
    # pip install qwen-asr[vllm]        # Qwen3ASRModel (vllm 后端, 仅 Linux/WSL2)
    pip install spacy                   # pipeline_segale_align_new 句子分割
    pip install transformers            # pipeline_segale_align_new 加载 embedding 模型
    pip install sentence-transformers   # LaBSE 等 embedding 模型
    pip install tqdm
    pip install sacrebleu               # BLEU 计算
    pip install pyyaml                  # 读 yaml
    pip install nagisa                  # pipeline_qwen3_forcealign_tokenizer2 日语分词
    pip install matplotlib              # pipeline_plot_ending_offset_delay 画图
    '''
    # ================================================================
    # 配置信息
    model_name = "seed"
    input_version = ""
    output_version = ""
    src_lang = "en"
    tgt_lang = "ja"
    # ASR 后端: "transformers" 用 pipeline_qwen3_asr2.step1_asr；"vllm" 用 pipeline_qwen3_asr2vllm.step1_asr_vllm（需 pip install qwen-asr[vllm]）
    asr_backend = "transformers"
    lang_map = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "ja": "Japanese",
        # 其他语言映射...
    }
    bleu_map = {
        "en": "13a",
        "zh": "zh",
        "de": "13a",
        "ja": "ja-mecab",
        # 其他语言映射...
    }

    # src yaml, src txt, ref txt的信息 --------------------------------
    src_segments_yaml = f"data2/input/ACL.ACLdev2023.{src_lang}-xx.gold_segments.yaml"
    src_txt = f"data2/input/ACL.6060.dev.{src_lang}-xx.{src_lang}.txt"
    tgt_ref_txt = f"data2/input/ACL.6060.dev.{src_lang}-xx.{tgt_lang}.txt"

    # step1 asr的信息--------------------------------------------------
    output_dir_asr = f"data3/{model_name}/{src_lang}_{tgt_lang}/output_asr{output_version}"
    batch_size = 3
    max_new_tokens = 1024
    tgt_language = lang_map.get(tgt_lang, tgt_lang)
    manifest = f"data3/{model_name}/{src_lang}_{tgt_lang}/output_tgt{input_version}/manifest.jsonl"

    # 只是我给 asr 的 unit 计算索引的路径，不想污染原来的 asr 文件
    output_dir_asr_ = output_dir_asr + "_"

    # 生成 instances.log 的路径
    output_path_instances = os.path.join(output_dir_asr_, "instances.log")

    # step2 segale的信息--------------------------------------------------
    output_dir_segale = f"data3/{model_name}/{src_lang}_{tgt_lang}/output_segale{output_version}"

    # step3 longyaal的信息--------------------------------------------------
    segale_file = os.path.join(output_dir_segale, "hyp/aligned_spacy_hyp.jsonl")  # 获得 segale 的 aligned_spacy_hyp.jsonl 文件
    output_dir_longyaal = f"data3/{model_name}/{src_lang}_{tgt_lang}/output_longyaal{output_version}"

    # ================================================================
    # 都会生成文件的，想单独跑某step：注释其他steps就好

    # # step1 asr -------------------------------------------------------
    # print("\n" + "=" * 60)
    # print("Starting ASR...")
    # if asr_backend == "vllm":
    #     step1_asr_vllm(
    #         manifest=manifest,
    #         tgt_language=tgt_language,
    #         out_dir=output_dir_asr,
    #         batch_size=batch_size,
    #         max_new_tokens=max_new_tokens,
    #         gpu_memory_utilization=0.7,
    #     )
    # else:
    #     step1_asr(
    #         manifest=manifest,
    #         tgt_language=tgt_language,
    #         out_dir=output_dir_asr,
    #         batch_size=batch_size,
    #         max_new_tokens=max_new_tokens,
    #     )
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # print("ASR finished.")

    # # ---------------------------------------------------------------
    # # 中间过程，算 asr 的 unit 索引 
    # add_char_spans_for_dir(
    #     output_dir_asr,
    #     output_dir_asr_
    # )
    # # 中间过程，生成 instances.log
    # asr_to_instances(
    #     s2s=True,
    #     yaml_file=src_segments_yaml,
    #     asr_dir=output_dir_asr_,
    #     output_file=output_path_instances,
    # )
    # # 中间过程，生成 segale 需要的 hyp.jsonl 和 ref.jsonl
    # instances_to_segale(
    #     src_txt=src_txt, 
    #     tgt_ref_txt=tgt_ref_txt, 
    #     src_segments_yaml=src_segments_yaml, 
    #     instances=output_path_instances,
    #     out_dir=output_dir_segale
    # )

    # # step2 segale ---------------------------------------------------------------
    # # (如果用 softsegmenter 的话，从下面开始直接去跑 pipeline_softsegmenter_longyaal2.py 就行)
    # print("\n" + "=" * 60)
    # print("Starting Segale...")
    # step2_segale(
    #     system_file=os.path.join(output_dir_segale, "hyp.jsonl"),
    #     ref_file=os.path.join(output_dir_segale, "ref.jsonl"),
    #     segmenter="spacy",
    #     task_lang=tgt_lang,
    #     proc_device="cuda",
    #     embedding_model= "sentence-transformers/LaBSE"  # "BAAI/bge-m3"
    # )
    # print("Segale finished.")

    # # step3 longyaal ---------------------------------------------------------------
    # print("\n" + "=" * 60)
    # print("Starting Longyaal...")
    # step3_longyaal(
    #     yaml_file=src_segments_yaml,
    #     source_sentences_file=src_txt,
    #     instances_log=output_path_instances,
    #     segale_file=segale_file,
    #     output_folder=output_dir_longyaal,
    #     bleu_tokenizer=bleu_map.get(tgt_lang, "13a")
    # )
    # print("Longyaal finished.")

    # # ending offset delay ---------------------------------------------------------------
    ending_offset_delay(
        instances=os.path.join(output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(output_dir_longyaal, "last_delay_minus_recording_end.png"),
        out_csv=os.path.join(output_dir_longyaal, "last_delay_minus_recording_end.csv"),
    )
    delay_span_vs_source_length(
        instances=os.path.join(output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(output_dir_longyaal, "delay_span_vs_source_length.png")
    )
    tgt_minus_src_length_histogram(
        instances=os.path.join(output_dir_longyaal, "instances.resegmented.json"),
        out_png=os.path.join(output_dir_longyaal, "tgt_minus_src_length_histogram.png"),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        only_doc_ids=("110", "117"),  # 仅某几个 acl-long 音频；不要则注释掉或传 None
    )

if __name__ == "__main__":
    main()