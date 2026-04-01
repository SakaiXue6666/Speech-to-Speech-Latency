# Speech-to-Speech Latency

Evaluation pipeline for measuring latency in speech-to-speech translation systems.

## Setup (AutoDL / Linux + GPU)

```bash
# AutoDL 加速网络（AutoDL 平台专用，其他环境跳过）
source /etc/network_turbo

# 1. 拉代码
cd ~/autodl-tmp
git clone https://github.com/SakaiXue6666/Speech-to-Speech-Latency.git
cd Speech-to-Speech-Latency

# 2. 创建环境
conda create -n s2s_latency python=3.10 -y
conda activate s2s_latency

# 3. 装依赖
pip install -r requirements.txt

# 4. 装 GPU 版 PyTorch
# 先运行 nvidia-smi 查看 CUDA 版本，按需选择：
#   CUDA 12.4 → cu124
#   CUDA 12.8 → cu128（RTX 50 系 Blackwell 必须用 cu128）
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# 5. 编译 vecalign Cython 扩展
cd SEGALE/vecalign
python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(ext_modules=cythonize('dp_core.pyx'), include_dirs=[numpy.get_include()], script_args=['build_ext', '--inplace'])" || true
# --inplace 会报路径错误，直接从 build/ 手动拷贝 .so
cp build/lib.*/SEGALE/vecalign/dp_core.*.so .
cd ../..

# 6. 安装 SEGALE
pip install --no-deps -e ./SEGALE

# 7. spaCy 语言模型（按需安装）
python -m spacy download zh_core_web_sm   # 中文
python -m spacy download de_core_news_sm  # 德文
pip install "ginza==5.2.0" "ja-ginza==5.2.0" "confection==0.1.5"  # 日文（confection 1.x 有 breaking change）

# 8. 下载 HuggingFace 模型权重（首次运行需要，之后自动离线）
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
huggingface-cli download Qwen/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B
huggingface-cli download sentence-transformers/LaBSE
```

## Usage

编辑 `config.py` 修改语言对、模型名称等参数，然后：

```bash
# 跑完整 pipeline（ASR → 对齐 → 评估）
python main.py

# 画图
python plot/ending_offset_delay.py
python plot/tgt_minus_src_length_histogram.py
```

## Input directory structure

```
input/
├── ACL.ACLdev2023.en-xx.gold_segments.yaml
├── acl_6060_dev/
│   ├── full_wavs/          # src 音频
│   └── text/txt/           # src/tgt 参考文本
└── acl_6060_dev_tgt_{model_name}/
    └── {src_lang}_{tgt_lang}/   # tgt 音频（按系统名分目录）
```

## Output directory structure

```
output/
└── {model_name}/
    └── {src_lang}_{tgt_lang}/
        ├── output_asr/          # ASR 结果
        ├── output_segale/       # 对齐结果
        └── output_evaluation/   # 评估结果 + 图
```
