# Speech-to-Speech Latency

Evaluation pipeline for measuring latency in speech-to-speech translation systems.

## Setup (Linux + GPU)

```bash
# 1. Clone the repository
cd <base_path>
git clone https://github.com/SakaiXue6666/Speech-to-Speech-Latency.git
cd Speech-to-Speech-Latency

# 2. Create conda environment
conda create -n s2s_latency python=3.10 -y
conda activate s2s_latency

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install GPU version of PyTorch
# Run nvidia-smi to check your CUDA version first:
#   CUDA 12.4 → cu124
#   CUDA 12.8 → cu128 (required for RTX 50 series / Blackwell)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# 5. Build vecalign Cython extension
cd SEGALE/vecalign
python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(ext_modules=cythonize('dp_core.pyx'), include_dirs=[numpy.get_include()], script_args=['build_ext', '--inplace'])" || true
# --inplace may report a path error; manually copy the .so from build/
cp build/lib.*/SEGALE/vecalign/dp_core.*.so .
cd ../..

# 6. Install SEGALE
pip install --no-deps -e ./SEGALE

# 7. spaCy language models (install as needed)
python -m spacy download zh_core_web_sm   # Chinese
python -m spacy download de_core_news_sm  # German
pip install "ginza==5.2.0" "ja-ginza==5.2.0" "confection==0.1.5"  # Japanese (confection 1.x has breaking changes)

# 8. Download HuggingFace model weights (required on first run, cached offline afterwards)
huggingface-cli download Qwen/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B
huggingface-cli download sentence-transformers/LaBSE
```

## Usage

```bash
python main.py \
  --src-lang en \
  --tgt-lang ja \
  --src-audio-dir input/acl_6060_dev/full_wavs \
  --tgt-audio-dir input/acl_6060_dev_tgt_seed/en_ja \
  --src-segments-yaml input/ACL.ACLdev2023.en-xx.gold_segments.yaml \
  --src-txt input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt \
  --tgt-ref-txt input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt \
  --output-dir output/seed_en_ja
```

All arguments have default values, so `python main.py` with no arguments runs the English→Japanese evaluation using the bundled sample data.

| Argument | Default | Description |
|----------|---------|-------------|
| `--src-lang` | `en` | Source language code |
| `--tgt-lang` | `ja` | Target language code (`zh` / `de` / `ja`) |
| `--src-audio-dir` | `input/acl_6060_dev/full_wavs` | Source audio WAV directory |
| `--tgt-audio-dir` | `input/acl_6060_dev_tgt_seed/en_ja` | Target (translated) audio WAV directory |
| `--src-segments-yaml` | `input/ACL.ACLdev2023.en-xx.gold_segments.yaml` | Source audio segmentation file |
| `--src-txt` | `input/.../ACL.6060.dev.en-xx.en.txt` | Source language transcript |
| `--tgt-ref-txt` | `input/.../ACL.6060.dev.en-xx.ja.txt` | Target language reference translation |
| `--output-dir` | `output` | Output directory for all results |

After the pipeline finishes, generate plots:

```bash
python plot/ending_offset_delay.py
python plot/tgt_minus_src_length_histogram.py
```

## Output directory structure

```
{output_dir}/
├── output_asr/          # ASR results
├── output_segmentation/ # Segmentation results
└── output_evaluation/   # Evaluation results + plots
```
