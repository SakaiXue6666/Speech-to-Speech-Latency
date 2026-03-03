# 本机安装 LASER 并设置 LASER_DIR

按 SEGALE 官方 Dockerfile 的方式，在 Windows 本机安装 LASER。

## 1. 选一个安装目录

例如：`D:\Li_Lab\LASER`（不要放在有空格或中文的路径下）。

## 2. 克隆 LASER 仓库

在 PowerShell 里执行：

```powershell
cd D:\Li_Lab
git clone https://github.com/facebookresearch/LASER
cd LASER
```

## 3. 用 pip 安装 LASER（可编辑模式）

```powershell
pip install -e .
```

（会装 fairseq、sentencepiece、sacremoses 等依赖。）

## 4. 下载编码器模型（二选一）

LASER 需要语言模型才能算句子向量。**任选一种**：

### 方式 A：用 NLLB 脚本（需 Git Bash 或 WSL）

在 **Git Bash** 里（在 `D:\Li_Lab\LASER` 下）：

```bash
bash ./nllb/download_models.sh eng_Latn
```

如需中文可再跑：`bash ./nllb/download_models.sh zho_Hans`  
更多语言见：<https://github.com/facebookresearch/LASER/tree/master/nllb>

### 方式 B：用 laser_encoders（不依赖完整 LASER 仓库）

若本机只装 `laser_encoders`、不克隆 LASER 仓库，可跳过 2–4 步，在代码里用 `--embedding_model` 指定 HuggingFace 模型，不设 LASER_DIR。

若已克隆 LASER 并希望用它的模型，则完成 2–4 步后，`LaserEncoderPipeline(model_dir=LASER_DIR, ...)` 会从该目录读模型。

## 5. 改 SEGALE 里的 LASER_DIR

编辑 **`SEGALE/segale_align.py`**，约第 37 行，把 `LASER_DIR` 改成你本机的 LASER 路径（用正斜杠或双反斜杠）：

```python
LASER_DIR = "D:/Li_Lab/LASER"
```

保存后，不传 `--embedding_model` 时，segale-align 会使用该目录下的 LASER。

## 6. 验证

不指定 `--embedding_model` 跑一次对齐，例如：

```powershell
cd D:\Li_Lab\Speech-to-Speech-Latency
segale-align --system_file data/segale/hyp.jsonl --ref_file data/segale/ref.jsonl --segmenter spacy --task_lang zh --proc_device cuda -v
```

若报错找不到模型，检查 LASER 目录下是否有 `nllb`、`tasks/embed` 等，以及 `laser_encoders` 是否期望的目录结构（可查 laser_encoders 文档）。

---

**说明**：官方推荐用 Docker，本机安装在 Windows 上可能遇到脚本（bash）、路径或依赖问题；若仅做对齐，用 `--embedding_model sentence-transformers/LaBSE` 通常更省事。
