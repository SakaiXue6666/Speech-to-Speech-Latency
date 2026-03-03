# 1. 进入项目根目录
cd D:\Li_Lab\Speech-to-Speech-Latency

# 2. 克隆 LASER 到项目根下
git clone https://github.com/facebookresearch/LASER LASER

# 3. 安装 LASER
cd LASER
pip install -e .
cd ..

# 4. 安装 laser_encoders（segale 用 LASER 时依赖）
pip install laser_encoders

# 5. 设置 LASER 路径（用绝对路径）
$env:LASER_DIR = "D:\Li_Lab\Speech-to-Speech-Latency\LASER"
$env:PYTHONPATH = "D:\Li_Lab\Speech-to-Speech-Latency;D:\Li_Lab\Speech-to-Speech-Latency\SEGALE"

# 6. 下载 NLLB 模型（必须，否则 LASER 无法编码）
# 尝试用 Git Bash 执行；若未安装 Git for Windows，请先安装或改用 WSL
$laserPath = "D:\Li_Lab\Speech-to-Speech-Latency\LASER"
$bashExe = "C:\Program Files\Git\bin\bash.exe"
if (Test-Path $bashExe) {
    & $bashExe -c "cd '$($laserPath -replace '\\','/')' && bash ./nllb/download_models.sh eng_Latn && bash ./nllb/download_models.sh zho_Hans"
} else {
    Write-Host "未找到 Git Bash。请手动在 Git Bash 中执行：" -ForegroundColor Yellow
    Write-Host "  cd D:/Li_Lab/Speech-to-Speech-Latency/LASER"
    Write-Host "  bash ./nllb/download_models.sh eng_Latn"
    Write-Host "  bash ./nllb/download_models.sh zho_Hans"
    Read-Host "下载完成后按回车继续"
}

# 7. 回到项目根再跑 segale（用 LASER，不传 --embedding_model）
cd D:\Li_Lab\Speech-to-Speech-Latency
python segale_align.py --system_file data/segale/hyp.jsonl --ref_file data/segale/ref.jsonl --segmenter spacy --task_lang zh --proc_device cuda -v