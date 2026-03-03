# 用 Docker 跑 SEGALE（镜像内已装 LASER，无需本机安装）
# 前置：已安装 Docker Desktop，并启用 GPU（NVIDIA Container Toolkit）。
# 在项目根目录执行： .\scripts\docker_segale_laser.ps1

$ProjectRoot = "D:\Li_Lab\Speech-to-Speech-Latency"
$ImageName = "segale-laser"

# ---------- 1. 构建镜像（仅首次或 Dockerfile 变更时需要，较耗时） ----------
Set-Location $ProjectRoot
docker build -t $ImageName -f SEGALE/Dockerfile .

# ---------- 2. 运行容器：挂载项目目录，在容器内安装 SEGALE 并跑 segale-align ----------
# 镜像默认只带 ace_Latn；task_lang zh 需 zho_Hans，在容器内先下载（已有则跳过）。
# 若不用 GPU：去掉 --gpus all，并把下面 --proc_device 改为 cpu。
$Mount = "${ProjectRoot}:/workspace"
$Cmd = @(
    "pip install -e ./SEGALE",
    "bash /opt/LASER/nllb/download_models.sh zho_Hans || true",
    "segale-align --system_file data/segale/hyp.jsonl --ref_file data/segale/ref.jsonl --segmenter spacy --task_lang zh --proc_device cuda -v"
) -join " && "
docker run --gpus all -v $Mount -w /workspace $ImageName bash -c $Cmd

# 结果在宿主机的 data/segale/hyp/aligned_spacy_hyp.jsonl（因 /workspace 挂载的是项目根）
