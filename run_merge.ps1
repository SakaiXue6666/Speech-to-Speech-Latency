param(
  [string]$InputJson = "input_template.json",
  [string]$OutputJson = "merge_output.json",
  [string]$AsrModel = "Qwen/Qwen3-ASR-1.7B",
  [string]$AlignerModel = "Qwen/Qwen3-ForcedAligner-0.6B",
  [string]$DeviceMap = "cuda:0",
  [string]$DType = "bfloat16",
  [string]$Language = "",
  [string]$Segmenter = "spacy",
  [string]$TaskLang = "zh",
  [string]$EmbeddingModel = "sentence-transformers/LaBSE",
  [string]$ProcDevice = "cpu",
  [int]$MaxSize = 8,
  [switch]$InstallDeps,
  [switch]$WriteTemplateOnly
)

$ErrorActionPreference = "Stop"

if ($InstallDeps) {
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  python -m pip install -r LongYAAL/requirements.txt
}

if ($WriteTemplateOnly) {
  python merge.py --write_template $InputJson
  Write-Host "Template generated at $InputJson"
  exit 0
}

$langArg = @()
if ($Language -ne "") {
  $langArg = @("--language", $Language)
}

python merge.py `
  --input_json $InputJson `
  --output_json $OutputJson `
  --asr_model $AsrModel `
  --forced_aligner $AlignerModel `
  --device_map $DeviceMap `
  --dtype $DType `
  --segmenter $Segmenter `
  --task_lang $TaskLang `
  --embedding_model $EmbeddingModel `
  --proc_device $ProcDevice `
  --max_size $MaxSize `
  @langArg
