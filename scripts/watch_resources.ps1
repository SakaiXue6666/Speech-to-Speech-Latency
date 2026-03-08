# 新开一个终端，在项目根目录执行:  .\scripts\watch_resources.ps1
# 或只刷一次:  nvidia-smi; Get-Process python -ErrorAction SilentlyContinue | Select Id, CPU, @{N='MemMB';E={[math]::Round($_.WS/1MB,1)}}
# 每 3 秒刷新：GPU (nvidia-smi) + 本机 Python 进程的 CPU/内存
$interval = 3
while ($true) {
    Clear-Host
    Write-Host "=== $(Get-Date -Format 'HH:mm:ss') === (Ctrl+C 退出)" -ForegroundColor Cyan
    Write-Host ""
    # GPU（若有 nvidia-smi）
    try {
        $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpu) { Write-Host "GPU: util%, mem_used_MB, mem_total_MB"; Write-Host $gpu }
    } catch {}
    Write-Host ""
    # Python 进程 CPU、内存
    Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
        $memMB = [math]::Round($_.WorkingSet64/1MB, 1)
        Write-Host "PID $($_.Id)  CPU: $([math]::Round($_.CPU,1))s  内存: ${memMB} MB"
    }
    if (-not (Get-Process -Name python -ErrorAction SilentlyContinue)) {
        Write-Host "当前没有 python 进程"
    }
    Start-Sleep -Seconds $interval
}
