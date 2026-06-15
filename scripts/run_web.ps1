$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$EnvName = if ($env:ENV_NAME) { $env:ENV_NAME } else { "RV" }
$ServerHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$Port = if ($env:PORT) { $env:PORT } else { "7860" }

# 当前脚本所在目录的上一级目录
$RootDir = Split-Path -Parent $PSScriptRoot

# 将 Conda 初始化到当前 PowerShell 会话
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression

conda activate $EnvName
Set-Location $RootDir

python -m exp5.web_server `
    --host $ServerHost `
    --port $Port `
    @args