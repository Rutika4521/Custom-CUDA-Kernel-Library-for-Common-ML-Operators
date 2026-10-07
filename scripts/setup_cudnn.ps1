$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskDependency = Join-Path $taskRoot 'third_party'
New-Item -ItemType Directory -Path $taskDependency -Force | Out-Null
$taskBase = 'https://developer.download.nvidia.com/compute/cudnn/redist/'
$taskManifest = Invoke-RestMethod -Uri ($taskBase + 'redistrib_9.17.1.json')
$taskPackage = $taskManifest.cudnn.'windows-x86_64'.cuda13
$taskArchive = Join-Path $taskDependency 'cudnn-9.17.1-cuda13.zip'
if (!(Test-Path -LiteralPath $taskArchive) -or (Get-Item -LiteralPath $taskArchive).Length -ne [long]$taskPackage.size) {
    Write-Host 'Downloading NVIDIA cuDNN 9.17.1 for CUDA 13 (342 MB).'
    Invoke-WebRequest -Uri ($taskBase + $taskPackage.relative_path) -OutFile $taskArchive -UseBasicParsing
}
$taskHash = (Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash
if ($taskHash -ne $taskPackage.sha256) { throw 'cuDNN SHA-256 checksum mismatch.' }
Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskDependency -Force
Write-Host 'cuDNN installed locally. Run build.bat to enable the vendor comparisons.'
