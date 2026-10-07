param([ValidateSet('All','MatMul','LayerNorm','Softmax')][string]$Operator='All')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'build/benchmark_targets.exe'
if (!(Test-Path -LiteralPath $taskExe)) { throw 'Run build.bat first.' }
$taskOutput=Join-Path $taskRoot 'results/profiles'
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskCases=@(
    @{Name='MatMul';Kernel='kernel_matmul_shape';Args=@('--matmul','128','768','768')},
    @{Name='LayerNorm';Kernel='kernel_layernorm_resident';Args=@('--layernorm','128','768')},
    @{Name='Softmax';Kernel='kernel_softmax_shape';Args=@('--softmax','12288','1024')}
)
foreach($taskCase in $taskCases) {
    if ($Operator -ne 'All' -and $taskCase.Name -ne $Operator) { continue }
    $taskReport=Join-Path $taskOutput $taskCase.Name.ToLower()
    $taskCsv=Join-Path $taskOutput ($taskCase.Name.ToLower()+'.instrumented.csv')
    $taskArgs=@('--set','full','--launch-count','1','--kernel-name',('regex:'+ $taskCase.Kernel),
                '--export',$taskReport,'--force-overwrite',$taskExe,
                '--trials','1','--samples','1','--csv',$taskCsv)+$taskCase.Args
    & ncu @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Nsight failed. For ERR_NVGPUCTRPERM, enable NVIDIA performance-counter access.' }
}
Write-Host 'Reports saved under results/profiles. Instrumented timings are not benchmark evidence.'
