param(
    [ValidateSet('Check','Test','Visualize','Backend','Frontend','Build','Preview','Research')]
    [string]$Task = 'Check',
    [string]$RunId = '',
    [string]$Python = 'python',
    [switch]$Full
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
# Prefer the interpreter already selected by the terminal, e.g. an activated venv.
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program gagal (exit $LASTEXITCODE). Proses dihentikan." }
}
switch ($Task) {
    'Check' {
        Invoke-Checked $Python @('-c', 'from shared.deployment_integrity import load_verified_pipeline, verify_selection_contract; verify_selection_contract(); p,m=load_verified_pipeline(); print(''Verified model:'',m[''model_version''])')
    }
    'Test' {
        $pytestArgs = @('-m','pytest','-q')
        if (!$Full) { $pytestArgs += @('-m','not slow') }
        Invoke-Checked $Python $pytestArgs
        Push-Location frontend
        try { Invoke-Checked npm.cmd @('test') } finally { Pop-Location }
    }
    'Visualize' { Invoke-Checked $Python @('scripts/visualize_verified_results.py') }
    'Backend' { Invoke-Checked $Python @('-m','flask','--app','backend.app','run','--host','127.0.0.1','--port','5001') }
    'Frontend' {
        Push-Location frontend
        try { Invoke-Checked npm.cmd @('run','dev','--','--host','127.0.0.1','--strictPort') } finally { Pop-Location }
    }
    'Build' {
        Push-Location frontend
        try { Invoke-Checked npm.cmd @('run','build') } finally { Pop-Location }
    }
    'Preview' {
        Push-Location frontend
        try { Invoke-Checked npm.cmd @('run','preview','--','--host','127.0.0.1','--strictPort') } finally { Pop-Location }
    }
    'Research' {
        if (!$RunId) { $RunId = Get-Date -Format 'yyyyMMdd_HHmmss' }
        if ($RunId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunId hanya boleh huruf, angka, underscore atau tanda minus.' }
        $run = "runs/$RunId"
        $preparation = "$run/preparation"
        $selected = "$run/selected"
        $researchOutput = "$run/output"
        if (Test-Path -LiteralPath $run) {
            throw 'Namespace sudah ada. Gunakan RunId baru; hasil sebelumnya dipertahankan.'
        }
        Invoke-Checked $Python @('scripts/check_environment.py')
        Invoke-Checked $Python @('scripts/prepare_dataset_snapshot.py','--output',$preparation)
        Invoke-Checked $Python @('scripts/run_research_experiments.py','--preparation',$preparation,'--selected',$selected,'--output',$researchOutput)
        Invoke-Checked $Python @('scripts/select_research_configuration.py','--output',$researchOutput)
        Write-Host "Selesai: $researchOutput/experiments/selected_configuration.json"
        Write-Host 'Review konfigurasi sebelum locking/training; deployment aktif tidak berubah.'
    }
}
