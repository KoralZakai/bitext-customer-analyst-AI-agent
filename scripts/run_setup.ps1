# Run all 3 setup steps: preprocess (download+clean), profile, verify
# Usage (PowerShell):
#   cd C:\Users\kali-linux\Projects\bitext-customer-analyst-agent
#   $env:HF_TOKEN = "hf_your_token"   # optional but recommended
#   .\scripts\run_setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Missing .venv. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 1/3: Download + preprocess ===" -ForegroundColor Cyan
& $Python -m src.data.preprocess
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 2/3: Quality profile ===" -ForegroundColor Cyan
& $Python scripts\profile_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 3/3: Verify dataset ===" -ForegroundColor Cyan
& $Python main.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nSetup complete. Start agent with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python main.py --session demo" -ForegroundColor Green
