if (-not (Test-Path -LiteralPath ".venv")) {
    python -m venv .venv
}

$python = Join-Path ".venv" "Scripts/python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "Setup complete. Activate the environment with: .venv\Scripts\Activate.ps1"