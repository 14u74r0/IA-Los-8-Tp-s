# Setup para Windows (PowerShell):  .\setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Se creó .env - ajustá OLLAMA_MODEL para esta máquina."
}

Write-Host ""
Write-Host "PyTorch con CUDA (cuando lo necesites): .\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt"
Write-Host ""
.\.venv\Scripts\python.exe scripts\check_env.py
