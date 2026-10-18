#!/usr/bin/env bash
# Setup para Linux (Nobara/Fedora):  bash setup.sh
set -euo pipefail

[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Se creó .env - ajustá OLLAMA_MODEL para esta máquina."
fi

echo
echo "PyTorch con CUDA (cuando lo necesites): .venv/bin/python -m pip install -r requirements-torch.txt"
echo
.venv/bin/python scripts/check_env.py
