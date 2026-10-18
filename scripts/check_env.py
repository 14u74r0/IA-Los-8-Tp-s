"""Diagnóstico rápido del entorno: python scripts/check_env.py"""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core import config, device, ollama_client  # noqa: E402


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[!!] {msg}")


print(f"Python {platform.python_version()} en {platform.system()} {platform.release()}")

gpu = device.gpu_info()
if gpu["disponible"]:
    ok(f"GPU: {gpu['nombre']} ({gpu['vram_mb']} MB) - driver {gpu['driver']}")
else:
    warn("No se encontró nvidia-smi: revisá el driver de NVIDIA")

try:
    import torch

    ok(f"PyTorch {torch.__version__} (CUDA de la build: {torch.version.cuda})")
    if torch.cuda.is_available():
        try:
            x = torch.randn(1024, 1024, device="cuda")
            float((x @ x).sum().item())
            ok(f"CUDA funciona: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            warn(f"CUDA visible pero falló una operación (¿build de PyTorch muy vieja para tu GPU?): {e}")
    else:
        warn("PyTorch no ve CUDA: instalá con  pip install -r requirements-torch.txt")
except ImportError:
    warn("PyTorch no está instalado (solo hace falta para t1, t2, t3, t5): pip install -r requirements-torch.txt")

for modulo, para in (
    ("torchvision", "t1, t3, t5"),
    ("transformers", "t2, t4, t6"),
    ("imageio_ffmpeg", "t2, t6 (leer audio)"),
    ("pymupdf", "t8"),
):
    try:
        __import__(modulo)
        ok(f"{modulo} instalado")
    except ImportError:
        warn(f"Falta {modulo} (para {para}): pip install -r requirements-torch.txt  y  pip install -r requirements-ml.txt")

if ollama_client.is_up():
    modelos = ollama_client.list_models()
    ok(f"Ollama activo en {config.OLLAMA_HOST} - modelos: {', '.join(modelos) or 'ninguno'}")
    if config.OLLAMA_MODEL not in modelos:
        warn(f"Falta el modelo de texto del .env: ollama pull {config.OLLAMA_MODEL}")
    if config.OLLAMA_VISION_MODEL not in modelos:
        warn(f"Falta el modelo de visión (t2 opcional, t5): ollama pull {config.OLLAMA_VISION_MODEL}")
else:
    warn(f"Ollama no responde en {config.OLLAMA_HOST} (instalalo desde ollama.com y abrilo)")

print(f"Modelo de texto: {config.OLLAMA_MODEL} | visión: {config.OLLAMA_VISION_MODEL}")
