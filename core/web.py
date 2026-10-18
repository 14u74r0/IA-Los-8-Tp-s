"""Plantillas Jinja compartidas: cada trabajo extiende core/templates/base.html."""
import re
import time
from pathlib import Path

from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, FileSystemLoader

from . import config, device, ollama_client

_CORE_TEMPLATES = Path(__file__).parent / "templates"


def make_templates(project_dir: str | Path) -> Jinja2Templates:
    local = Path(project_dir) / "templates"
    templates = Jinja2Templates(directory=str(local))
    templates.env.loader = ChoiceLoader(
        [FileSystemLoader(str(local)), FileSystemLoader(str(_CORE_TEMPLATES))]
    )
    templates.env.globals["gpu_info"] = device.gpu_info
    templates.env.globals["ollama_up"] = ollama_client.is_up
    templates.env.globals["ollama_model"] = config.OLLAMA_MODEL
    templates.env.globals["ollama_vision_model"] = config.OLLAMA_VISION_MODEL
    return templates


def guardar_subida(archivo, subcarpeta: str = "") -> Path:
    """Guarda un UploadFile en data/uploads/ con un nombre seguro y devuelve la ruta."""
    destino = config.UPLOADS_DIR / subcarpeta
    destino.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(archivo.filename or "archivo").name) or "archivo"
    ruta = destino / f"{int(time.time() * 1000)}_{base}"
    with open(ruta, "wb") as f:
        while chunk := archivo.file.read(1024 * 1024):
            f.write(chunk)
    return ruta
