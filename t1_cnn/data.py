"""Datos del trabajo 1: CIFAR-10 (se descarga solo) o un dataset propio por carpetas.

Dataset propio:  data/t1_dataset/<clase>/*.jpg  (una carpeta por clase).
Todo se carga a memoria como tensores uint8 para entrenar rápido en la GPU.
"""
import io
import re
import shutil
import time

import numpy as np
import torch
from PIL import Image

from core import config

CIFAR_CLASES = ("avión", "auto", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión")
DATASET_DIR = config.DATA_DIR / "t1_dataset"
TORCHVISION_DIR = config.DATA_DIR / "torchvision"
TAM_PROPIO = 64
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatosError(ValueError):
    """Problema con los datos, con un mensaje legible."""


# ---------- dataset propio ----------

def _nombre_clase(nombre: str) -> str:
    limpio = re.sub(r"[^\w\- ]+", "", nombre, flags=re.UNICODE).strip().replace(" ", "_")
    if not limpio:
        raise DatosError("Poné un nombre de clase (ej: gato, perro).")
    return limpio[:40]


def clases_propias() -> dict[str, int]:
    if not DATASET_DIR.exists():
        return {}
    return {
        d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in EXTENSIONES)
        for d in sorted(DATASET_DIR.iterdir())
        if d.is_dir()
    }


def guardar_imagenes(clase: str, archivos: list[tuple[str, bytes]]) -> tuple[int, list[str]]:
    """Guarda imágenes (reducidas a 256 px máx.) en la carpeta de la clase. Devuelve (guardadas, rechazadas)."""
    carpeta = DATASET_DIR / _nombre_clase(clase)
    carpeta.mkdir(parents=True, exist_ok=True)
    guardadas, rechazadas = 0, []
    for i, (nombre, contenido) in enumerate(archivos):
        try:
            img = Image.open(io.BytesIO(contenido))
            img.load()
            img = img.convert("RGB")
            img.thumbnail((256, 256))
        except Exception:
            rechazadas.append(nombre)
            continue
        img.save(carpeta / f"{int(time.time() * 1000)}_{i}.jpg", quality=92)
        guardadas += 1
    return guardadas, rechazadas


def borrar_clase(clase: str) -> None:
    carpeta = DATASET_DIR / _nombre_clase(clase)
    if carpeta.is_dir() and carpeta.parent == DATASET_DIR:
        shutil.rmtree(carpeta)


def _cargar_propio():
    clases = [c for c, n in clases_propias().items() if n > 0]
    if len(clases) < 2:
        raise DatosError("Para entrenar con tu dataset cargá imágenes de al menos 2 clases.")
    xs, ys = [], []
    for idx, clase in enumerate(clases):
        for f in sorted((DATASET_DIR / clase).iterdir()):
            if f.suffix.lower() not in EXTENSIONES:
                continue
            try:
                img = Image.open(f).convert("RGB").resize((TAM_PROPIO, TAM_PROPIO))
            except Exception:
                continue
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(idx)
    if len(xs) < 10:
        raise DatosError(f"Hacen falta al menos 10 imágenes en total (hay {len(xs)}). Idealmente 30 o más por clase.")
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ys)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y), generator=g)
    n_test = max(2, int(0.2 * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    return x[tr], y[tr], x[te], y[te], clases, TAM_PROPIO, "mi dataset"


# ---------- CIFAR-10 ----------

def _cargar_cifar():
    from torchvision.datasets import CIFAR10

    TORCHVISION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tr = CIFAR10(str(TORCHVISION_DIR), train=True, download=True)
        te = CIFAR10(str(TORCHVISION_DIR), train=False, download=True)
    except Exception as e:
        raise DatosError(f"No se pudo descargar CIFAR-10 (¿hay internet?): {e}") from e

    def a_tensor(ds):
        return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)

    xtr, ytr = a_tensor(tr)
    xte, yte = a_tensor(te)
    return xtr, ytr, xte, yte, list(CIFAR_CLASES), 32, "CIFAR-10"


def cargar(nombre: str):
    """Devuelve (x_train, y_train, x_test, y_test, clases, tamaño, nombre) con x en uint8 NCHW."""
    if nombre == "cifar10":
        return _cargar_cifar()
    if nombre == "propio":
        return _cargar_propio()
    raise DatosError(f"Dataset desconocido: {nombre}")
