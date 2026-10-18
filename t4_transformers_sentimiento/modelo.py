"""Clasificador de sentimientos basado en un Transformer (XLM-RoBERTa) de Hugging Face.

Corre en la GPU con CUDA si PyTorch la ve; si no, en CPU.
"""
import time
from functools import lru_cache

from core import config

ETIQUETAS = ("positivo", "neutro", "negativo")


class ModeloError(RuntimeError):
    """Error legible para mostrar al usuario."""


def normalizar_etiqueta(label: str) -> str:
    """Convierte las etiquetas del modelo (negative/POS/LABEL_2...) a positivo/neutro/negativo."""
    l = label.strip().lower()
    if l.startswith("pos"):
        return "positivo"
    if l.startswith("neg"):
        return "negativo"
    if l.startswith("neu"):
        return "neutro"
    if l.startswith("label_"):
        try:
            return ("negativo", "neutro", "positivo")[int(l.split("_")[1])]
        except (ValueError, IndexError):
            pass
    return l


@lru_cache(maxsize=1)
def _cargar():
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ModeloError(
            "Faltan librerías. Con el entorno activado ejecutá: "
            "pip install -r requirements-torch.txt  y luego  pip install -r requirements-ml.txt"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = pipeline(
            "text-classification",
            model=config.SENTIMENT_MODEL,
            device=0 if device == "cuda" else -1,
        )
    except OSError as e:
        raise ModeloError(
            f"No se pudo cargar el modelo '{config.SENTIMENT_MODEL}'. "
            "La primera vez necesita internet para descargarlo (~1 GB)."
        ) from e
    return pipe, device


def predecir(textos: list[str]) -> dict:
    """Clasifica una lista de textos. Devuelve resultados, dispositivo usado y tiempo."""
    pipe, device = _cargar()
    t0 = time.perf_counter()
    salida = pipe(textos, top_k=None, truncation=True, max_length=512, batch_size=16)
    ms = (time.perf_counter() - t0) * 1000

    resultados = []
    for texto, scores in zip(textos, salida):
        por_etiqueta = {normalizar_etiqueta(s["label"]): float(s["score"]) for s in scores}
        etiqueta = max(por_etiqueta, key=por_etiqueta.get)
        resultados.append(
            {
                "texto": texto,
                "etiqueta": etiqueta,
                "confianza": por_etiqueta[etiqueta],
                "scores": {k: por_etiqueta.get(k, 0.0) for k in ETIQUETAS},
            }
        )
    return {"resultados": resultados, "device": device, "ms": ms, "modelo": config.SENTIMENT_MODEL}
