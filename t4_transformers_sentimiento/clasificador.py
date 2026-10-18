"""Clasificador de sentimientos con un Transformer (XLM-RoBERTa) de Hugging Face.

El modelo se carga una sola vez y corre en la GPU con CUDA si PyTorch la ve.
"""
import threading
import time

from core import config, hf

_lock = threading.Lock()
_pipe = None
_error: str | None = None

ETIQUETAS = ("positivo", "neutro", "negativo")


def _get_pipe():
    global _pipe
    with _lock:
        if _pipe is None:
            from transformers import pipeline

            _pipe = pipeline(
                "text-classification",
                model=config.SENTIMENT_MODEL,
                device=hf.pipeline_device(),
                top_k=None,  # devuelve la probabilidad de todas las clases
            )
        return _pipe


def precargar() -> None:
    """Carga el modelo en segundo plano para que la primera consulta no espere."""
    global _error
    try:
        _get_pipe()
    except Exception as e:  # sin internet la primera vez, modelo mal escrito, etc.
        _error = f"No se pudo cargar el modelo {config.SENTIMENT_MODEL}: {e}"


def estado() -> dict:
    return {
        "listo": _pipe is not None,
        "error": _error,
        "modelo": config.SENTIMENT_MODEL,
        "dispositivo": "cuda" if hf.cuda_disponible() else "cpu",
    }


def _normalizar(etiqueta: str) -> str:
    e = etiqueta.lower()
    if "pos" in e or e in ("label_2", "5 stars", "4 stars"):
        return "positivo"
    if "neg" in e or e in ("label_0", "1 star", "2 stars"):
        return "negativo"
    return "neutro"


def predecir(textos: list[str]) -> dict:
    """Devuelve {'resultados': [...], 'segundos', 'dispositivo', 'modelo'}."""
    pipe = _get_pipe()
    t0 = time.perf_counter()
    crudo = pipe(textos, batch_size=16, truncation=True, max_length=512)
    segundos = time.perf_counter() - t0

    resultados = []
    for texto, scores in zip(textos, crudo):
        probs = {e: 0.0 for e in ETIQUETAS}
        for s in scores:
            probs[_normalizar(s["label"])] += float(s["score"])
        mejor = max(probs, key=probs.get)
        resultados.append(
            {
                "texto": texto,
                "sentimiento": mejor,
                "confianza": round(probs[mejor], 4),
                "probabilidades": {k: round(v, 4) for k, v in probs.items()},
            }
        )
    return {
        "resultados": resultados,
        "segundos": round(segundos, 3),
        "dispositivo": "cuda" if hf.cuda_disponible() else "cpu",
        "modelo": config.SENTIMENT_MODEL,
    }
