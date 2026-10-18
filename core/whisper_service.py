"""Transcripción de audio con Whisper (Hugging Face + PyTorch, usa CUDA si hay).

El modelo se descarga solo la primera vez (whisper-small pesa ~1 GB) y queda en
memoria mientras la app esté corriendo.
"""
from functools import lru_cache

from . import audio, config


class WhisperError(RuntimeError):
    """Error legible para mostrar al usuario."""


@lru_cache(maxsize=1)
def _asr():
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise WhisperError(
            "Faltan librerías. Con el entorno activado ejecutá: "
            "pip install -r requirements-torch.txt  y luego  pip install -r requirements-ml.txt"
        ) from e

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL,
            device=device,
            chunk_length_s=30,
        )
    except OSError as e:
        raise WhisperError(
            f"No se pudo cargar el modelo '{config.WHISPER_MODEL}'. "
            "La primera vez necesita internet para descargarlo."
        ) from e
    return asr, device


def transcribir(path, idioma: str = "spanish") -> dict:
    """Devuelve {'texto', 'device'} para el archivo de audio indicado."""
    try:
        muestras = audio.load_audio(path)
    except audio.AudioError as e:
        raise WhisperError(str(e)) from e
    asr, device = _asr()
    salida = asr(
        {"raw": muestras, "sampling_rate": audio.SAMPLE_RATE},
        generate_kwargs={"language": idioma, "task": "transcribe"},
    )
    return {"texto": salida["text"].strip(), "device": device}
