"""Transcripción de audio con Whisper (transformer) sobre PyTorch/CUDA.

Compartido por el trabajo 2 y el 6. Los formatos wav/flac/ogg/mp3 se leen sin
ffmpeg; m4a, webm (grabación del navegador) y otros necesitan ffmpeg instalado.
"""
import threading
import time

from . import config, hf

_lock = threading.Lock()
_pipe = None


class AudioError(RuntimeError):
    """Error legible para mostrar en la interfaz."""


def _get_pipe():
    global _pipe
    with _lock:
        if _pipe is None:
            from transformers import pipeline

            _pipe = pipeline(
                "automatic-speech-recognition",
                model=config.WHISPER_MODEL,
                device=hf.pipeline_device(),
                chunk_length_s=30,
                **hf.dtype_kwarg(hf.dtype_preferido()),
            )
        return _pipe


_error: str | None = None


def precargar() -> None:
    """Carga Whisper en segundo plano (la primera vez descarga el modelo)."""
    global _error
    try:
        _get_pipe()
    except Exception as e:
        _error = f"No se pudo cargar {config.WHISPER_MODEL}: {e}"


def estado() -> dict:
    return {
        "listo": _pipe is not None,
        "error": _error,
        "modelo": config.WHISPER_MODEL,
        "dispositivo": "cuda" if hf.cuda_disponible() else "cpu",
    }


def _entrada(path: str):
    """Decodifica el audio a 16 kHz mono con el ffmpeg que trae `imageio-ffmpeg`.

    Así no hace falta instalar ffmpeg a mano en Windows ni en Linux. Si el
    paquete no está, se le pasa la ruta al pipeline (necesita ffmpeg en el PATH).
    """
    try:
        from . import audio

        return {"raw": audio.load_audio(path), "sampling_rate": audio.SAMPLE_RATE}
    except audio.AudioError as e:
        if "librerías" in str(e):  # imageio-ffmpeg no instalado
            return path
        raise AudioError(str(e)) from e


def transcribir(path: str, idioma: str | None = None) -> dict:
    """Devuelve {'texto', 'segundos', 'modelo', 'dispositivo'}."""
    pipe = _get_pipe()
    gen = {"task": "transcribe"}
    idioma = idioma or config.WHISPER_LANGUAGE
    if idioma and idioma != "auto":
        gen["language"] = idioma

    t0 = time.perf_counter()
    try:
        out = pipe(_entrada(path), generate_kwargs=gen, return_timestamps=True, batch_size=8)
    except ValueError as e:
        if "ffmpeg" in str(e).lower():
            raise AudioError(
                "Para este formato de audio hace falta ffmpeg. Instalalo con: winget install Gyan.FFmpeg "
                "(Windows) o sudo dnf install ffmpeg (Nobara) y reiniciá la terminal."
            ) from e
        raise
    return {
        "texto": out["text"].strip(),
        "segundos": round(time.perf_counter() - t0, 2),
        "modelo": config.WHISPER_MODEL,
        "dispositivo": "cuda" if hf.cuda_disponible() else "cpu",
    }
