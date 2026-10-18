"""Carga de audio a un arreglo mono de 16 kHz, sin instalar ffmpeg a mano.

Usa el ffmpeg que trae el paquete `imageio-ffmpeg`, así funciona igual en
Windows y en Linux con wav, mp3, m4a, ogg, flac, webm, etc.
"""
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    """Error legible para mostrar al usuario."""


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE):
    try:
        import imageio_ffmpeg
        import numpy as np
    except ImportError as e:
        raise AudioError("Faltan librerías de audio. Ejecutá: pip install -r requirements-ml.txt") from e

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-i", str(path),
        "-vn",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True)
    except OSError as e:
        raise AudioError(f"No se pudo ejecutar ffmpeg: {e}") from e
    if out.returncode != 0 or not out.stdout:
        raise AudioError("No se pudo leer el audio. ¿Es un archivo de audio válido?")
    return np.frombuffer(out.stdout, dtype=np.float32)
