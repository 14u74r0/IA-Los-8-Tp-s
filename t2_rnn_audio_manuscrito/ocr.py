"""Lectura de texto manuscrito: TrOCR (Transformer, GPU) o un modelo de visión de Ollama."""
import threading
import time

import numpy as np
from PIL import Image

from core import config, hf, images, ollama_client

_lock = threading.Lock()
_modelo = None  # (processor, model, device)


def _cargar():
    global _modelo
    with _lock:
        if _modelo is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = TrOCRProcessor.from_pretrained(config.TROCR_MODEL)
            model = VisionEncoderDecoderModel.from_pretrained(config.TROCR_MODEL)
            if device == "cuda":
                model = model.half()
            model.to(device).eval()
            _modelo = (processor, model, device)
        return _modelo


def segmentar_lineas(img: Image.Image) -> list[Image.Image]:
    """Separa una hoja en renglones con un perfil horizontal de tinta (heurística simple).

    TrOCR lee una línea por vez. Funciona bien con letra oscura sobre fondo claro;
    para páginas difíciles conviene usar el motor de visión.
    """
    gris = np.asarray(img.convert("L"), dtype=np.float32)
    fondo = np.percentile(gris, 90)
    oscuro = np.percentile(gris, 2)
    if fondo - oscuro < 30:  # casi sin contraste
        return [img.convert("RGB")]
    tinta = gris < fondo - 0.35 * (fondo - oscuro)

    perfil = tinta.sum(axis=1).astype(np.float32)
    perfil = np.convolve(perfil, np.ones(5) / 5, mode="same")
    activo = perfil > 0.12 * perfil.max()

    bandas, inicio = [], None
    for y, a in enumerate(activo):
        if a and inicio is None:
            inicio = y
        elif not a and inicio is not None:
            bandas.append([inicio, y])
            inicio = None
    if inicio is not None:
        bandas.append([inicio, len(activo)])
    if not bandas:
        return [img.convert("RGB")]

    # Unir bandas muy pegadas (tildes, letras con rabito) y descartar ruido.
    alturas = [b[1] - b[0] for b in bandas]
    tipica = float(np.median(alturas))
    unidas = [bandas[0]]
    for b in bandas[1:]:
        if b[0] - unidas[-1][1] < 0.4 * tipica:
            unidas[-1][1] = b[1]
        else:
            unidas.append(b)
    unidas = [b for b in unidas if (b[1] - b[0]) >= 0.35 * tipica]

    lineas = []
    alto, ancho = gris.shape
    for y0, y1 in unidas:
        margen = int(0.25 * (y1 - y0)) + 2
        ya, yb = max(0, y0 - margen), min(alto, y1 + margen)
        cols = np.where(tinta[y0:y1].any(axis=0))[0]
        xa = max(0, int(cols.min()) - 20) if len(cols) else 0
        xb = min(ancho, int(cols.max()) + 20) if len(cols) else ancho
        lineas.append(img.convert("RGB").crop((xa, ya, xb, yb)))
    return lineas or [img.convert("RGB")]


def leer_trocr(data: bytes) -> dict:
    import torch

    processor, model, device = _cargar()
    img = images.abrir(data)
    lineas = segmentar_lineas(img)

    t0 = time.perf_counter()
    textos = []
    for i in range(0, len(lineas), 8):  # lotes de 8 renglones
        lote = lineas[i : i + 8]
        pixeles = processor(images=lote, return_tensors="pt").pixel_values.to(device, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(pixeles, max_new_tokens=128)
        textos += processor.batch_decode(ids, skip_special_tokens=True)
    return {
        "texto": "\n".join(t.strip() for t in textos if t.strip()),
        "lineas": len(lineas),
        "segundos": round(time.perf_counter() - t0, 2),
        "motor": f"TrOCR ({config.TROCR_MODEL}) en {device}",
    }


def leer_vision(data: bytes) -> dict:
    t0 = time.perf_counter()
    b64 = images.a_base64_jpeg(images.abrir(data), max_lado=1600)
    texto = ollama_client.chat(
        "Transcribí exactamente el texto manuscrito de esta imagen, respetando los renglones. "
        "Respondé solo con el texto transcripto, sin comentarios.",
        model=config.OLLAMA_VISION_MODEL,
        images=[b64],
        temperature=0.0,
    ).strip()
    return {
        "texto": texto,
        "lineas": len([l for l in texto.splitlines() if l.strip()]),
        "segundos": round(time.perf_counter() - t0, 2),
        "motor": f"Modelo de visión de Ollama ({config.OLLAMA_VISION_MODEL})",
    }
