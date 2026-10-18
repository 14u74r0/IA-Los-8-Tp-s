"""Utilidades de imágenes compartidas."""
import base64
import io

from PIL import Image, ImageOps


def abrir(data: bytes) -> Image.Image:
    """Abre una imagen respetando la rotación EXIF (fotos de celular)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img


def a_base64_jpeg(img: Image.Image, max_lado: int = 1024) -> str:
    """Reduce y codifica en base64 para enviar a un modelo de visión de Ollama."""
    img = img.convert("RGB")
    img.thumbnail((max_lado, max_lado))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()
