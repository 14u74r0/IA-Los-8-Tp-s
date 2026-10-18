"""Extracción de texto de PDFs (PyMuPDF) y resumen con Llama local (Ollama).

Los documentos largos no entran en la ventana de contexto del modelo, así que
se resumen por fragmentos y luego se resume el conjunto de resúmenes (map-reduce).
"""
from core import config, ollama_client

CHUNK_CHARS = 5000
MAX_CHUNKS = 12


class PDFError(ValueError):
    """Problema con el PDF, con un mensaje legible."""


def extraer_texto(ruta) -> tuple[str, int]:
    """Devuelve (texto, cantidad de páginas)."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # nombre viejo del paquete
        except ImportError as e:
            raise PDFError("Falta PyMuPDF. Con el entorno activado: pip install -r requirements-ml.txt") from e
    try:
        doc = pymupdf.open(str(ruta))
    except Exception as e:
        raise PDFError("No se pudo abrir el archivo. ¿Es un PDF válido?") from e
    with doc:
        if doc.needs_pass:
            raise PDFError("El PDF está protegido con contraseña.")
        paginas = doc.page_count
        texto = "\n\n".join(p.get_text("text") for p in doc)
    texto = "\n".join(l.rstrip() for l in texto.splitlines())
    while "\n\n\n" in texto:
        texto = texto.replace("\n\n\n", "\n\n")
    return texto.strip(), paginas


def trocear(texto: str, tam: int = CHUNK_CHARS) -> list[str]:
    """Parte el texto en fragmentos de hasta `tam` caracteres, cortando en párrafos u oraciones."""
    chunks, actual = [], ""
    for parr in texto.split("\n\n"):
        while len(parr) > tam:  # párrafo gigante: cortar en el último espacio
            corte = parr.rfind(" ", 0, tam)
            corte = corte if corte > tam // 2 else tam
            if actual:
                chunks.append(actual)
                actual = ""
            chunks.append(parr[:corte])
            parr = parr[corte:].lstrip()
        if len(actual) + len(parr) + 2 > tam and actual:
            chunks.append(actual)
            actual = parr
        else:
            actual = f"{actual}\n\n{parr}" if actual else parr
    if actual.strip():
        chunks.append(actual)
    return [c for c in chunks if c.strip()]


def _muestrear(chunks: list[str], maximo: int) -> list[str]:
    """Si hay demasiados fragmentos, toma `maximo` repartidos de forma pareja (incluye inicio y final)."""
    if len(chunks) <= maximo:
        return chunks
    pasos = [round(i * (len(chunks) - 1) / (maximo - 1)) for i in range(maximo)]
    return [chunks[i] for i in pasos]


SYSTEM = "Sos un asistente que resume documentos con precisión, en español, sin inventar datos."


def _resumen_parcial(fragmento: str) -> str:
    return ollama_client.chat(
        "Resumí en español, en 3 a 5 oraciones, los puntos principales de este fragmento de un documento. "
        "Respondé solo con el resumen.\n\n" + fragmento,
        system=SYSTEM,
        num_ctx=4096,
    ).strip()


def resumir(texto: str, nombre: str = "") -> dict:
    """Devuelve {'resumen', 'fragmentos', 'parcial'} (parcial=True si el documento era demasiado largo)."""
    if len(texto.strip()) < 40:
        raise PDFError("El PDF no tiene texto seleccionable (¿es un documento escaneado? Haría falta OCR).")
    chunks = trocear(texto)
    parcial = len(chunks) > MAX_CHUNKS
    usados = _muestrear(chunks, MAX_CHUNKS)

    if len(usados) == 1:
        material, instruccion = usados[0], "Resumí en español, en 5 a 8 oraciones, el siguiente documento."
    else:
        material = "\n\n".join(f"[Parte {i + 1}] {_resumen_parcial(c)}" for i, c in enumerate(usados))
        instruccion = (
            "A continuación hay resúmenes parciales, en orden, de un mismo documento. "
            "Escribí un único resumen final en español, de 5 a 8 oraciones, coherente y sin repeticiones."
        )
    resumen = ollama_client.chat(
        f"{instruccion} Respondé solo con el resumen.\n\n{material}", system=SYSTEM, num_ctx=8192
    ).strip()
    return {"resumen": resumen, "fragmentos": len(usados), "parcial": parcial, "modelo": config.OLLAMA_MODEL}
