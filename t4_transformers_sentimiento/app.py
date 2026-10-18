"""Trabajo 4 - Análisis de sentimientos con Transformers.

Ejecutar desde la raíz del repo:  python run.py t4_transformers_sentimiento
"""
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from core import web

from . import clasificador

MAX_TEXTOS = 500

app = FastAPI(title="T4 - Sentimientos con Transformers")
templates = web.make_templates(Path(__file__).parent)
threading.Thread(target=clasificador.precargar, daemon=True).start()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"estado": clasificador.estado()})


@app.get("/api/estado")
def estado():
    return clasificador.estado()


@app.post("/api/predecir")
def predecir(texto: str = Form(""), archivo: UploadFile | None = File(None)):
    """Un texto por línea, escrito en el cuadro o cargado desde un .txt / .csv."""
    contenido = texto
    if archivo is not None and archivo.filename:
        contenido += "\n" + archivo.file.read().decode("utf-8", errors="replace")
    textos = [l.strip() for l in contenido.splitlines() if l.strip()][:MAX_TEXTOS]
    if not textos:
        return JSONResponse({"error": "Escribí al menos un texto o cargá un archivo."}, status_code=422)
    try:
        return clasificador.predecir(textos)
    except Exception as e:
        return JSONResponse({"error": f"No se pudo analizar: {e}"}, status_code=500)
