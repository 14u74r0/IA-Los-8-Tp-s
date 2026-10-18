"""Trabajo 8 - Procesamiento de PDF: texto -> resumen con Llama local -> tabla `pdfs`.

Ejecutar desde la raíz del repo:  python run.py t8_pdf
"""
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from core import web
from core.ollama_client import OllamaError

from . import procesador, tabla

MAX_MB = 50

app = FastAPI(title="T8 - PDF")
templates = web.make_templates(Path(__file__).parent)
tabla.init()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/pdfs")
def listar():
    return tabla.listar()


@app.post("/api/procesar")
def procesar(pdf: UploadFile = File(...)):
    """Procesa UN PDF (la interfaz los manda de a uno para mostrar el avance)."""
    nombre = pdf.filename or "documento.pdf"
    ruta = web.guardar_subida(pdf, "pdfs")
    if ruta.stat().st_size > MAX_MB * 1024 * 1024:
        ruta.unlink(missing_ok=True)
        return JSONResponse({"error": f"El PDF supera los {MAX_MB} MB."}, status_code=413)
    try:
        texto, paginas = procesador.extraer_texto(ruta)
        r = procesador.resumir(texto, nombre)
    except procesador.PDFError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except OllamaError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    id_ = tabla.guardar(nombre, r["resumen"], paginas, len(texto), r["parcial"], r["modelo"])
    return {"id": id_, "nombre_archivo": nombre, "paginas": paginas, "caracteres": len(texto), **r}


@app.post("/api/borrar/{id_}")
def borrar(id_: int):
    tabla.borrar(id_)
    return {"ok": True}
