"""Trabajo 7 - Procesamiento de textos.

Recibe comentarios de productos (escritos o desde un .txt), los analiza con un
LLM local vía Ollama y guarda el resultado en la tabla `comentarios`.

Ejecutar desde la raíz del repo:  python run.py t7_texto
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core import comentarios, config, device, ollama_client
from core.ollama_client import OllamaError
from core.sentimiento import analizar

ORIGEN = "texto"
MAX_LINEAS = 100

app = FastAPI(title="T7 - Procesamiento de textos")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
comentarios.init()


def _contexto(error: str | None = None, ok: str | None = None) -> dict:
    return {
        "filas": comentarios.listar(ORIGEN),
        "stats": comentarios.resumen(ORIGEN),
        "modelo": config.OLLAMA_MODEL,
        "ollama_activo": ollama_client.is_up(),
        "gpu": device.gpu_info(),
        "error": error,
        "ok": ok,
    }


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _contexto())


@app.post("/analizar")
def analizar_texto(
    request: Request,
    producto: str = Form(...),
    cliente: str = Form(...),
    comentario: str = Form(...),
):
    producto, cliente, comentario = producto.strip(), cliente.strip(), comentario.strip()
    if not (producto and cliente and comentario):
        return templates.TemplateResponse(
            request, "index.html", _contexto(error="Completá producto, cliente y comentario."), status_code=422
        )
    try:
        resultado = analizar(comentario)
    except OllamaError as e:
        return templates.TemplateResponse(request, "index.html", _contexto(error=str(e)), status_code=503)
    comentarios.guardar(producto, cliente, comentario, resultado, ORIGEN)
    return RedirectResponse("/", status_code=303)


@app.post("/analizar-archivo")
def analizar_archivo(
    request: Request,
    producto: str = Form(...),
    cliente: str = Form(...),
    archivo: UploadFile = File(...),
):
    """Un comentario por línea; todos se asocian al producto y cliente indicados."""
    texto = archivo.file.read().decode("utf-8", errors="replace")
    lineas = [l.strip() for l in texto.splitlines() if l.strip()][:MAX_LINEAS]
    if not lineas:
        return templates.TemplateResponse(
            request, "index.html", _contexto(error="El archivo está vacío."), status_code=422
        )
    guardados = 0
    try:
        for linea in lineas:
            comentarios.guardar(producto.strip(), cliente.strip(), linea, analizar(linea), ORIGEN)
            guardados += 1
    except OllamaError as e:
        msg = f"Se guardaron {guardados} de {len(lineas)} comentarios. {e}"
        return templates.TemplateResponse(request, "index.html", _contexto(error=msg), status_code=503)
    return templates.TemplateResponse(
        request, "index.html", _contexto(ok=f"Se analizaron y guardaron {guardados} comentarios.")
    )


@app.post("/borrar/{id_}")
def borrar(id_: int):
    comentarios.borrar(id_)
    return RedirectResponse("/", status_code=303)


@app.get("/api/comentarios")
def api_comentarios():
    return JSONResponse(comentarios.listar(ORIGEN))
