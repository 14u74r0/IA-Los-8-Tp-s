"""Trabajo 6 - Procesamiento de sonidos.

Audio de un comentario -> Whisper (transcripción, GPU) -> sentimiento (Llama
local con Ollama) -> fila en la tabla `comentarios` (origen = 'audio').

Ejecutar desde la raíz del repo:  python run.py t6_audio_comentarios
"""
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from core import comentarios, web, whisper_asr
from core.ollama_client import OllamaError
from core.sentimiento import analizar
from core.whisper_asr import AudioError

ORIGEN = "audio"

app = FastAPI(title="T6 - Comentarios en audio")
templates = web.make_templates(Path(__file__).parent)
comentarios.init()
threading.Thread(target=whisper_asr.precargar, daemon=True).start()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"estado": whisper_asr.estado()})


@app.get("/api/estado")
def estado():
    return whisper_asr.estado()


@app.get("/api/comentarios")
def listar():
    return {"filas": comentarios.listar(ORIGEN), "stats": comentarios.resumen(ORIGEN)}


@app.post("/api/analizar-audio")
def analizar_audio(
    producto: str = Form(...),
    cliente: str = Form(...),
    audio: UploadFile = File(...),
):
    producto, cliente = producto.strip(), cliente.strip()
    if not (producto and cliente):
        return JSONResponse({"error": "Completá producto y cliente."}, status_code=422)

    ruta = web.guardar_subida(audio, "audios")
    try:
        trans = whisper_asr.transcribir(str(ruta))
    except AudioError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": f"No se pudo transcribir el audio: {e}"}, status_code=500)

    if not trans["texto"]:
        return JSONResponse({"error": "No se detectó voz en el audio."}, status_code=422)

    try:
        resultado = analizar(trans["texto"])
    except OllamaError as e:
        return JSONResponse({"error": f"Audio transcripto, pero falló el análisis: {e}"}, status_code=503)

    id_ = comentarios.guardar(producto, cliente, trans["texto"], resultado, ORIGEN, archivo=ruta.name)
    return {"id": id_, "transcripcion": trans, "analisis": resultado}


@app.post("/api/borrar/{id_}")
def borrar(id_: int):
    comentarios.borrar(id_)
    return {"ok": True}
