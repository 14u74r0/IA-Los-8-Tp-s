"""Trabajo 2 - Audio y texto manuscrito a texto, con traducción de idiomas.

Audio -> Whisper (GPU). Imagen manuscrita -> TrOCR (GPU) o modelo de visión de
Ollama. La traducción la hace Llama local (Ollama).

Ejecutar desde la raíz del repo:  python run.py t2_rnn_audio_manuscrito
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from core import web, whisper_asr
from core.ollama_client import OllamaError
from core.whisper_asr import AudioError

from . import ocr
from .traductor import DESTINOS, IDIOMAS_AUDIO, traducir

app = FastAPI(title="T2 - Audio y manuscrito")
templates = web.make_templates(Path(__file__).parent)


def _con_traduccion(resultado: dict, destino: str) -> dict:
    if destino and destino != "ninguno" and resultado["texto"]:
        try:
            resultado["traduccion"] = traducir(resultado["texto"], destino)
            resultado["destino"] = destino
        except OllamaError as e:
            resultado["traduccion_error"] = str(e)
    return resultado


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"destinos": list(DESTINOS), "idiomas_audio": IDIOMAS_AUDIO},
    )


@app.post("/api/audio")
def procesar_audio(
    audio: UploadFile = File(...),
    idioma: str = Form("auto"),
    destino: str = Form("ninguno"),
):
    if idioma not in IDIOMAS_AUDIO:
        return JSONResponse({"error": "Idioma de audio no válido."}, status_code=422)
    ruta = web.guardar_subida(audio, "audios")
    try:
        r = whisper_asr.transcribir(str(ruta), idioma=idioma)
    except AudioError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": f"No se pudo transcribir el audio: {e}"}, status_code=500)
    r["motor"] = f"Whisper ({r['modelo']}) en {r['dispositivo']}"
    return _con_traduccion(r, destino)


@app.post("/api/manuscrito")
def procesar_manuscrito(
    imagen: UploadFile = File(...),
    motor: str = Form("trocr"),
    destino: str = Form("ninguno"),
):
    data = imagen.file.read()
    try:
        if motor == "vision":
            r = ocr.leer_vision(data)
        else:
            r = ocr.leer_trocr(data)
    except OllamaError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": f"No se pudo leer la imagen: {e}"}, status_code=500)
    if not r["texto"]:
        return JSONResponse({"error": "No se reconoció texto en la imagen."}, status_code=422)
    return _con_traduccion(r, destino)


@app.post("/api/traducir")
def traducir_texto(texto: str = Form(...), destino: str = Form(...)):
    try:
        return {"traduccion": traducir(texto, destino)}
    except (OllamaError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=503)
