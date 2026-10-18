"""Trabajo 5 - Procesamiento de imágenes: fotos de animales a una base de datos.

Cada foto pasa por dos modelos:
  - un clasificador de PyTorch (EfficientNet, usa CUDA) que dice qué animal es,
  - un modelo de visión de Ollama que describe color, tamaño, hábitat y dieta.
El análisis se guarda en la tabla `animales`.

Ejecutar desde la raíz del repo:  python run.py t5_imagenes_animales
"""
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from core import config, images, web
from core.ollama_client import OllamaError

from . import clasificador, descriptor, tabla

app = FastAPI(title="T5 - Animales")
templates = web.make_templates(Path(__file__).parent)
tabla.init()
SUBCARPETA = "animales"


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"dispositivo": clasificador.nombre_dispositivo()})


@app.get("/api/animales")
def listar():
    return tabla.listar()


@app.get("/imagen/{id_}")
def imagen(id_: int):
    fila = tabla.obtener(id_)
    ruta = config.UPLOADS_DIR / SUBCARPETA / fila["archivo"] if fila else None
    if not ruta or not ruta.exists():
        return JSONResponse({"error": "Imagen no encontrada."}, status_code=404)
    return FileResponse(ruta)


@app.post("/api/analizar")
def analizar(foto: UploadFile = File(...)):
    """Analiza UNA foto (la interfaz las manda de a una para mostrar el avance)."""
    nombre_original = foto.filename or "foto"
    ruta = web.guardar_subida(foto, SUBCARPETA)
    try:
        img = images.abrir(ruta.read_bytes()).convert("RGB")
    except Exception:
        ruta.unlink(missing_ok=True)
        return JSONResponse({"error": f"«{nombre_original}» no es una imagen válida."}, status_code=422)

    notas, clasif, desc = [], None, None
    try:
        clasif = clasificador.clasificar(img)
    except Exception as e:  # PyTorch sin instalar, sin internet para bajar los pesos, etc.
        notas.append(f"Clasificador PyTorch no disponible: {e}")
    try:
        desc = descriptor.describir(img)
    except OllamaError as e:
        notas.append(f"Modelo de visión no disponible: {e}")

    if clasif is None and desc is None:
        ruta.unlink(missing_ok=True)
        return JSONResponse({"error": " | ".join(notas)}, status_code=503)

    top = clasif["top"] if clasif else []
    es_animal = desc["es_animal"] if desc else clasif["prob_animal"] >= 0.3
    if clasif and clasif["prob_animal"] < 0.3 and desc is None:
        notas.append("El clasificador no ve un animal claro en la foto.")
    fila = {
        "archivo": ruta.name,
        "nombre_original": nombre_original,
        "especie": (desc["especie"] if desc and desc["especie"] not in ("", "desconocido", "-") else None)
        or (top[0]["categoria"] if top else None),
        "categoria_imagenet": top[0]["categoria"] if top else None,
        "confianza": top[0]["prob"] if top else None,
        "top3": top,
        "color": desc["color"] if desc else None,
        "tamano": desc["tamano"] if desc else None,
        "habitat": desc["habitat"] if desc else None,
        "dieta": desc["dieta"] if desc else None,
        "descripcion": desc["descripcion"] if desc else None,
        "es_animal": 1 if es_animal else 0,
        "notas": " | ".join(notas) or None,
    }
    fila["id"] = tabla.guardar(fila)
    return fila


@app.post("/api/borrar/{id_}")
def borrar(id_: int):
    fila = tabla.obtener(id_)
    if fila:
        (config.UPLOADS_DIR / SUBCARPETA / fila["archivo"]).unlink(missing_ok=True)
        tabla.borrar(id_)
    return {"ok": True}
