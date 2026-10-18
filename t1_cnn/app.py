"""Trabajo 1 - CNN para clasificar imágenes (PyTorch + CUDA).

Entrena una red convolucional con CIFAR-10 o con tus propias imágenes, y
predice la clase de una imagen que cargues.

Ejecutar desde la raíz del repo:  python run.py t1_cnn
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from core import device, jobs, web

try:
    from . import data, trainer

    TORCH_ERROR = None
except ImportError:  # PyTorch no instalado
    data = trainer = None
    TORCH_ERROR = "PyTorch no está instalado. Con el entorno activado: pip install -r requirements-torch.txt"

app = FastAPI(title="T1 - CNN")
templates = web.make_templates(Path(__file__).parent)
job = jobs.BackgroundJob()


def _sin_torch():
    return JSONResponse({"error": TORCH_ERROR}, status_code=503)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"error": TORCH_ERROR})


@app.get("/api/estado")
def estado():
    if TORCH_ERROR:
        return {"error": TORCH_ERROR}
    return {
        "job": job.snapshot(),
        "modelo": trainer.info_modelo(),
        "clases_propias": data.clases_propias(),
        "torch": device.torch_info(),
        "gpu": device.gpu_stats(),
    }


@app.post("/api/entrenar")
def entrenar(
    dataset: str = Form("cifar10"),
    epocas: int = Form(10),
    batch: int = Form(256),
    lr: float = Form(0.003),
    dispositivo: str = Form("auto"),
    amp: bool = Form(False),
):
    if TORCH_ERROR:
        return _sin_torch()
    if dataset not in ("cifar10", "propio") or dispositivo not in ("auto", "cpu", "cuda"):
        return JSONResponse({"error": "Parámetros no válidos."}, status_code=422)
    if not (1 <= epocas <= 200 and 8 <= batch <= 4096 and 0 < lr <= 1):
        return JSONResponse({"error": "Revisá épocas (1-200), batch (8-4096) y learning rate."}, status_code=422)
    try:
        job.start(
            trainer.entrenar, dataset=dataset, epocas=epocas, batch=batch, lr=lr, dispositivo=dispositivo, amp=amp
        )
    except jobs.JobBusy as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return {"ok": True}


@app.post("/api/detener")
def detener():
    job.stop()
    return {"ok": True}


@app.post("/api/subir")
def subir(clase: str = Form(...), imagenes: list[UploadFile] = File(...)):
    if TORCH_ERROR:
        return _sin_torch()
    archivos = [(f.filename or "imagen", f.file.read()) for f in imagenes]
    try:
        guardadas, rechazadas = data.guardar_imagenes(clase, archivos)
    except data.DatosError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return {"guardadas": guardadas, "rechazadas": rechazadas, "clases": data.clases_propias()}


@app.post("/api/borrar-clase")
def borrar_clase(clase: str = Form(...)):
    if TORCH_ERROR:
        return _sin_torch()
    data.borrar_clase(clase)
    return {"clases": data.clases_propias()}


@app.post("/api/predecir")
def predecir(imagen: UploadFile = File(...)):
    if TORCH_ERROR:
        return _sin_torch()
    try:
        return trainer.predecir(imagen.file.read())
    except data.DatosError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": f"No se pudo procesar la imagen: {e}"}, status_code=500)
