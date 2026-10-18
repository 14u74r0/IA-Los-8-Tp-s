"""Trabajo 3 - GAN que genera imágenes aleatorias (PyTorch + CUDA).

Incluye un benchmark CPU vs GPU para mostrar el poder de los núcleos CUDA y
los Tensor Cores de NVIDIA. Es el trabajo pensado para la presentación.

Ejecutar desde la raíz del repo:  python run.py t3_gan
"""
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, Response

from core import device, jobs, web

try:
    from . import gan

    TORCH_ERROR = None
except ImportError:  # PyTorch no instalado
    gan = None
    TORCH_ERROR = "PyTorch no está instalado. Con el entorno activado: pip install -r requirements-torch.txt"

app = FastAPI(title="T3 - GAN")
templates = web.make_templates(Path(__file__).parent)
job = jobs.BackgroundJob()  # entrenamiento
bench = jobs.BackgroundJob()  # benchmark


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"error": TORCH_ERROR, "datasets": gan.DATASETS if gan else {}}
    )


@app.get("/api/estado")
def estado():
    if TORCH_ERROR:
        return {"error": TORCH_ERROR}
    _, version = gan.muestra_actual()
    return {
        "job": job.snapshot(),
        "bench": bench.snapshot(),
        "modelos": gan.info_modelos(),
        "muestra_version": version,
        "torch": device.torch_info(),
        "gpu": device.gpu_stats(),
    }


def _ocupado() -> JSONResponse | None:
    if job.running or bench.running:
        return JSONResponse({"error": "Hay un proceso en ejecución. Esperá a que termine."}, status_code=409)
    return None


@app.post("/api/entrenar")
def entrenar(
    dataset: str = Form("mnist"),
    epocas: int = Form(10),
    batch: int = Form(128),
    dispositivo: str = Form("auto"),
    amp: bool = Form(False),
):
    if TORCH_ERROR:
        return JSONResponse({"error": TORCH_ERROR}, status_code=503)
    if dataset not in gan.DATASETS or dispositivo not in ("auto", "cpu", "cuda"):
        return JSONResponse({"error": "Parámetros no válidos."}, status_code=422)
    if not (1 <= epocas <= 200 and 16 <= batch <= 4096):
        return JSONResponse({"error": "Revisá épocas (1-200) y batch (16-4096)."}, status_code=422)
    if (busy := _ocupado()) is not None:
        return busy
    job.start(gan.entrenar, dataset=dataset, epocas=epocas, batch=batch, dispositivo=dispositivo, amp=amp)
    return {"ok": True}


@app.post("/api/detener")
def detener():
    job.stop()
    bench.stop()
    return {"ok": True}


@app.get("/api/muestra.png")
def muestra():
    if TORCH_ERROR:
        return Response(status_code=404)
    png, _ = gan.muestra_actual()
    if png is None:
        return Response(status_code=404)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/generar")
def generar(dataset: str = Form("mnist"), n: int = Form(16), semilla: str = Form("")):
    if TORCH_ERROR:
        return JSONResponse({"error": TORCH_ERROR}, status_code=503)
    if dataset not in gan.DATASETS or not (1 <= n <= 64):
        return JSONResponse({"error": "Parámetros no válidos (1 a 64 imágenes)."}, status_code=422)
    try:
        sem = int(semilla) if semilla.strip() else None
    except ValueError:
        return JSONResponse({"error": "La semilla debe ser un número entero."}, status_code=422)
    try:
        png, ms, disp = gan.generar(dataset, n, sem)
    except gan.DatosError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return Response(png, media_type="image/png", headers={"X-Ms": f"{ms:.1f}", "X-Dispositivo": disp})


@app.post("/api/benchmark")
def benchmark(pasos: int = Form(100), batch: int = Form(256)):
    if TORCH_ERROR:
        return JSONResponse({"error": TORCH_ERROR}, status_code=503)
    if not (10 <= pasos <= 2000 and 16 <= batch <= 4096):
        return JSONResponse({"error": "Revisá pasos (10-2000) y batch (16-4096)."}, status_code=422)
    if (busy := _ocupado()) is not None:
        return busy
    bench.start(gan.benchmark, pasos=pasos, batch=batch)
    return {"ok": True}
