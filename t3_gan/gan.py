"""GAN convolucional (DCGAN) del trabajo 3: genera imágenes nuevas a partir de ruido aleatorio.

Dos redes compiten: el Generador crea imágenes falsas desde un vector de ruido y
el Discriminador intenta distinguirlas de las reales. Todo corre en PyTorch/CUDA.
"""
import io
import math
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from core import config, device as dev

LATENT = 100
MUESTRAS = 64  # imágenes de la cuadrícula que se muestra mientras entrena
DATASETS = {
    "mnist": "Dígitos escritos a mano (MNIST)",
    "fashion": "Prendas de ropa (Fashion-MNIST)",
}
DATA_DIR = config.DATA_DIR / "torchvision"


class DatosError(ValueError):
    """Problema con los datos o el modelo, con un mensaje legible."""


def checkpoint(nombre: str):
    return config.WEIGHTS_DIR / f"t3_gan_{nombre}.pt"


# ---------- redes ----------

class Generator(nn.Module):
    def __init__(self, z: int = LATENT):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z, 128 * 7 * 7, bias=False), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 28 -> 14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x)


def _init_pesos(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def _crear_redes(device):
    g, d = Generator().to(device), Discriminator().to(device)
    g.apply(_init_pesos)
    d.apply(_init_pesos)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return g, d, opt_g, opt_d


def _bce(logits, objetivo: float):
    return F.binary_cross_entropy_with_logits(logits.float(), torch.full_like(logits, objetivo, dtype=torch.float32))


def _paso(g, d, opt_g, opt_d, real_u8, device, amp: bool):
    """Un paso de entrenamiento: primero el discriminador, luego el generador."""
    b = real_u8.size(0)
    real = real_u8.float().div(127.5).sub(1).unsqueeze(1)
    z = torch.randn(b, LATENT, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        falsa = g(z)
        loss_d = _bce(d(real), 0.9) + _bce(d(falsa.detach()), 0.0)  # 0.9: suavizado de etiquetas
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        loss_g = _bce(d(falsa), 1.0)
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()
    return loss_d.detach(), loss_g.detach()


# ---------- imágenes ----------

def cuadricula_png(img: torch.Tensor, columnas: int, escala: int = 4) -> bytes:
    """Arma una cuadrícula con las imágenes generadas (tensor N x 1 x 28 x 28 en [-1, 1])."""
    a = ((img.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()[:, 0]
    n = len(a)
    columnas = max(1, min(columnas, n))
    filas = math.ceil(n / columnas)
    pad = 2
    lienzo = np.full((filas * (28 + pad) + pad, columnas * (28 + pad) + pad), 35, dtype=np.uint8)
    for i in range(n):
        f, c = divmod(i, columnas)
        y, x = pad + f * (28 + pad), pad + c * (28 + pad)
        lienzo[y : y + 28, x : x + 28] = a[i]
    im = Image.fromarray(lienzo, "L")
    im = im.resize((im.width * escala, im.height * escala), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


# ---------- entrenamiento ----------

_lock = threading.Lock()
_ultima_muestra: dict = {"png": None, "version": 0}
_cache: dict = {}


def muestra_actual() -> tuple[bytes | None, int]:
    with _lock:
        return _ultima_muestra["png"], _ultima_muestra["version"]


def cargar_dataset(nombre: str) -> torch.Tensor:
    """Devuelve las imágenes de entrenamiento como tensor uint8 (N, 28, 28)."""
    from torchvision import datasets

    clase = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}.get(nombre)
    if clase is None:
        raise DatosError(f"Dataset desconocido: {nombre}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return clase(str(DATA_DIR), train=True, download=True).data
    except Exception as e:
        raise DatosError(f"No se pudo descargar el dataset (¿hay internet?): {e}") from e


def entrenar(report, stop, *, dataset: str, epocas: int, batch: int, dispositivo: str, amp: bool) -> None:
    device = dev.resolve_torch_device(dispositivo)
    usar_amp = bool(amp) and device.type == "cuda"
    nombre_disp = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    report(estado="cargando datos", dispositivo=nombre_disp, device_tipo=device.type, amp=usar_amp)

    x = cargar_dataset(dataset).to(device)
    n = len(x)
    pasos_epoca = n // batch
    if pasos_epoca < 1:
        raise DatosError("El batch es más grande que el dataset.")
    report(n_train=n, pasos_total=pasos_epoca * epocas, paso=0, epocas=epocas)

    g, d, opt_g, opt_d = _crear_redes(device)
    ruido_fijo = torch.randn(MUESTRAS, LATENT, generator=torch.Generator().manual_seed(7)).to(device)
    hechas = 0
    paso = 0
    for ep in range(1, epocas + 1):
        if stop.is_set():
            break
        report(estado=f"entrenando época {ep}/{epocas}", epoca=ep)
        g.train()
        d.train()
        perm = torch.randperm(n, device=device)
        suma_d = torch.zeros((), device=device)
        suma_g = torch.zeros((), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pasos_hechos = 0
        for k in range(pasos_epoca):
            if stop.is_set():
                break
            ld, lg = _paso(g, d, opt_g, opt_d, x[perm[k * batch : (k + 1) * batch]], device, usar_amp)
            suma_d += ld
            suma_g += lg
            pasos_hechos += 1
            paso += 1
            if paso % 10 == 0:
                report(paso=paso)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seg = time.perf_counter() - t0
        if pasos_hechos == 0:
            break
        hechas = ep
        g.eval()
        with torch.no_grad():
            png = cuadricula_png(g(ruido_fijo), 8)
        with _lock:
            _ultima_muestra["png"] = png
            _ultima_muestra["version"] += 1
        _guardar(g, dataset, hechas, nombre_disp)
        report(
            paso=paso,
            agregar={
                "historial": {
                    "epoca": ep,
                    "loss_d": round(float(suma_d) / pasos_hechos, 3),
                    "loss_g": round(float(suma_g) / pasos_hechos, 3),
                    "segundos": round(seg, 2),
                    "img_s": int(pasos_hechos * batch / seg),
                }
            },
        )
    if hechas:
        report(guardado=True)


def _guardar(g: Generator, dataset: str, epocas: int, dispositivo: str) -> None:
    ruta = checkpoint(dataset)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"G": {k: v.detach().cpu() for k, v in g.state_dict().items()}, "dataset": dataset, "epocas": epocas, "dispositivo": dispositivo},
        ruta,
    )
    with _lock:
        _cache.pop(dataset, None)


# ---------- generación ----------

def info_modelos() -> dict:
    out = {}
    for nombre in DATASETS:
        ruta = checkpoint(nombre)
        if ruta.exists():
            ck = torch.load(ruta, map_location="cpu", weights_only=True)
            out[nombre] = {"epocas": ck["epocas"], "dispositivo": ck["dispositivo"]}
        else:
            out[nombre] = None
    return out


def _generador(nombre: str):
    ruta = checkpoint(nombre)
    if not ruta.exists():
        raise DatosError(f"Todavía no hay un generador entrenado para «{DATASETS.get(nombre, nombre)}». Entrená uno primero.")
    mtime = ruta.stat().st_mtime
    with _lock:
        c = _cache.get(nombre)
        if c is None or c["mtime"] != mtime:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            g = Generator()
            g.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True)["G"])
            c = _cache[nombre] = {"mtime": mtime, "g": g.to(device).eval(), "device": device}
        return c["g"], c["device"]


@torch.no_grad()
def generar(nombre: str, n: int, semilla: int | None = None) -> tuple[bytes, float, str]:
    """Genera n imágenes aleatorias. Devuelve (png, milisegundos, dispositivo)."""
    g, device = _generador(nombre)
    gen = torch.Generator()
    gen.manual_seed(semilla) if semilla is not None else gen.seed()
    z = torch.randn(n, LATENT, generator=gen).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = g(z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return cuadricula_png(img, min(8, n)), ms, device.type


# ---------- benchmark CPU vs GPU ----------

def _medir_gan(device, amp: bool, pasos: int, batch: int) -> dict:
    g, d, opt_g, opt_d = _crear_redes(device)
    real = torch.randint(0, 256, (batch, 28, 28), dtype=torch.uint8, device=device)
    for _ in range(3):  # calentamiento (compila kernels, reserva memoria)
        _paso(g, d, opt_g, opt_d, real, device, amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(pasos):
        _paso(g, d, opt_g, opt_d, real, device, amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seg = time.perf_counter() - t0
    return {"pasos": pasos, "segundos": round(seg, 3), "img_s": int(pasos * batch / seg)}


def _medir_matmul(device, dtype, tf32: bool, n: int, iters: int) -> dict:
    previo = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(2):
            a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        seg = time.perf_counter() - t0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previo
    return {"segundos": round(seg, 3), "tflops": round(2 * n**3 * iters / seg / 1e12, 2)}


def benchmark(report, stop, *, pasos: int, batch: int) -> None:
    hay_gpu = torch.cuda.is_available()
    cpu, gpu = torch.device("cpu"), torch.device("cuda") if hay_gpu else None
    bf16 = hay_gpu and torch.cuda.is_bf16_supported()

    casos_gan = [("CPU · FP32", cpu, False, max(5, pasos // 5))]
    if hay_gpu:
        casos_gan.append(("GPU · FP32", gpu, False, pasos))
        if bf16:
            casos_gan.append(("GPU · BF16 (Tensor Cores)", gpu, True, pasos))
    n = 4096
    casos_mm = [("CPU · FP32", cpu, torch.float32, False, 3)]
    if hay_gpu:
        casos_mm += [
            ("GPU · FP32 (núcleos CUDA)", gpu, torch.float32, False, 100),
            ("GPU · TF32 (Tensor Cores)", gpu, torch.float32, True, 100),
            ("GPU · FP16 (Tensor Cores)", gpu, torch.float16, False, 100),
        ]
        if bf16:
            casos_mm.append(("GPU · BF16 (Tensor Cores)", gpu, torch.bfloat16, False, 100))

    total = len(casos_gan) + len(casos_mm)
    hecho = 0
    for nombre, device, amp, np_ in casos_gan:
        if stop.is_set():
            return
        report(estado=f"GAN · {nombre} ({np_} pasos de {batch} imágenes)…", progreso=100 * hecho / total)
        report(agregar={"gan": {"nombre": nombre, "tipo": device.type, **_medir_gan(device, amp, np_, batch)}})
        hecho += 1
    for nombre, device, dtype, tf32, iters in casos_mm:
        if stop.is_set():
            return
        report(estado=f"Multiplicación de matrices {n}×{n} · {nombre}…", progreso=100 * hecho / total)
        report(agregar={"matmul": {"nombre": nombre, "tipo": device.type, "n": n, **_medir_matmul(device, dtype, tf32, n, iters)}})
        hecho += 1
    report(progreso=100, hay_gpu=hay_gpu)
