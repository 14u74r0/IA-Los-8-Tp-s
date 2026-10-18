"""Entrenamiento y predicción de la CNN (PyTorch, usa CUDA si está disponible)."""
import math
import random
import threading
import time

import torch
import torch.nn.functional as F
from PIL import Image

from core import config, device as dev, images

from . import data
from .model import CNN

CHECKPOINT = config.WEIGHTS_DIR / "t1_cnn.pt"

_lock = threading.Lock()
_cache: dict = {}


def _normalizar(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - 0.5) / 0.5


def _aumentar(x: torch.Tensor) -> torch.Tensor:
    """Espejo horizontal al azar + recorte aleatorio (todo en la GPU)."""
    giro = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(giro[:, None, None, None], x.flip(3), x)
    alto, ancho = x.shape[-2:]
    pad = max(2, alto // 8)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i, j = random.randint(0, 2 * pad), random.randint(0, 2 * pad)
    return xp[:, :, i : i + alto, j : j + ancho]


@torch.no_grad()
def _evaluar(modelo, x, y, device, amp: bool) -> float:
    modelo.eval()
    bien = 0
    for k in range(0, len(x), 1024):
        xb = _normalizar(x[k : k + 1024])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            bien += (modelo(xb).argmax(1) == y[k : k + 1024]).sum().item()
    return bien / len(x)


def entrenar(report, stop, *, dataset: str, epocas: int, batch: int, lr: float, dispositivo: str, amp: bool) -> None:
    device = dev.resolve_torch_device(dispositivo)
    usar_amp = bool(amp) and device.type == "cuda"
    nombre_disp = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    report(estado="cargando datos", dispositivo=nombre_disp, device_tipo=device.type, amp=usar_amp)

    xtr, ytr, xte, yte, clases, tam, nombre_ds = data.cargar(dataset)
    xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))
    n = len(xtr)
    pasos_epoca = math.ceil(n / batch)
    pasos_total = pasos_epoca * epocas
    report(clases=clases, dataset_nombre=nombre_ds, n_train=n, n_test=len(xte), pasos_total=pasos_total, paso=0)

    modelo = CNN(len(clases)).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos_total)

    paso, acc_test, epocas_hechas = 0, 0.0, 0
    for ep in range(1, epocas + 1):
        if stop.is_set():
            break
        modelo.train()
        perm = torch.randperm(n, device=device)
        suma_loss = torch.zeros((), device=device)
        aciertos = torch.zeros((), device=device)
        report(estado=f"entrenando época {ep}/{epocas}", epoca=ep, epocas=epocas)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vistos = 0
        for k in range(0, n, batch):
            if stop.is_set():
                break
            idx = perm[k : k + batch]
            xb, yb = _aumentar(_normalizar(xtr[idx])), ytr[idx]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=usar_amp):
                salida = modelo(xb)
                loss = F.cross_entropy(salida, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            suma_loss += loss.detach() * len(idx)
            aciertos += (salida.argmax(1) == yb).sum()
            vistos += len(idx)
            paso += 1
            if paso % 10 == 0:
                report(paso=paso, loss_actual=round(float(loss.detach()), 4))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seg = time.perf_counter() - t0
        if vistos == 0:
            break
        acc_test = _evaluar(modelo, xte, yte, device, usar_amp)
        epocas_hechas = ep
        fila = {
            "epoca": ep,
            "loss": round(float(suma_loss) / vistos, 4),
            "acc_train": round(float(aciertos) / vistos, 4),
            "acc_test": round(acc_test, 4),
            "segundos": round(seg, 2),
            "img_s": int(vistos / seg),
        }
        report(paso=paso, agregar={"historial": fila})

    if epocas_hechas == 0:
        return
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
            "clases": clases,
            "tam": tam,
            "dataset": nombre_ds,
            "acc": acc_test,
            "epocas": epocas_hechas,
            "dispositivo": nombre_disp,
        },
        CHECKPOINT,
    )
    with _lock:
        _cache.clear()
    report(guardado=True, acc_final=round(acc_test, 4))


# ---------- predicción ----------

def info_modelo() -> dict | None:
    if not CHECKPOINT.exists():
        return None
    ck = _cargar_checkpoint()
    return {k: ck[k] for k in ("clases", "tam", "dataset", "acc", "epocas", "dispositivo")}


def _cargar_checkpoint() -> dict:
    mtime = CHECKPOINT.stat().st_mtime
    with _lock:
        if _cache.get("mtime") != mtime:
            _cache.clear()
            _cache["mtime"] = mtime
            _cache["ck"] = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            _cache.pop("modelo", None)
        return _cache["ck"]


def _modelo_listo():
    ck = _cargar_checkpoint()
    with _lock:
        if "modelo" not in _cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = CNN(len(ck["clases"]))
            m.load_state_dict(ck["state"])
            _cache["modelo"] = (m.to(device).eval(), device)
        return ck, _cache["modelo"]


@torch.no_grad()
def predecir(contenido: bytes, top: int = 3) -> dict:
    if not CHECKPOINT.exists():
        raise data.DatosError("Todavía no hay un modelo entrenado. Entrená uno primero.")
    ck, (modelo, device) = _modelo_listo()
    img = images.abrir(contenido).convert("RGB").resize((ck["tam"], ck["tam"]))
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(ck["tam"], ck["tam"], 3)
    x = _normalizar(x.permute(2, 0, 1).unsqueeze(0).to(device))
    t0 = time.perf_counter()
    probs = F.softmax(modelo(x), dim=1)[0].cpu()
    ms = (time.perf_counter() - t0) * 1000
    valores, indices = probs.topk(min(top, len(ck["clases"])))
    return {
        "predicciones": [{"clase": ck["clases"][i], "prob": round(float(v), 4)} for v, i in zip(valores, indices)],
        "dispositivo": device.type,
        "ms": round(ms, 1),
        "entrenado_con": ck["dataset"],
    }
