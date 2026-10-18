"""Clasificador de animales: EfficientNet-B0 de PyTorch preentrenado en ImageNet.

ImageNet tiene 1000 clases; las primeras 398 (índices 0 a 397) son animales. Se
calcula la probabilidad entre esas clases y además cuánta probabilidad total cae
en "animal", para avisar si la foto no parece un animal.
"""
import threading

N_ANIMALES = 398
_lock = threading.Lock()
_cache: dict = {}


def _cargar():
    with _lock:
        if not _cache:
            import torch
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            pesos = EfficientNet_B0_Weights.IMAGENET1K_V1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            modelo = efficientnet_b0(weights=pesos).to(device).eval()
            _cache.update(modelo=modelo, prep=pesos.transforms(), cats=pesos.meta["categories"], device=device)
        return _cache


def nombre_dispositivo() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "PyTorch no instalado"


def clasificar(img, top: int = 3) -> dict:
    """img: PIL.Image. Devuelve {'top': [{'categoria','prob'}], 'prob_animal', 'dispositivo'}."""
    import torch
    import torch.nn.functional as F

    c = _cargar()
    x = c["prep"](img.convert("RGB")).unsqueeze(0).to(c["device"])
    with torch.no_grad():
        logits = c["modelo"](x)[0].float().cpu()
    prob_animal = float(F.softmax(logits, dim=0)[:N_ANIMALES].sum())
    probs = F.softmax(logits[:N_ANIMALES], dim=0)
    valores, indices = probs.topk(top)
    return {
        "top": [{"categoria": c["cats"][int(i)], "prob": round(float(v), 4)} for v, i in zip(valores, indices)],
        "prob_animal": round(prob_animal, 4),
        "dispositivo": c["device"].type,
    }
