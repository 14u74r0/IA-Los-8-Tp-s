"""Detección de GPU NVIDIA / CUDA, sin exigir que PyTorch esté instalado."""
import shutil
import subprocess


def gpu_info() -> dict:
    """Información de la GPU vía nvidia-smi (funciona igual en Windows y Linux)."""
    exe = shutil.which("nvidia-smi")
    vacio = {"disponible": False, "nombre": None, "vram_mb": None, "driver": None}
    if not exe:
        return vacio
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()[0]
        nombre, mem, driver = [x.strip() for x in out.split(",")]
        return {"disponible": True, "nombre": nombre, "vram_mb": int(mem), "driver": driver}
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return vacio


def gpu_stats() -> dict:
    """Uso en vivo de la GPU: utilización, VRAM, temperatura y consumo."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"disponible": False}
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()[0]
        util, usada, total, temp, watts, mhz = [x.strip() for x in out.split(",")]

        def num(v: str):
            try:
                return float(v)
            except ValueError:  # "[N/A]" en algunas notebooks
                return None

        return {
            "disponible": True,
            "uso_pct": num(util),
            "vram_usada_mb": num(usada),
            "vram_total_mb": num(total),
            "temp_c": num(temp),
            "watts": num(watts),
            "mhz": num(mhz),
        }
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return {"disponible": False}


def torch_device() -> str:
    """'cuda' si PyTorch ve la GPU, si no 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_device(preferencia: str = "auto"):
    """Devuelve un torch.device según 'auto', 'cpu' o 'cuda'."""
    import torch

    if preferencia == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if preferencia == "cuda":
        raise RuntimeError("PyTorch no ve la GPU (CUDA). Instalá la build con CUDA: pip install -r requirements-torch.txt")
    return torch.device("cpu")


def torch_info() -> dict:
    """Datos de PyTorch/CUDA para mostrar en la interfaz (núcleos CUDA, SMs, VRAM)."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False}
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_build": torch.version.cuda}
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        mayor, menor = p.major, p.minor
        # FP32 por SM: 64 en Volta/Turing/A100, 128 en Ampere GA10x, Ada, Hopper y Blackwell
        por_sm = 64 if (mayor, menor) in ((7, 0), (7, 5), (8, 0)) else 128
        info.update(
            nombre=p.name,
            capacidad=f"{mayor}.{menor}",
            sms=p.multi_processor_count,
            nucleos_cuda=p.multi_processor_count * por_sm,
            vram_gb=round(p.total_memory / 1024**3, 1),
            bf16=torch.cuda.is_bf16_supported(),
        )
    return info
