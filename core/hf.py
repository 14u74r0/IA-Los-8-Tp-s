"""Ayudas para Hugging Face + PyTorch que funcionan entre versiones."""


def cuda_disponible() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pipeline_device() -> int:
    """Argumento `device` de los pipelines: 0 = primera GPU, -1 = CPU."""
    return 0 if cuda_disponible() else -1


def dtype_kwarg(dtype) -> dict:
    """transformers renombró `torch_dtype` a `dtype`; usamos el que corresponda."""
    import transformers
    from packaging.version import Version

    clave = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {clave: dtype}


def dtype_preferido():
    """float16 en GPU (más rápido, usa Tensor Cores), float32 en CPU."""
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32
