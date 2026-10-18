"""Lanzador único: python run.py t7_texto  (opcional: --port 8001)."""
import argparse
import importlib.util
import sys

import uvicorn

from core import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta uno de los trabajos")
    parser.add_argument("proyecto", help="nombre de la carpeta, ej: t7_texto")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    modulo = f"{args.proyecto}.app"
    try:
        existe = importlib.util.find_spec(modulo) is not None
    except ModuleNotFoundError:
        existe = False
    if not existe:
        sys.exit(f"No existe {args.proyecto}/app.py (todavía). Carpetas disponibles: t1_cnn ... t8_pdf")

    print(f"Abrí http://{args.host}:{args.port}")
    uvicorn.run(f"{modulo}:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
