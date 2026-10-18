"""Configuración leída del archivo .env de la raíz del repo.

Cada máquina tiene su propio .env (no se sube a Git), así el mismo código
corre en el escritorio y en la notebook con modelos distintos.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b")

# Modelos de Hugging Face (se descargan solos la primera vez)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "spanish")  # "auto" para detectar
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
TROCR_MODEL = os.getenv("TROCR_MODEL", "microsoft/trocr-base-handwritten")

DB_PATH = ROOT / os.getenv("DB_PATH", "data/ia.db")
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
WEIGHTS_DIR = ROOT / "weights"
PORT = int(os.getenv("PORT", "8000"))
