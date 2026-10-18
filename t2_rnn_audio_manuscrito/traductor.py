"""Traducción de texto con Llama local (Ollama)."""
from core import ollama_client

# Valor que se muestra -> nombre que se le pide al modelo
DESTINOS = {
    "Español": "español",
    "Inglés": "inglés",
    "Portugués": "portugués",
    "Francés": "francés",
    "Italiano": "italiano",
    "Alemán": "alemán",
    "Japonés": "japonés",
    "Chino": "chino simplificado",
}

# Idiomas que se le pueden indicar a Whisper (nombre en inglés, como lo espera el modelo)
IDIOMAS_AUDIO = {
    "auto": "Detectar automáticamente",
    "spanish": "Español",
    "english": "Inglés",
    "portuguese": "Portugués",
    "french": "Francés",
    "italian": "Italiano",
    "german": "Alemán",
}


def traducir(texto: str, destino: str) -> str:
    if destino not in DESTINOS:
        raise ValueError(f"Idioma de destino no soportado: {destino}")
    return ollama_client.chat(
        f"Traducí el siguiente texto al {DESTINOS[destino]}. "
        "Respondé únicamente con la traducción, sin comentarios ni explicaciones.\n\n"
        f"{texto}",
        system="Sos un traductor profesional. Conservás el significado, el tono y la puntuación.",
        temperature=0.1,
        num_ctx=4096,
    ).strip()
