"""Análisis de sentimiento de comentarios usando un LLM local (Ollama)."""
from . import ollama_client

SYSTEM = (
    "Sos un analizador de comentarios de clientes sobre productos. "
    "Respondé SOLO un objeto JSON con estas claves: "
    '"sentimiento" (uno de: "positivo", "negativo", "neutro") y '
    '"motivo" (una frase corta en español que explique la decisión). '
    "Si el comentario mezcla cosas buenas y malas, elegí el sentimiento dominante."
)

VALIDOS = {"positivo", "negativo", "neutro"}


def analizar(texto: str, model: str | None = None) -> dict:
    """Devuelve {'sentimiento', 'likes', 'dislikes', 'motivo'}.

    like/dislike se derivan del sentimiento (no se le pide al modelo), así el
    resultado siempre es consistente aunque el modelo sea chico.
    """
    data = ollama_client.chat_json(
        f"Comentario del cliente:\n\"\"\"{texto}\"\"\"", system=SYSTEM, model=model
    )
    sentimiento = str(data.get("sentimiento", "")).strip().lower()
    if sentimiento not in VALIDOS:
        sentimiento = "neutro"
    return {
        "sentimiento": sentimiento,
        "likes": 1 if sentimiento == "positivo" else 0,
        "dislikes": 1 if sentimiento == "negativo" else 0,
        "motivo": str(data.get("motivo", "")).strip()[:300],
    }
