"""Cliente mínimo para Ollama (misma API en Windows y Linux)."""
import json

import httpx

from . import config


class OllamaError(RuntimeError):
    """Error legible para mostrar al usuario en la interfaz."""


def _url(path: str) -> str:
    return f"{config.OLLAMA_HOST}{path}"


def is_up() -> bool:
    try:
        return httpx.get(_url("/api/tags"), timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def list_models() -> list[str]:
    try:
        r = httpx.get(_url("/api/tags"), timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError:
        return []


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    images: list[str] | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    num_ctx: int | None = None,
    timeout: float = 300,
) -> str:
    """Envía un mensaje y devuelve el texto de la respuesta.

    images: lista de imágenes en base64 (solo modelos de visión).
    num_ctx: tamaño de la ventana de contexto (útil para textos largos).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    options: dict = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = num_ctx

    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"

    try:
        r = httpx.post(_url("/api/chat"), json=payload, timeout=timeout)
    except httpx.ConnectError as e:
        raise OllamaError(
            "No se pudo conectar con Ollama. ¿Está corriendo? (abrí la app o ejecutá: ollama serve)"
        ) from e
    except httpx.HTTPError as e:
        raise OllamaError(f"Error hablando con Ollama: {e}") from e

    if r.status_code == 404:
        raise OllamaError(
            f"El modelo '{payload['model']}' no está descargado. Ejecutá: ollama pull {payload['model']}"
        )
    if r.status_code >= 400:
        raise OllamaError(f"Ollama respondió {r.status_code}: {r.text[:200]}")

    return r.json()["message"]["content"]


def chat_json(prompt: str, **kwargs) -> dict:
    """Como chat(), pero devuelve un dict (el modelo responde en JSON)."""
    text = chat(prompt, json_mode=True, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise OllamaError("El modelo no devolvió un JSON válido. Probá de nuevo.")
