"""Características del animal (especie, color, tamaño, hábitat, dieta) con un modelo de visión de Ollama."""
from core import config, images, ollama_client

SYSTEM = "Sos un biólogo que describe animales a partir de fotografías. Respondés solo con JSON."

PROMPT = (
    "Mirá la fotografía y respondé SOLO un objeto JSON con estas claves:\n"
    '- "es_animal": true o false (hay un animal real en la foto)\n'
    '- "especie": nombre común en español (ej: "perro labrador"); "desconocido" si no se sabe\n'
    '- "color": colores principales del animal\n'
    '- "tamano": "pequeño", "mediano" o "grande" para esa especie\n'
    '- "habitat": dónde vive normalmente\n'
    '- "dieta": "herbívoro", "carnívoro", "omnívoro" o "desconocida"\n'
    '- "descripcion": una o dos frases sobre lo que se ve en la foto\n'
    "Si no hay ningún animal, poné es_animal en false y el resto en \"-\"."
)

CAMPOS = ("especie", "color", "tamano", "habitat", "dieta", "descripcion")


def _texto(valor, limite: int = 300) -> str:
    if isinstance(valor, (list, tuple)):
        valor = ", ".join(str(v) for v in valor)
    return str(valor or "").strip()[:limite]


def describir(img) -> dict:
    data = ollama_client.chat_json(
        PROMPT,
        system=SYSTEM,
        model=config.OLLAMA_VISION_MODEL,
        images=[images.a_base64_jpeg(img, 768)],
        temperature=0.1,
    )
    es_animal = data.get("es_animal", True)
    if isinstance(es_animal, str):
        es_animal = es_animal.strip().lower() in ("true", "si", "sí", "1")
    out = {c: _texto(data.get(c)) for c in CAMPOS}
    out["es_animal"] = bool(es_animal)
    return out
