"""Tabla `animales`: características de cada animal analizado en una fotografía."""
import json

from core import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS animales (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    archivo            TEXT NOT NULL,
    nombre_original    TEXT,
    especie            TEXT,
    categoria_imagenet TEXT,
    confianza          REAL,
    top3               TEXT,
    color              TEXT,
    tamano             TEXT,
    habitat            TEXT,
    dieta              TEXT,
    descripcion        TEXT,
    es_animal          INTEGER NOT NULL DEFAULT 1,
    notas              TEXT,
    creado_en          TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

COLUMNAS = (
    "archivo", "nombre_original", "especie", "categoria_imagenet", "confianza", "top3",
    "color", "tamano", "habitat", "dieta", "descripcion", "es_animal", "notas",
)


def init() -> None:
    db.ensure(SCHEMA)


def guardar(fila: dict) -> int:
    valores = [fila.get(c) for c in COLUMNAS]
    valores[COLUMNAS.index("top3")] = json.dumps(fila.get("top3") or [], ensure_ascii=False)
    with db.connect() as conn:
        cur = conn.execute(
            f"INSERT INTO animales ({', '.join(COLUMNAS)}) VALUES ({', '.join('?' * len(COLUMNAS))})", valores
        )
        return cur.lastrowid


def _fila(r) -> dict:
    d = dict(r)
    d["top3"] = json.loads(d["top3"] or "[]")
    return d


def listar(limite: int = 300) -> list[dict]:
    with db.connect() as conn:
        return [_fila(r) for r in conn.execute("SELECT * FROM animales ORDER BY id DESC LIMIT ?", (limite,))]


def obtener(id_: int) -> dict | None:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM animales WHERE id = ?", (id_,)).fetchone()
        return _fila(r) if r else None


def borrar(id_: int) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM animales WHERE id = ?", (id_,))
