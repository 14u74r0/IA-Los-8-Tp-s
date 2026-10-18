"""Tabla `pdfs`: nombre del archivo y resumen generado por la IA."""
from core import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS pdfs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_archivo TEXT NOT NULL,
    resumen        TEXT NOT NULL,
    paginas        INTEGER,
    caracteres     INTEGER,
    parcial        INTEGER NOT NULL DEFAULT 0,
    modelo         TEXT,
    creado_en      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


def init() -> None:
    db.ensure(SCHEMA)


def guardar(nombre_archivo: str, resumen: str, paginas: int, caracteres: int, parcial: bool, modelo: str) -> int:
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO pdfs (nombre_archivo, resumen, paginas, caracteres, parcial, modelo) VALUES (?, ?, ?, ?, ?, ?)",
            (nombre_archivo, resumen, paginas, caracteres, 1 if parcial else 0, modelo),
        )
        return cur.lastrowid


def listar(limite: int = 300) -> list[dict]:
    with db.connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM pdfs ORDER BY id DESC LIMIT ?", (limite,))]


def borrar(id_: int) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM pdfs WHERE id = ?", (id_,))
