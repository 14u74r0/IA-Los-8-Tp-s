"""Tabla de comentarios de productos (Producto, Like, Dislike, cliente).

La usan el trabajo 6 (audio) y el 7 (texto); la columna `origen` los distingue.
likes / dislikes son banderas 0/1. `archivo` guarda el nombre del audio (trabajo 6).
"""
from . import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS comentarios (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    producto     TEXT NOT NULL,
    cliente      TEXT NOT NULL,
    comentario   TEXT NOT NULL,
    likes        INTEGER NOT NULL DEFAULT 0,
    dislikes     INTEGER NOT NULL DEFAULT 0,
    sentimiento  TEXT NOT NULL,
    motivo       TEXT,
    origen       TEXT NOT NULL DEFAULT 'texto',
    archivo      TEXT,
    creado_en    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


def init() -> None:
    db.ensure(SCHEMA)
    # Migración: bases creadas con la primera versión no tenían la columna `archivo`.
    with db.connect() as conn:
        columnas = {r["name"] for r in conn.execute("PRAGMA table_info(comentarios)")}
        if "archivo" not in columnas:
            conn.execute("ALTER TABLE comentarios ADD COLUMN archivo TEXT")


def guardar(
    producto: str,
    cliente: str,
    comentario: str,
    analisis: dict,
    origen: str = "texto",
    archivo: str | None = None,
) -> int:
    with db.connect() as conn:
        cur = conn.execute(
            """INSERT INTO comentarios
               (producto, cliente, comentario, likes, dislikes, sentimiento, motivo, origen, archivo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                producto,
                cliente,
                comentario,
                analisis["likes"],
                analisis["dislikes"],
                analisis["sentimiento"],
                analisis.get("motivo", ""),
                origen,
                archivo,
            ),
        )
        return cur.lastrowid


def listar(origen: str | None = None, limite: int = 200) -> list[dict]:
    sql = "SELECT * FROM comentarios"
    params: list = []
    if origen:
        sql += " WHERE origen = ?"
        params.append(origen)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limite)
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def borrar(id_: int) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM comentarios WHERE id = ?", (id_,))


def resumen(origen: str | None = None) -> dict:
    filas = listar(origen, limite=100000)
    return {
        "total": len(filas),
        "positivos": sum(1 for f in filas if f["sentimiento"] == "positivo"),
        "negativos": sum(1 for f in filas if f["sentimiento"] == "negativo"),
        "neutros": sum(1 for f in filas if f["sentimiento"] == "neutro"),
    }
