"""Acceso a SQLite compartido por los trabajos que guardan resultados."""
import sqlite3
from contextlib import contextmanager

from . import config


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure(schema_sql: str) -> None:
    """Crea las tablas si no existen (idempotente)."""
    with connect() as conn:
        conn.executescript(schema_sql)
