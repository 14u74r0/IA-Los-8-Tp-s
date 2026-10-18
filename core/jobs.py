"""Tareas largas (entrenar, benchmark) en un hilo aparte, con estado consultable.

La interfaz web consulta `snapshot()` cada segundo y muestra el progreso.
La función que se ejecuta recibe (report, stop, **parametros):
  - report(**campos) actualiza el estado visible,
  - stop es un threading.Event que se activa al pedir "Detener".
"""
import threading
import time
import traceback


class JobBusy(RuntimeError):
    """Ya hay una tarea corriendo."""


class BackgroundJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict = {"estado": "inactivo"}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, fn, **kwargs) -> None:
        with self._lock:
            if self.running:
                raise JobBusy("Ya hay un proceso en ejecución. Esperá a que termine o detenelo.")
            self._stop.clear()
            self._state = {"estado": "iniciando", "inicio": time.time(), "historial": []}
            self._thread = threading.Thread(target=self._run, args=(fn, kwargs), daemon=True)
            self._thread.start()

    def _run(self, fn, kwargs) -> None:
        try:
            fn(self.update, self._stop, **kwargs)
            self.update(estado="detenido" if self._stop.is_set() else "terminado")
        except Exception as e:  # el error se muestra en la interfaz
            traceback.print_exc()
            # Los errores propios (ValueError/RuntimeError con mensaje en español) se muestran tal cual
            texto = str(e) if isinstance(e, (ValueError, RuntimeError)) else f"{type(e).__name__}: {e}"
            self.update(estado="error", error=texto)
        finally:
            self.update(fin=time.time())

    def update(self, agregar: dict | None = None, **fields) -> None:
        """Actualiza campos del estado. `agregar={'historial': fila}` suma un elemento a una lista."""
        with self._lock:
            self._state.update(fields)
            for clave, item in (agregar or {}).items():
                self._state.setdefault(clave, []).append(item)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        with self._lock:
            s = {k: (list(v) if isinstance(v, list) else v) for k, v in self._state.items()}
        s["corriendo"] = self.running
        if "inicio" in s:
            s["transcurrido"] = round((s.get("fin") or time.time()) - s["inicio"], 1)
        return s
