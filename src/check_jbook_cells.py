"""Ejecuta las celdas de código de los capítulos MyST del libro sin Sphinx.

Uso:  python src/check_jbook_cells.py [capitulo.md ...]
Sirve como verificación rápida (sin jupyter-book) de que todas las celdas
``{code-cell}`` se ejecutan en orden y que los ``assert`` de coherencia pasan.
``myst_nb.glue`` se sustituye por un stub si el paquete no está instalado.
"""
import os
import re
import sys
import types
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent / "book"
os.chdir(BOOK)
sys.path.insert(0, str(BOOK))
os.environ.setdefault("MPLBACKEND", "Agg")

try:  # pragma: no cover
    import myst_nb  # noqa: F401
except ImportError:
    stub = types.ModuleType("myst_nb")
    stub.glue = lambda *a, **k: None
    sys.modules["myst_nb"] = stub

try:  # pragma: no cover
    import IPython  # noqa: F401
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(?::[^\n]*\n)*(.*?)^```", re.S | re.M)


def run(md: Path) -> None:
    ns = {"__name__": "__main__"}
    cells = CELL.findall(md.read_text(encoding="utf-8"))
    for i, code in enumerate(cells, 1):
        if not HAS_IPYTHON:  # sin IPython: display() pasa a ser un no-op
            code = code.replace("from IPython.display import display", "display = lambda *a, **k: None")
        try:
            exec(compile(code, f"{md.name}[celda {i}]", "exec"), ns)
        except Exception as exc:
            print(f"FALLO {md.name} celda {i}: {type(exc).__name__}: {exc}")
            raise
    print(f"OK {md.name}: {len(cells)} celdas")


if __name__ == "__main__":
    targets = [BOOK / a for a in sys.argv[1:]] or sorted(BOOK.glob("cap*.md"))
    for t in targets:
        run(t)
