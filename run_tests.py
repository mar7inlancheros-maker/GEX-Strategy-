"""Ejecuta todas las suites y devuelve el codigo de salida agregado.

Cada `tests/test_*.py` es un script autonomo que imprime su propio informe y sale
con 1 si algo falla. Esto solo los encadena para poder correr todo de una y para
que CI tenga un unico comando. Misma convencion que el proyecto hermano: sin
pytest, porque las suites son scripts y no hace falta una dependencia mas.

    python run_tests.py            # todas, solo el resumen
    python run_tests.py -v         # ademas la salida completa de cada una
    python run_tests.py gex        # solo las que casen con ese texto
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    patterns = [a for a in argv if not a.startswith("-")]

    suites = sorted((ROOT / "tests").glob("test_*.py"))
    if patterns:
        suites = [s for s in suites if any(p in s.name for p in patterns)]
    if not suites:
        print("ninguna suite coincide con:", " ".join(patterns))
        return 1

    failed: list[str] = []
    for suite in suites:
        # encoding explicito: las suites reconfiguran su stdout a UTF-8 y en
        # Windows el descodificador por defecto es cp1252, que revienta con la
        # primera 'sigma' o '+-' del informe.
        proc = subprocess.run(
            [sys.executable, str(suite)],
            capture_output=not verbose,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            failed.append(suite.name)
            if not verbose:
                print(proc.stdout or "")
                print(proc.stderr or "")
            print(f"[FALLA]  {suite.name}")
        else:
            tail = ""
            if not verbose and proc.stdout:
                for line in reversed(proc.stdout.splitlines()):
                    if "RESULTADO" in line:
                        tail = "  " + line.strip()
                        break
            print(f"[OK]     {suite.name}{tail}")

    print()
    if failed:
        print(f"FALLAN {len(failed)} de {len(suites)} suites: {', '.join(failed)}")
        return 1
    print(f"Las {len(suites)} suites pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
