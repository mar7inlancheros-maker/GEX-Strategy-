"""Recupera spot y dividendo de SPX de una cadena de Bloomberg, y calibra la gamma.

POR QUE ESTE SCRIPT EXISTE
--------------------------
`configs/default.toml` fija `universe.SPX.dividend_yield = 0.0030` y afirma que
ese numero esta MEDIDO, no supuesto. Una afirmacion asi solo vale si se puede
volver a producir con un comando. Esto es ese comando.

Antes vivia unicamente dentro de `tests/test_bloomberg_omon.py`, que comprueba
que el resultado no se rompe pero no lo IMPRIME: para ver de donde sale el 0,30%
habia que leer el codigo de una suite. Un valor de configuracion que nadie puede
reproducir sin leer un test es, a efectos practicos, un numero magico con buena
prensa.

EL METODO, EN DOS IDENTIDADES
-----------------------------
El export de OMON no trae el precio del subyacente. No hace falta pedirselo a
nadie, esta dentro de la propia cadena:

    1. Paridad put-call, por vencimiento:   F = K + (C - P) e^{rT}
    2. Deriva del forward:                  ln F = ln S + (r - q) T

Una regresion de ln F contra T sobre varios vencimientos da `ln S` en la ordenada
y `(r - q)` en la pendiente. Con el `r` que el propio Bloomberg publica en la
cabecera de cada grupo, `q` queda despejado.

Dos supuestos convertidos en dos medidas. Ese es el punto, mas que el numero.

USO
---
    python research/calibrate_from_bloomberg.py
    python research/calibrate_from_bloomberg.py --file "otro export.xlsx"
    python research/calibrate_from_bloomberg.py --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gamma_quant.config import Config, project_root, reports_dir
from gamma_quant.data.ingestion.bloomberg_omon import (
    calibrate_against_bloomberg,
    implied_spot_and_dividend,
    read_omon_xlsx,
)

OMON_DIR = project_root() / "data" / "external" / "bloomberg"
DEFAULT_FILE = OMON_DIR / "SPX OMON Greek Options.xlsx"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Calibracion contra Bloomberg OMON")
    ap.add_argument("--file", default=None, help="export XLSX de OMON")
    ap.add_argument("--symbol", default="SPX")
    ap.add_argument("--no-write", action="store_true",
                    help="no escribe el informe en reports/")
    args = ap.parse_args(argv)

    path = Path(args.file) if args.file else DEFAULT_FILE
    if not path.is_absolute():
        path = OMON_DIR / path
    if not path.is_file():
        print(f"no existe el export: {path}")
        print(f"Los exports de OMON viven en {OMON_DIR}")
        return 1

    print("=" * 78)
    print(f"CALIBRACION CONTRA BLOOMBERG — {args.symbol}")
    print("=" * 78)

    export = read_omon_xlsx(path, symbol=args.symbol)
    print("\n" + export.report())

    # --- Paso 1: spot y q, de la propia cadena ---------------------------- #
    spot, q, forwards = implied_spot_and_dividend(export.chain)

    print("\nPASO 1 — spot y dividendo por paridad put-call")
    print("-" * 46)
    print(forwards.to_string(index=False))
    print(f"\n  spot implicito : {spot:,.2f}")
    print(f"  q implicito    : {q:.4%}")

    # --- Paso 2: nuestra gamma contra la suya ----------------------------- #
    cal = calibrate_against_bloomberg(export.chain, spot=spot, dividend_yield=q)
    raw_ratio = float(cal["raw_ratio"].median())
    med = float(cal["rel_error"].median())
    p90 = float(cal["rel_error"].quantile(0.90))

    print("\nPASO 2 — unidades de la gamma de Bloomberg")
    print("-" * 46)
    print(f"  mediana de GL/gamma_BS : {raw_ratio:.2f}")
    print(f"  S * 0,01               : {spot * 0.01:.2f}")
    print("  => GL es gamma por movimiento del 1%, NO por dolar.")
    print("     Usarla cruda multiplica el GEX por ~%.0f sin dar ningun error." % raw_ratio)

    print("\nPASO 3 — acuerdo tras convertir")
    print("-" * 46)
    print(f"  contratos comparados : {len(cal)}")
    print(f"  error relativo mediano : {med:.2%}")
    print(f"  p90                    : {p90:.2%}")

    # --- Contraste contra lo que hay escrito en la configuracion ---------- #
    cfg = Config.load()
    configured = cfg.underlying(args.symbol).dividend_yield
    drift = abs(q - configured)
    print("\nCONTRA LA CONFIGURACION")
    print("-" * 46)
    print(f"  configs/default.toml : dividend_yield = {configured:.4%}")
    print(f"  medido aqui          : {q:.4%}")
    if drift > 0.002:
        print(f"  DISCREPA en {drift:.2%}. O el export es de otro regimen de tipos, o")
        print("  el valor de la configuracion se quedo atras. Hay que decidirlo a mano:")
        print("  este script NO reescribe la configuracion, a proposito.")
    else:
        print(f"  coinciden dentro de {drift:.2%}. El valor de la configuracion se sostiene.")

    if not args.no_write:
        out = {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_file": path.name,
            "as_of": str(export.as_of),
            "symbol": args.symbol,
            "n_contracts": int(export.n_contracts),
            "implied_spot": spot,
            "implied_dividend_yield": q,
            "configured_dividend_yield": configured,
            "gamma_raw_ratio_median": raw_ratio,
            "gamma_rel_error_median": med,
            "gamma_rel_error_p90": p90,
            "n_calibration_contracts": int(len(cal)),
        }
        dest = reports_dir() / "bloomberg_calibration.json"
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ninforme: {dest}")

    print("\nAVISO: el export son ~150 contratos de una cadena de ~28.000. Sirve para")
    print("CALIBRAR el motor, no como panel de investigacion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
