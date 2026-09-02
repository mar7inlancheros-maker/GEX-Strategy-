"""Contraste del lector de OMON y CALIBRACION contra Bloomberg.

Esta suite es distinta de las otras dos: las demas comprueban que el codigo
reproduce matematicas conocidas; esta comprueba que reproduce LA REFERENCIA
INSTITUCIONAL sobre datos reales de SPX.

Lo que fija:

  1. Que la fecha del export se DEDUZCA del contenido y no del nombre del
     fichero. "SPX OMON as of 29may.xlsx" debe dar 2026-05-29 porque sus grupos
     dicen "18-Jun-26 (20d)", no porque lo ponga el nombre.
  2. Que el strike salga de la COLUMNA Strike. Los tickers de OMON vienen
     truncados por Excel ("SPX 9/18/26 C763" es el strike 7635) y leerlos seria
     un error silencioso de varios cientos de puntos.
  3. Que la IV se convierta de porcentaje (12,03 -> 0,1203).
  4. Que SPX y SPXW se conserven como raices DISTINTAS.
  5. LA CALIBRACION: que `GL` de Bloomberg sea gamma por 1% y que, convertida,
     coincida con la nuestra dentro de un margen estrecho.

Si el punto 5 empieza a fallar, o hemos roto las griegas o hemos cambiado un
supuesto (r, q, tau) sin darnos cuenta. En ambos casos hay que enterarse.

Se ejecuta solo:  python tests/test_bloomberg_omon.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from gamma_quant.data.ingestion.bloomberg_omon import (
    calibrate_against_bloomberg,
    implied_spot_and_dividend,
    read_omon_xlsx,
)

_FAILURES: list[str] = []
_CHECKS = 0

# Los exports viven en `data/external/bloomberg/`, no en la raiz: son datos de
# entrada. Se versionan a proposito, al contrario que el resto de `data/`, porque
# son irreemplazables (exports manuales del Terminal) y porque sin ellos esta
# suite entera se salta en silencio.
OMON_DIR = ROOT / "data" / "external" / "bloomberg"
GREEKS_FILE = OMON_DIR / "SPX OMON Greek Options.xlsx"
MAY_FILE = OMON_DIR / "SPX OMON as of 29may.xlsx"


def check(condition: bool, label: str, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        msg = label + (f"  -> {detail}" if detail else "")
        print(f"  [FALLO] {msg}")
        _FAILURES.append(msg)


def block(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def test_parsing() -> None:
    block("BLOQUE 1 — lectura del export y sus trampas")

    if not GREEKS_FILE.exists():
        print(f"  [salta] no esta {GREEKS_FILE.name}")
        return

    e = read_omon_xlsx(GREEKS_FILE)

    check(e.as_of == date(2026, 9, 1),
          "la fecha se deduce del contenido (vencimiento - dias)", str(e.as_of))
    check(e.n_contracts > 100, f"se leen {e.n_contracts} contratos")
    check(set(e.chain["root"].unique()) == {"SPX", "SPXW"},
          "SPX y SPXW se conservan como raices distintas",
          str(sorted(e.chain["root"].unique())))
    check(set(e.chain["option_type"].unique()) == {"C", "P"},
          "se leen los dos bloques: calls Y puts")

    # TRAMPA DEL TICKER TRUNCADO: el ticker dice 'C763' pero el strike es 7635.
    row = e.chain.iloc[0]
    check(row["strike"] > 1000,
          "el strike sale de su columna, no del ticker truncado",
          f"strike={row['strike']} ticker={row['contract_symbol']!r}")

    # TRAMPA DE LA IV EN PORCENTAJE.
    iv = e.chain["implied_volatility"].dropna()
    check(bool((iv > 0.01).all() and (iv < 2.0).all()),
          "la IV esta en tanto por uno (0,12 y no 12,03)",
          f"rango {iv.min():.4f}-{iv.max():.4f}")

    # El tipo de Bloomberg entra como fraccion.
    r = e.chain["risk_free_rate"].dropna()
    check(bool((r > 0.01).all() and (r < 0.10).all()),
          "el tipo de Bloomberg entra como fraccion (0,0411)",
          f"rango {r.min():.4f}-{r.max():.4f}")

    check(bool((e.chain["open_interest"].fillna(0) >= 0).all()),
          "no hay open interest negativo")
    check(e.chain["open_interest"].sum() > 0,
          "el export trae open interest (sin el no hay GEX)")


def test_historical_export() -> None:
    block("BLOQUE 2 — el export historico se autoverifica")

    if not MAY_FILE.exists():
        print(f"  [salta] no esta {MAY_FILE.name}")
        return

    e = read_omon_xlsx(MAY_FILE)
    check(e.as_of == date(2026, 5, 29),
          "'as of 29may' se CONFIRMA desde los datos, no desde el nombre",
          str(e.as_of))
    check(all(x > e.as_of for x in e.expirations),
          "todos los vencimientos son posteriores a la fecha del export")

    dias = (date(2026, 9, 1) - e.as_of).days
    print(f"  [info] el export historico esta a {dias} dias vista. La ventana "
          f"documentada de Bloomberg son 90.")
    check(dias > 90,
          "Bloomberg sirvio datos MAS ALLA de su ventana documentada de 90 dias",
          f"{dias} dias")


def test_calibration() -> None:
    block("BLOQUE 3 — CALIBRACION contra Bloomberg (lo importante)")

    if not GREEKS_FILE.exists():
        print(f"  [salta] no esta {GREEKS_FILE.name}")
        return

    e = read_omon_xlsx(GREEKS_FILE)
    spot, q, forwards = implied_spot_and_dividend(e.chain)

    print(f"  [info] spot implicito por paridad put-call: {spot:,.2f}")
    print(f"  [info] q implicito: {q:.4%}")

    check(7000 < spot < 8500,
          "el spot recuperado por paridad es plausible para SPX", f"{spot:,.2f}")
    check(-0.01 < q < 0.05,
          "el dividendo implicito es plausible (0-5%)", f"{q:.4%}")
    check(bool(forwards["forward"].is_monotonic_increasing),
          "la curva de forwards crece con el plazo (r > q)")

    cal = calibrate_against_bloomberg(e.chain, spot=spot, dividend_yield=q)

    # 1) LAS UNIDADES. GL = gamma_BS * S * 0,01, luego el ratio crudo debe salir
    #    ~S/100 (~76 en SPX) y NO ~1. Es la comprobacion que evita un GEX x76.
    ratio = float(cal["raw_ratio"].median())
    expected = spot * 0.01
    print(f"  [info] ratio GL/gamma_BS = {ratio:.2f}   |   S*0,01 = {expected:.2f}")
    check(abs(ratio / expected - 1.0) < 0.05,
          "CONFIRMADO: la gamma de Bloomberg es por movimiento del 1%, no por $1",
          f"ratio {ratio:.2f} vs esperado {expected:.2f}")
    check(ratio > 10.0,
          "y no es la gamma BSM cruda (el ratio no esta cerca de 1)")

    # 2) EL ACUERDO. Una vez convertida, ¿coincide con la nuestra?
    med = float(cal["rel_error"].median())
    p90 = float(cal["rel_error"].quantile(0.90))
    print(f"  [info] error relativo tras convertir: mediana {med:.2%}, p90 {p90:.2%}")
    check(med < 0.05,
          "nuestra gamma BSM coincide con la de Bloomberg (mediana < 5%)",
          f"{med:.2%}")
    check(p90 < 0.20,
          "y el 90% de los contratos queda por debajo del 20%",
          f"{p90:.2%}")
    check(int(len(cal)) > 100, f"la calibracion usa {len(cal)} contratos")


def test_gex_from_bloomberg() -> None:
    block("BLOQUE 4 — GEX extremo a extremo desde el export")

    if not GREEKS_FILE.exists():
        print(f"  [salta] no esta {GREEKS_FILE.name}")
        return

    from gamma_quant.options import compute_gex
    from gamma_quant.options.greeks import bs_gamma

    e = read_omon_xlsx(GREEKS_FILE)
    spot, q, _ = implied_spot_and_dividend(e.chain)
    ch = e.chain.copy()
    ch["underlying_price"] = spot
    ch["gamma"] = bs_gamma(
        spot, ch["strike"].to_numpy(float), ch["tau"].to_numpy(float),
        ch["risk_free_rate"].to_numpy(float),
        ch["implied_volatility"].to_numpy(float), q,
    )
    ch = ch[ch["gamma"].notna() & ch["open_interest"].notna()]

    res = compute_gex(ch, multiplier=100)
    check(np.isfinite(res.total), "el GEX se calcula sobre datos de Bloomberg")
    print(f"  [info] GEX (solo los {res.n_contracts} contratos del export): "
          f"{res.total:,.0f}")
    print("  [info] AVISO: es una MUESTRA de la cadena (~150 de ~28.000 contratos),")
    print("         no la cadena completa. Este total NO es el GEX de SPX.")
    check(True, "queda constancia de que el export es una muestra, no una cadena")


def main() -> int:
    print("=" * 78)
    print("LECTOR DE BLOOMBERG OMON Y CALIBRACION — gamma_quant")
    print("=" * 78)

    if not GREEKS_FILE.exists() and not MAY_FILE.exists():
        print(f"\nNo hay ficheros OMON en {OMON_DIR}. Suite omitida.")
        return 0

    test_parsing()
    test_historical_export()
    test_calibration()
    test_gex_from_bloomberg()

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"RESULTADO: {len(_FAILURES)} FALLOS de {_CHECKS} comprobaciones")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"RESULTADO: {_CHECKS} comprobaciones, todas correctas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
