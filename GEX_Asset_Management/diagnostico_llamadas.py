#!/usr/bin/env python3
"""Diagnostico: cual de las tres llamadas se atasca, y cuanto tarda cada una.

Imprime ANTES y DESPUES de cada paso, con flush, para que se vea el avance en
vivo. Primero con un solo simbolo (rapido y ~$0.10), luego opcionalmente con los
32 subyacentes para medir el escalado real.

USO
    python3 diagnostico_llamadas.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parent
DAY = "2025-09-02"
NEXT = "2025-09-03"

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
TODOS = NUCLEO + DISPERSION + ["SPY", "QQQ"]


def p(s=""):
    print(s, flush=True)


def load_key():
    k = os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if line.strip().startswith("DATABENTO_API_KEY"):
            return line.split("=", 1)[1].strip()
    sys.exit("sin API key")


PASOS = [
    ("definition", DAY,                    NEXT,                 "dia completo"),
    ("statistics", f"{DAY}T10:00:00",       f"{DAY}T12:00:00",    "ventana OI"),
    ("cbbo-1m",    f"{DAY}T19:55:00",       f"{DAY}T20:00:00",    "ventana cierre"),
]


def probar(c, symbols, etiqueta):
    p("")
    p("=" * 78)
    p(f"{etiqueta}  ({len(symbols)} simbolo(s))")
    p("=" * 78)
    for sch, s, e, win in PASOS:
        p("")
        p(f"  --> {sch:<12} {win:<16} pidiendo...")
        t0 = time.time()
        try:
            store = c.timeseries.get_range(
                dataset="OPRA.PILLAR", symbols=symbols, stype_in="parent",
                schema=sch, start=s, end=e)
            t_net = time.time() - t0
            p(f"      descarga terminada en {t_net:>7.1f}s")
            t1 = time.time()
            df = store.to_df()
            t_df = time.time() - t1
            p(f"      to_df() en           {t_df:>7.1f}s   filas: {len(df):,}")
            if sch == "definition":
                p(f"      columnas: {', '.join(list(df.columns)[:14])}")
            p(f"      TOTAL {sch}: {t_net + t_df:.1f}s")
        except KeyboardInterrupt:
            p("      INTERRUMPIDO por el usuario en este paso")
            raise
        except Exception as ex:
            p(f"      ERROR tras {time.time()-t0:.1f}s: {type(ex).__name__}: {ex}")


def main():
    import databento as db
    c = db.Historical(load_key())
    p(f"cliente listo · dia de prueba {DAY}")
    p("Se imprime antes y despues de cada llamada. Si algo se queda quieto,")
    p("la ultima linea impresa dice exactamente donde.")

    probar(c, ["AAPL.OPT"], "PRUEBA 1: un solo subyacente (~$0.10)")

    p("")
    p("=" * 78)
    r = input("Probar ahora con los 32 subyacentes? Cuesta ~$1.70. [s/N]: ").strip().lower()
    if r == "s":
        probar(c, [f"{x}.OPT" for x in TODOS], "PRUEBA 2: los 32 subyacentes")
    else:
        p("Omitido. Con la prueba 1 ya se puede extrapolar.")
    p("")
    p("Pegale esta salida completa a Claude.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        p("\ninterrumpido")
