#!/usr/bin/env python3
"""FASE 0.B -- Optimizacion del costo y decision de alcance.

Contexto: la Fase 0.A revelo que `statistics` (el open interest) cuesta $1.360/ano
para 32 subyacentes, 24x mas que las cotizaciones. Pero con las cotizaciones ya
aprendimos la leccion: pedir la sesion entera costaba $2.128 y pedir solo la
ventana de 5 min al cierre costo $56 -- 38x menos por el mismo dato util.

Este script averigua si el mismo truco aplica al open interest:
  1. A que hora del dia llegan realmente los registros de `statistics` (gratis,
     con get_record_count por franjas horarias).
  2. Cuanto costaria pedir solo esa franja.
  3. Que stat_types trae y cual es el timestamp real del open interest
     (una descarga minima de ~$0.20, la unica que cuesta algo).
  4. Proyeccion de costo para tres alcances posibles.

Tambien corrige el bug de la Fase 0.A: get_record_count exige end > start,
no admite start == end.

USO
    python3 fase0b_optimizacion.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports" / "fase0b_optimizacion.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
INDICES = ["SPY", "QQQ"]
TODOS = NUCLEO + DISPERSION + INDICES
PARENTS = [f"{s}.OPT" for s in TODOS]

OPT = "OPRA.PILLAR"
DAY = "2026-06-10"
DAY_END = "2026-06-11"
TRADING_DAYS = 250
PROBE_SYMBOL = "AAPL.OPT"

_lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _lines.append(s)


def load_key() -> str:
    k = os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("DATABENTO_API_KEY"):
                return line.split("=", 1)[1].strip()
    sys.exit("No encuentro la API key.")


def main() -> int:
    import databento as db
    c = db.Historical(load_key())

    say("=" * 84)
    say("FASE 0.B -- OPTIMIZACION DE COSTO Y ALCANCE".center(84))
    say(f"{datetime.now():%Y-%m-%d %H:%M}".center(84))
    say("=" * 84)

    # ------------------------------- 1. tamano real de las cadenas (bug corregido)
    say("\n" + "-" * 84)
    say(f"1. TAMANO REAL DE LAS CADENAS  (dia {DAY}, end corregido a {DAY_END})")
    say("-" * 84)
    total = 0
    try:
        total = c.metadata.get_record_count(
            dataset=OPT, symbols=PARENTS, stype_in="parent",
            schema="definition", start=DAY, end=DAY_END)
        say(f"    Contratos vivos, los {len(TODOS)} subyacentes: {total:,}")
        say(f"    Proyeccion ano completo: {total * TRADING_DAYS:,} filas contrato-dia")
    except Exception as ex:
        say(f"    ERROR: {type(ex).__name__}: {ex}")

    say("")
    say(f"    {'ticker':<8}{'contratos':>11}")
    for s in ["AAPL", "NVDA", "TSLA", "GME", "MSTR", "KO", "SPY", "QQQ"]:
        try:
            n = c.metadata.get_record_count(
                dataset=OPT, symbols=[f"{s}.OPT"], stype_in="parent",
                schema="definition", start=DAY, end=DAY_END)
            say(f"    {s:<8}{n:>11,}")
        except Exception as ex:
            say(f"    {s:<8}{'ERROR':>11}  {ex}")

    # ---------------------- 2. a que hora llegan los registros de `statistics`
    say("\n" + "-" * 84)
    say("2. DISTRIBUCION HORARIA DE `statistics`  (gratis, por franjas UTC)")
    say("-" * 84)
    say("    Objetivo: si el open interest llega en una franja estrecha, se puede")
    say("    pedir solo esa franja -- el mismo truco que bajo las cotizaciones 38x.")
    say("")
    say(f"    {'franja UTC':<18}{'registros':>12}{'% del dia':>12}   (ET aprox)")
    buckets, tot_rec = [], 0
    for h in range(0, 24, 2):
        s = f"{DAY}T{h:02d}:00:00"
        e = f"{DAY}T{h+2:02d}:00:00" if h < 22 else DAY_END
        try:
            n = c.metadata.get_record_count(
                dataset=OPT, symbols=[PROBE_SYMBOL], stype_in="parent",
                schema="statistics", start=s, end=e)
        except Exception:
            n = -1
        buckets.append((h, n)); tot_rec += max(n, 0)
    for h, n in buckets:
        pct = (n / tot_rec * 100) if tot_rec and n > 0 else 0.0
        et = (h - 4) % 24
        bar = "#" * int(pct / 3)
        say(f"    {h:02d}:00-{h+2:02d}:00 UTC {n:>12,}{pct:>11.1f}%   {et:02d}:00 ET {bar}")
    say(f"\n    Total de registros de statistics en el dia, solo AAPL: {tot_rec:,}")

    peak = max(buckets, key=lambda x: x[1]) if buckets else (0, 0)
    say(f"    Franja con mas registros: {peak[0]:02d}:00-{peak[0]+2:02d}:00 UTC")

    # ------------------------- 3. costo de pedir solo la franja util
    say("\n" + "-" * 84)
    say("3. COSTO DE `statistics` SEGUN LA VENTANA QUE SE PIDA")
    say("-" * 84)
    say(f"    {'ventana':<40}{'costo/dia':>12}{'x250 dias':>14}")
    say(f"    {'-'*40}{'-'*12}{'-'*14}")

    def cost_win(label, s, e):
        try:
            v = c.metadata.get_cost(dataset=OPT, symbols=PARENTS, stype_in="parent",
                                    schema="statistics", start=s, end=e,
                                    mode="historical-streaming")
            say(f"    {label:<40}{'$'+format(v,',.2f'):>12}{'$'+format(v*TRADING_DAYS,',.2f'):>14}")
            return v
        except Exception as ex:
            say(f"    {label:<40}{'ERROR':>12}   {type(ex).__name__}")
            return None

    full = cost_win("dia completo (lo que costo en Fase 0.A)", DAY, DAY_END)
    windows = [("solo la franja pico (2h)", f"{DAY}T{peak[0]:02d}:00:00", f"{DAY}T{peak[0]+2:02d}:00:00"),
               ("pre-apertura 08:00-13:30 UTC", f"{DAY}T08:00:00", f"{DAY}T13:30:00"),
               ("cierre 19:55-20:00 UTC", f"{DAY}T19:55:00", f"{DAY}T20:00:00"),
               ("post-cierre 20:00-24:00 UTC", f"{DAY}T20:00:00", DAY_END)]
    best = None
    for lab, s, e in windows:
        v = cost_win(lab, s, e)
        if v is not None and (best is None or v < best[1]):
            best = (lab, v)
    if full and best:
        say("")
        say(f"    >>> Ventana mas barata: {best[0]} -- ${best[1]*TRADING_DAYS:,.2f}/ano")
        say(f"    >>> Ahorro vs dia completo: {full/best[1]:.0f}x  "
            f"(${full*TRADING_DAYS:,.0f} -> ${best[1]*TRADING_DAYS:,.0f})")
        say("    >>> OJO: barato no sirve si esa franja no contiene el open interest.")
        say("        El bloque 4 confirma que stat_type=9 esta realmente ahi.")

    # ------------- 4. que trae statistics de verdad (unica descarga que cuesta)
    say("\n" + "-" * 84)
    say("4. CONTENIDO REAL DE `statistics`  (descarga minima, ~$0.20)")
    say("-" * 84)
    resp = input("    Descargar 1 simbolo x 1 dia para inspeccionar? [s/N]: ").strip().lower()
    if resp != "s":
        say("    Omitido por el usuario.")
    else:
        try:
            data = c.timeseries.get_range(
                dataset=OPT, symbols=[PROBE_SYMBOL], stype_in="parent",
                schema="statistics", start=DAY, end=DAY_END)
            df = data.to_df()
            say(f"    Registros descargados: {len(df):,}")
            say(f"    Columnas: {', '.join(df.columns[:14])}")
            if "stat_type" in df.columns:
                say("")
                say(f"    {'stat_type':>10}{'registros':>12}   {'primer ts':<28}{'ultimo ts':<28}")
                for st, g in df.groupby("stat_type"):
                    ts = g.index
                    nombre = " <-- OPEN INTEREST" if st == 9 else ""
                    say(f"    {st:>10}{len(g):>12}   {str(ts.min())[:26]:<28}{str(ts.max())[:26]:<28}{nombre}")
                if 9 in set(df["stat_type"]):
                    oi = df[df["stat_type"] == 9]
                    say("")
                    say(f"    >>> OPEN INTEREST CONFIRMADO: {len(oi):,} registros")
                    say(f"    >>> Ventana horaria real: {str(oi.index.min())[:26]} -> {str(oi.index.max())[:26]}")
                    say(f"    >>> Contratos distintos con OI: {oi['instrument_id'].nunique():,}")
                    col = "quantity" if "quantity" in oi.columns else ("price" if "price" in oi.columns else None)
                    if col:
                        say(f"    >>> OI: min={oi[col].min():,.0f}  mediana={oi[col].median():,.0f}  max={oi[col].max():,.0f}")
                else:
                    say("\n    >>> stat_type=9 NO aparece en este dia. Revisar antes de seguir.")
        except Exception as ex:
            say(f"    ERROR en la descarga: {type(ex).__name__}: {ex}")

    # ---------------------------------------- 5. proyeccion por alcance
    say("\n" + "-" * 84)
    say("5. PROYECCION DE COSTO POR ALCANCE")
    say("-" * 84)
    say("    OPRA tiene historia desde 2013-04. El limite real para el build")
    say("    completo son los datos de EQUITIES: XNAS.ITCH y XNYS.PILLAR")
    say("    arrancan en 2018-05, EQUS.SUMMARY solo en 2024-07.")
    say("")
    st_year = (best[1] * TRADING_DAYS) if best else 1360.24
    say(f"    {'alcance':<34}{'meses':>8}{'definition':>13}{'statistics':>13}{'quotes':>11}{'TOTAL':>12}")
    say(f"    {'-'*34}{'-'*8}{'-'*13}{'-'*13}{'-'*11}{'-'*12}")
    for lab, yrs, meses in [("Piloto: 1 ano", 1.0, 12),
                            ("Medio: 2024-07 -> hoy", 2.2, 26),
                            ("Completo: 2018-05 -> hoy", 8.3, 100)]:
        d, s_, q = 44.28 * yrs, st_year * yrs, 56.42 * yrs
        say(f"    {lab:<34}{meses:>8}{'$'+format(d,',.0f'):>13}{'$'+format(s_,',.0f'):>13}"
            f"{'$'+format(q,',.0f'):>11}{'$'+format(d+s_+q,',.0f'):>12}")
    say("")
    say("    (statistics proyectado con la ventana optimizada; si el bloque 4 muestra")
    say("     que el OI no cae en esa ventana, hay que reproyectar con el dia completo)")

    say("\n" + "=" * 84)
    say(f"Salida guardada en: {OUT}")
    say("=" * 84)
    OUT.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
