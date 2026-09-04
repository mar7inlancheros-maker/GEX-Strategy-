#!/usr/bin/env python3
"""FASE 0.C -- Donde vive el open interest, y cuanto cuesta de verdad.

La Fase 0.B mostro que los registros de `statistics` de OPRA se concentran en
DOS franjas y nada mas:
    10:00-12:00 UTC (06:00-08:00 ET)  22.3% de los registros
    20:00-22:00 UTC (16:00-18:00 ET)  77.7% de los registros
Hipotesis: la franja de la manana trae el OPEN INTEREST (OPRA lo disemina antes
de la apertura, reflejando el cierre de la sesion anterior) y la de la tarde trae
precios de cierre, settlement, maximos/minimos de sesion y net change.

Si la hipotesis es correcta, se pide solo la franja de la manana y el costo del
open interest cae ~4x. Este script lo confirma descargando las DOS franjas para
un simbolo y un dia, y contando que stat_types hay en cada una.

Corrige tambien el bug de la Fase 0.B: elegir la ventana mas barata sin verificar
que contenga registros. Una ventana de $0.00 no es barata, esta vacia.

USO
    python3 fase0c_open_interest.py
Costo: ~$0.10-0.30 (dos descargas de 1 simbolo x 1 dia). Pide confirmacion.
"""
from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports" / "fase0c_open_interest.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
INDICES = ["SPY", "QQQ"]
TODOS = NUCLEO + DISPERSION + INDICES
PARENTS = [f"{s}.OPT" for s in TODOS]

OPT = "OPRA.PILLAR"
DAY, DAY_END = "2026-06-10", "2026-06-11"
TRADING_DAYS = 250
PROBE = "AAPL.OPT"

VENTANAS = [
    ("manana  10:00-12:00 UTC (06-08 ET)", f"{DAY}T10:00:00", f"{DAY}T12:00:00"),
    ("tarde   20:00-22:00 UTC (16-18 ET)", f"{DAY}T20:00:00", f"{DAY}T22:00:00"),
]

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


def stat_name(v):
    try:
        from databento_dbn import StatType
        return StatType(int(v)).name.lower()
    except Exception:
        return {1: "opening_price", 2: "indicative_opening_price", 3: "settlement_price",
                4: "session_low_price", 5: "session_high_price", 6: "cleared_volume",
                7: "lowest_offer", 8: "highest_bid", 9: "OPEN_INTEREST",
                10: "fixing_price", 11: "cleared_price", 12: "net_change",
                13: "vwap", 14: "volume_weighted_price"}.get(int(v), f"tipo_{v}")


def main() -> int:
    import databento as db
    c = db.Historical(load_key())

    say("=" * 88)
    say("FASE 0.C -- LOCALIZACION DEL OPEN INTEREST".center(88))
    say(f"{datetime.now():%Y-%m-%d %H:%M}  ·  dia de muestra {DAY}".center(88))
    say("=" * 88)

    # ---------------- 1. costo de cada ventana, verificando que tenga registros
    say("\n" + "-" * 88)
    say("1. COSTO POR VENTANA  (con guarda: una ventana vacia no es barata)")
    say("-" * 88)
    say(f"    {'ventana':<38}{'registros/dia':>15}{'costo/dia':>12}{'x250 dias':>14}")
    say(f"    {'-'*38}{'-'*15}{'-'*12}{'-'*14}")
    opciones = []
    todas = [("dia completo", DAY, DAY_END)] + VENTANAS
    for lab, s, e in todas:
        try:
            n = c.metadata.get_record_count(dataset=OPT, symbols=PARENTS,
                                            stype_in="parent", schema="statistics",
                                            start=s, end=e)
        except Exception:
            n = -1
        try:
            v = c.metadata.get_cost(dataset=OPT, symbols=PARENTS, stype_in="parent",
                                    schema="statistics", start=s, end=e,
                                    mode="historical-streaming")
        except Exception as ex:
            say(f"    {lab:<38}{n:>15,}{'ERROR':>12}  {type(ex).__name__}")
            continue
        say(f"    {lab:<38}{n:>15,}{'$'+format(v,',.2f'):>12}{'$'+format(v*TRADING_DAYS,',.2f'):>14}")
        if n > 0:                                  # GUARDA: solo ventanas con datos
            opciones.append((lab, s, e, n, v))

    # ------------------------------- 2. que stat_types hay en cada ventana
    say("\n" + "-" * 88)
    say("2. CONTENIDO REAL DE CADA VENTANA  (descarga minima)")
    say("-" * 88)
    est = sum(v for _, _, _, _, v in opciones if _ ) if False else None
    say(f"    Se descargan las 2 ventanas para {PROBE}, 1 dia. Costo estimado ~$0.10-0.30.")
    if input("    Continuar? [s/N]: ").strip().lower() != "s":
        say("    Cancelado por el usuario.")
        OUT.write_text("\n".join(_lines) + "\n")
        return 0

    hallazgo = {}
    for lab, s, e in VENTANAS:
        say("")
        say(f"    >>> {lab}")
        try:
            df = c.timeseries.get_range(dataset=OPT, symbols=[PROBE], stype_in="parent",
                                        schema="statistics", start=s, end=e).to_df()
        except Exception as ex:
            say(f"        ERROR: {type(ex).__name__}: {ex}")
            continue
        if df.empty:
            say("        sin registros")
            continue
        say(f"        registros: {len(df):,}   columnas: {', '.join(list(df.columns)[:12])}")
        if "stat_type" not in df.columns:
            say("        (no hay columna stat_type)")
            continue
        say(f"        {'stat_type':>10}  {'nombre':<26}{'registros':>11}{'contratos':>11}   {'primer ts':<22}")
        for st, g in df.groupby("stat_type"):
            nid = g["instrument_id"].nunique() if "instrument_id" in g.columns else 0
            marca = "  <== OPEN INTEREST" if int(st) == 9 else ""
            say(f"        {int(st):>10}  {stat_name(st):<26}{len(g):>11,}{nid:>11,}"
                f"   {str(g.index.min())[:19]:<22}{marca}")
        if 9 in set(int(x) for x in df["stat_type"].unique()):
            oi = df[df["stat_type"].astype(int) == 9]
            col = next((x for x in ("quantity", "price", "value") if x in oi.columns), None)
            hallazgo[lab] = (len(oi), oi["instrument_id"].nunique() if "instrument_id" in oi.columns else 0)
            say("")
            say(f"        OPEN INTEREST ENCONTRADO AQUI: {len(oi):,} registros, "
                f"{hallazgo[lab][1]:,} contratos distintos")
            say(f"        ventana horaria real: {str(oi.index.min())[:19]} -> {str(oi.index.max())[:19]}")
            if col:
                say(f"        OI: min={oi[col].min():,.0f}  mediana={oi[col].median():,.0f}  "
                    f"max={oi[col].max():,.0f}  suma={oi[col].sum():,.0f}")

    # ------------------------------------------- 3. veredicto y reproyeccion
    say("\n" + "=" * 88)
    say("3. VEREDICTO Y COSTO REPROYECTADO")
    say("=" * 88)
    if not hallazgo:
        say("    stat_type=9 NO aparecio en ninguna de las dos ventanas.")
        say("    Hay que revisar el dia de muestra o pedir el dia completo. NO optimizar a ciegas.")
        OUT.write_text("\n".join(_lines) + "\n")
        return 1

    ganadora = max(hallazgo.items(), key=lambda kv: kv[1][0])
    say(f"    El open interest vive en: {ganadora[0]}")
    say(f"    ({ganadora[1][0]:,} registros sobre {ganadora[1][1]:,} contratos de AAPL)")
    sel = next((o for o in opciones if o[0] == ganadora[0]), None)
    full = next((o for o in opciones if o[0] == "dia completo"), None)
    if sel and full and sel[4] > 0:
        say("")
        say(f"    Costo con la ventana correcta: ${sel[4]*TRADING_DAYS:,.2f}/ano")
        say(f"    Costo pidiendo el dia completo: ${full[4]*TRADING_DAYS:,.2f}/ano")
        say(f"    Ahorro: {full[4]/sel[4]:.1f}x")
        say("")
        st_year = sel[4] * TRADING_DAYS
        say(f"    {'alcance':<32}{'meses':>7}{'definition':>12}{'statistics':>12}{'quotes':>10}{'TOTAL':>12}")
        say(f"    {'-'*32}{'-'*7}{'-'*12}{'-'*12}{'-'*10}{'-'*12}")
        for lab, yrs, meses in [("Piloto: 1 ano", 1.0, 12),
                                ("Medio: 2024-07 -> hoy", 2.2, 26),
                                ("Completo: 2018-05 -> hoy", 8.3, 100)]:
            d, s_, q = 44.28 * yrs, st_year * yrs, 56.42 * yrs
            say(f"    {lab:<32}{meses:>7}{'$'+format(d,',.0f'):>12}{'$'+format(s_,',.0f'):>12}"
                f"{'$'+format(q,',.0f'):>10}{'$'+format(d+s_+q,',.0f'):>12}")
    say("")
    say(f"    Salida guardada en: {OUT}")
    say("=" * 88)
    OUT.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
