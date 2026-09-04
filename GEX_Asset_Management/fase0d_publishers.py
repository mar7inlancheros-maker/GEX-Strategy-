#!/usr/bin/env python3
"""FASE 0.D -- La trampa de los publishers, y el modo batch.

HALLAZGO QUE MOTIVA ESTE SCRIPT
La Fase 0.C reporto, para AAPL en un dia:
    65.700 registros de open_interest  sobre  3.650 contratos distintos
    65.700 / 3.650 = 18.0 registros por contrato, exactamente
    suma de todos los OI = 97.290.396
OPRA es un feed CONSOLIDADO: cada bolsa de opciones que lista un contrato
disemina su propio mensaje. Hay ~18 bolsas de opciones en EE.UU. El open
interest real de un contrato es UNO (lo liquida la OCC), no dieciocho.

Si sumaramos los 18 registros, el OI quedaria ~18x inflado. 97.290.396 / 18 =
5.405.022, que es un OI total plausible para AAPL. Sumar seria un error.

Y hay un peligro peor que el factor 18 uniforme: los contratos poco liquidos se
listan en MENOS bolsas que los liquidos. Es decir, el multiplicador varia por
contrato -- lo que no inflaria Gamma de forma uniforme, sino que DISTORSIONARIA
LA COMPARACION CROSS-SECCIONAL, que es exactamente lo que la estrategia ordena.

Este script determina la regla de deduplicacion correcta:
  1. Cuantos publishers por contrato, y como se distribuye.
  2. Coinciden todos los publishers en el mismo valor de OI, o cada uno reporta
     su propia porcion? (decide entre deduplicar y sumar)
  3. Que significa ts_ref frente a ts_event (confirma el desfase de un dia).
  4. Y aparte: cuanto cuesta en modo batch en vez de streaming.

USO
    python3 fase0d_publishers.py
Costo: ~$0.05 (un simbolo, un dia, solo la ventana de OI).
"""
from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports" / "fase0d_publishers.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
INDICES = ["SPY", "QQQ"]
PARENTS = [f"{s}.OPT" for s in NUCLEO + DISPERSION + INDICES]

OPT = "OPRA.PILLAR"
DAY, DAY_END = "2026-06-10", "2026-06-11"
OI_START, OI_END = f"{DAY}T10:00:00", f"{DAY}T12:00:00"
TRADING_DAYS = 250

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
    import numpy as np
    c = db.Historical(load_key())

    say("=" * 90)
    say("FASE 0.D -- DEDUPLICACION DE PUBLISHERS Y MODO BATCH".center(90))
    say(f"{datetime.now():%Y-%m-%d %H:%M}".center(90))
    say("=" * 90)

    # -------------------------------------------- 1. streaming vs batch
    say("\n" + "-" * 90)
    say("1. COSTO: STREAMING vs BATCH  (gratis)")
    say("-" * 90)
    say(f"    {'schema / ventana':<42}{'streaming':>14}{'batch':>14}{'ahorro':>10}")
    say(f"    {'-'*42}{'-'*14}{'-'*14}{'-'*10}")
    pruebas = [("statistics, ventana OI (10-12 UTC)", "statistics", OI_START, OI_END),
               ("definition, dia completo", "definition", DAY, DAY_END),
               ("cbbo-1m, cierre 19:55-20:00 UTC", "cbbo-1m", f"{DAY}T19:55:00", f"{DAY}T20:00:00")]
    costos = {}
    for lab, sch, s, e in pruebas:
        vals = {}
        for modo in ("historical-streaming", "historical"):
            try:
                vals[modo] = c.metadata.get_cost(dataset=OPT, symbols=PARENTS,
                                                 stype_in="parent", schema=sch,
                                                 start=s, end=e, mode=modo)
            except Exception:
                vals[modo] = None
        st, ba = vals.get("historical-streaming"), vals.get("historical")
        ratio = f"{st/ba:.2f}x" if (st and ba and ba > 0) else "-"
        say(f"    {lab:<42}{('$'+format(st,',.2f')) if st else 'ERROR':>14}"
            f"{('$'+format(ba,',.2f')) if ba else 'ERROR':>14}{ratio:>10}")
        costos[sch] = (st, ba)

    # -------------------------------- 2. estructura de publishers en el OI
    say("\n" + "-" * 90)
    say("2. ESTRUCTURA DE PUBLISHERS EN LOS REGISTROS DE OPEN INTEREST")
    say("-" * 90)
    say("    Se descarga AAPL, 1 dia, solo la ventana de OI. Costo ~$0.05.")
    if input("    Continuar? [s/N]: ").strip().lower() != "s":
        say("    Cancelado.")
        OUT.write_text("\n".join(_lines) + "\n")
        return 0

    df = c.timeseries.get_range(dataset=OPT, symbols=["AAPL.OPT"], stype_in="parent",
                                schema="statistics", start=OI_START, end=OI_END).to_df()
    df = df[df["stat_type"].astype(int) == 9].copy()
    qcol = next((x for x in ("quantity", "price", "value") if x in df.columns), None)
    say(f"\n    Registros de OI: {len(df):,}   contratos: {df['instrument_id'].nunique():,}"
        f"   publishers distintos: {df['publisher_id'].nunique()}")
    say(f"    Columna de cantidad usada: {qcol}")

    say("")
    say("    a) Registros por publisher")
    say(f"       {'publisher_id':>14}{'registros':>11}{'contratos':>11}{'OI total':>16}")
    for pid, g in df.groupby("publisher_id"):
        tot = g[qcol].sum() if qcol else 0
        say(f"       {pid:>14}{len(g):>11,}{g['instrument_id'].nunique():>11,}{tot:>16,.0f}")

    say("")
    say("    b) Cuantos publishers reportan cada contrato")
    pc = df.groupby("instrument_id")["publisher_id"].nunique()
    say(f"       {'publishers':>12}{'contratos':>12}{'% del total':>13}")
    for k, v in pc.value_counts().sort_index().items():
        say(f"       {k:>12}{v:>12,}{v/len(pc)*100:>12.1f}%")
    say(f"       min={pc.min()}  mediana={pc.median():.0f}  max={pc.max()}")
    if pc.min() != pc.max():
        say("       >>> EL NUMERO DE PUBLISHERS VARIA POR CONTRATO.")
        say("       >>> Sumar sin deduplicar distorsionaria el cross-section, no solo la escala.")

    say("")
    say("    c) Coinciden los publishers en el valor de OI del mismo contrato?")
    if qcol:
        g = df.groupby("instrument_id")[qcol]
        nun = g.nunique()
        say(f"       contratos donde TODOS los publishers dan el mismo OI: "
            f"{(nun == 1).sum():,} de {len(nun):,}  ({(nun==1).mean()*100:.1f}%)")
        say(f"       contratos con valores discrepantes: {(nun > 1).sum():,}")
        if (nun > 1).any():
            ej = nun[nun > 1].index[:3]
            for iid in ej:
                sub = df[df["instrument_id"] == iid][["publisher_id", qcol]]
                say(f"         ejemplo instrument_id={iid}: " +
                    ", ".join(f"pub{int(r.publisher_id)}={int(getattr(r, qcol)):,}"
                              for r in sub.itertuples()))
        say("")
        dedup = df.drop_duplicates(subset=["instrument_id"])[qcol].sum()
        suma = df[qcol].sum()
        say(f"       OI total SUMANDO todo:        {suma:>16,.0f}")
        say(f"       OI total DEDUPLICANDO:        {dedup:>16,.0f}")
        say(f"       factor de inflacion:          {suma/dedup:>16.2f}x")
        say("")
        if (nun == 1).mean() > 0.95:
            say("       >>> VEREDICTO: los publishers replican el MISMO OI.")
            say("       >>> REGLA: deduplicar por instrument_id. NUNCA sumar.")
        else:
            say("       >>> VEREDICTO: hay discrepancias entre publishers.")
            say("       >>> Revisar antes de fijar la regla: puede requerir tomar el maximo")
            say("       >>> o el publisher primario, no un drop_duplicates ciego.")

    # ------------------------------------ 3. ts_ref vs ts_event (desfase)
    say("\n" + "-" * 90)
    say("3. ts_ref vs ts_event  (confirma el desfase de un dia del OI)")
    say("-" * 90)
    say(f"    ts_event (llegada):  {str(df.index.min())[:19]} -> {str(df.index.max())[:19]}")
    if "ts_ref" in df.columns:
        try:
            tr = df["ts_ref"]
            say(f"    ts_ref (referencia): {str(tr.min())[:19]} -> {str(tr.max())[:19]}")
            say(f"    valores distintos de ts_ref: {tr.nunique()}")
            say("")
            say("    >>> Si ts_ref apunta al dia habil ANTERIOR, el desfase de un dia")
            say("    >>> queda confirmado por los datos y no por inferencia.")
        except Exception as ex:
            say(f"    no se pudo interpretar ts_ref: {ex}")
    else:
        say("    (no hay columna ts_ref en este schema)")

    # ------------------------------------------------ 4. costo final
    say("\n" + "=" * 90)
    say("4. COSTO FINAL POR ALCANCE  (con el modo mas barato)")
    say("=" * 90)
    st_s, st_b = costos.get("statistics", (None, None))
    df_s, df_b = costos.get("definition", (None, None))
    q_s, q_b = costos.get("cbbo-1m", (None, None))
    st = min(x for x in (st_s, st_b) if x) if (st_s or st_b) else 1.38
    dfc = min(x for x in (df_s, df_b) if x) if (df_s or df_b) else 0.18
    q = min(x for x in (q_s, q_b) if x) if (q_s or q_b) else 0.23
    say(f"    Costos diarios usados: statistics ${st:.2f}  definition ${dfc:.2f}  quotes ${q:.2f}")
    say("")
    say(f"    {'alcance':<32}{'meses':>7}{'definition':>12}{'statistics':>12}{'quotes':>10}{'TOTAL':>12}")
    say(f"    {'-'*32}{'-'*7}{'-'*12}{'-'*12}{'-'*10}{'-'*12}")
    for lab, yrs, meses in [("Piloto: 1 ano", 1.0, 12),
                            ("Medio: 2024-07 -> hoy", 2.2, 26),
                            ("Completo: 2018-05 -> hoy", 8.3, 100)]:
        d_, s_, q_ = dfc*TRADING_DAYS*yrs, st*TRADING_DAYS*yrs, q*TRADING_DAYS*yrs
        say(f"    {lab:<32}{meses:>7}{'$'+format(d_,',.0f'):>12}{'$'+format(s_,',.0f'):>12}"
            f"{'$'+format(q_,',.0f'):>10}{'$'+format(d_+s_+q_,',.0f'):>12}")
    say("")
    say(f"    Salida guardada en: {OUT}")
    say("=" * 90)
    OUT.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
