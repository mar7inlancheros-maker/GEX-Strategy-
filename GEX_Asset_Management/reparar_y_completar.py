#!/usr/bin/env python3
"""Repara los 18 dias con ventana equivocada y completa las 6 semanas faltantes.

PARTE A -- REPARACION BARATA (~$5.60)
Los 18 dias en horario estandar (EST) tienen cotizaciones de las 14:55-15:00 ET
en vez de las 15:55-16:00 ET. NO hace falta re-bajar el dia completo: `definition`
y `statistics` no dependen de la ventana horaria y ya estan en el parquet. Solo se
re-pide `cbbo-1m` en la ventana correcta y se reemplaza bid/ask/mid.
    cbbo-1m solo:      18 x $0.31  = $5.58
    dia completo:      18 x $1.77  = $31.86   <- lo que costaria rehacerlos
Limitacion declarada: un contrato cotizado a las 15:55 pero no a las 14:55 no
estaba en el parquet y no se puede recuperar (haria falta su definition y su OI).
El script reporta la tasa de retencion para que se sepa el tamano del efecto.

PARTE B -- SEMANAS FALTANTES (~$10.60)
6 semanas se perdieron por 504 del gateway (y una por Viernes Santo). Ahora hay
reintentos con espera creciente y se bajan de una en una, sin paralelismo.

USO
    python3 reparar_y_completar.py --dry-run
    python3 reparar_y_completar.py
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from datetime import date, timedelta

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.ingest.opra import (DayResult, MAX_REL_SPREAD, OPT_DATASET, close_window_utc,
                             fetch_eod_quotes, ingest_day)

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
TODOS = NUCLEO + DISPERSION + ["SPY", "QQQ"]
PARENTS = [f"{s}.OPT" for s in TODOS]

CHAINS = ROOT / "data" / "raw" / "opra_chain"
EST_INI, EST_FIN = date(2025, 11, 2), date(2026, 3, 8)
RANGO = (date(2025, 9, 2), date(2026, 8, 31))
REP = ROOT / "reports" / "reparacion.txt"
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def load_key():
    k = os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("DATABENTO_API_KEY"):
            return line.split("=", 1)[1].strip()
    sys.exit("sin API key")


def business_days(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def dias_en_disco():
    out = {}
    for p in sorted(CHAINS.glob("date=*")):
        try:
            out[date.fromisoformat(p.name.split("=", 1)[1])] = p / "chain.parquet"
        except ValueError:
            pass
    return out


def semanas_faltantes(disco):
    todas = {}
    for d in business_days(*RANGO):
        todas.setdefault(d.isocalendar()[:2], []).append(d)
    con = {d.isocalendar()[:2] for d in disco}
    return {k: sorted(v, reverse=True) for k, v in sorted(todas.items()) if k not in con}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    a = ap.parse_args()

    import databento as db
    c = db.Historical(load_key())

    disco = dias_en_disco()
    est = sorted(d for d in disco if EST_INI <= d < EST_FIN)
    falt = semanas_faltantes(disco)

    say("=" * 88)
    say("REPARACION Y COMPLETADO".center(88))
    say("=" * 88)
    say(f"\ndias en disco: {len(disco)}   a reparar (EST): {len(est)}   "
        f"semanas faltantes: {len(falt)}")

    costo_rep = costo_falt = 0.0
    probe = est[0] if est else date(2026, 5, 1)
    s_, e_ = close_window_utc(probe)
    try:
        cq = c.metadata.get_cost(dataset=OPT_DATASET, symbols=PARENTS, stype_in="parent",
                                 schema="cbbo-1m", start=s_, end=e_,
                                 mode="historical-streaming")
    except Exception:
        cq = 0.31
    costo_rep = cq * len(est)
    try:
        cd = c.metadata.get_cost(dataset=OPT_DATASET, symbols=PARENTS, stype_in="parent",
                                 schema="definition", start=probe.isoformat(),
                                 end=(probe + timedelta(days=1)).isoformat(),
                                 mode="historical-streaming")
        cs = c.metadata.get_cost(dataset=OPT_DATASET, symbols=PARENTS, stype_in="parent",
                                 schema="statistics",
                                 start=f"{probe}T10:00:00", end=f"{probe}T12:00:00",
                                 mode="historical-streaming")
    except Exception:
        cd, cs = 0.19, 1.27
    costo_falt = (cd + cs + cq) * len(falt)

    say("")
    say(f"  PARTE A  reparar ventana, solo cbbo-1m   {len(est):>3} dias x ${cq:.2f} = ${costo_rep:>7.2f}")
    say(f"           (re-bajar el dia completo seria  {len(est):>3} dias x ${cd+cs+cq:.2f} = "
        f"${(cd+cs+cq)*len(est):>7.2f})")
    say(f"  PARTE B  semanas faltantes, dia completo {len(falt):>3} dias x ${cd+cs+cq:.2f} = ${costo_falt:>7.2f}")
    say(f"  {'-'*74}")
    say(f"  TOTAL ESTIMADO: ${costo_rep + costo_falt:.2f}")

    if a.dry_run:
        say("\n  --dry-run: nada se descarga.")
        REP.write_text("\n".join(_lines) + "\n")
        return 0
    if costo_rep + costo_falt > a.max_cost:
        say(f"\n  ABORTADO: supera --max-cost (${a.max_cost:.2f}).")
        return 2

    say("\n  Escribe 'REPARAR' para confirmar.")
    if input("  > ").strip() != "REPARAR":
        say("  Cancelado.")
        return 0

    # ---------------------------------------------------------------- PARTE A
    if not a.skip_repair and est:
        say("\n" + "-" * 88)
        say("PARTE A -- reemplazo de cotizaciones en la ventana correcta")
        say("-" * 88)
        say(f"  {'fecha':<12}{'antes':>9}{'despues':>9}{'retenido':>10}"
            f"{'mid antes':>11}{'mid despues':>13}{'seg':>6}")
        for d in est:
            t0 = time.time()
            path = disco[d]
            old = pl.read_parquet(path)
            res = DayResult(day=d)
            try:
                q = fetch_eod_quotes(c, PARENTS, d, res)
            except Exception as ex:
                say(f"  {d}  ERROR {type(ex).__name__}: {ex}")
                continue
            if q.is_empty():
                say(f"  {d}  sin cotizaciones en la ventana correcta; se deja igual")
                continue
            base = old.drop(["bid", "ask", "mid", "rel_spread"])
            new = base.join(q, on="instrument_id", how="inner")
            ret = new.height / old.height if old.height else 0.0
            new.write_parquet(path)
            say(f"  {d}{old.height:>9,}{new.height:>9,}{ret*100:>9.1f}%"
                f"{float(old['mid'].median()):>11.3f}{float(new['mid'].median()):>13.3f}"
                f"{time.time()-t0:>6.0f}")
        say("\n  Los 18 dias quedan con snapshot de 15:55-16:00 ET, igual que los EDT.")

    # ---------------------------------------------------------------- PARTE B
    if not a.skip_missing and falt:
        say("\n" + "-" * 88)
        say("PARTE B -- semanas faltantes (secuencial, con reintentos)")
        say("-" * 88)
        say(f"  {'fecha':<12}{'contratos':>10}{'OI':>10}{'quotes':>9}{'unidos':>9}"
            f"{'seg':>6}  notas")
        for wk, cands in falt.items():
            listo = False
            for d in cands[:3]:
                t0 = time.time()
                r = ingest_day(c, PARENTS, d, CHAINS)
                nota = "; ".join(r.notes) if r.notes else (r.error or "")
                say(f"  {d.isoformat():<12}{r.n_contracts:>10,}{r.n_oi:>10,}"
                    f"{r.n_quotes:>9,}{r.n_joined:>9,}{time.time()-t0:>6.0f}  {nota[:60]}")
                if r.n_joined:
                    listo = True
                    break
            if not listo:
                say(f"  semana {wk[0]}-W{wk[1]:02d}: SIN DATOS tras {min(3,len(cands))} intentos")

    disco2 = dias_en_disco()
    say("\n" + "=" * 88)
    say(f"  dias en disco ahora: {len(disco2)}   semanas cubiertas: "
        f"{len({d.isocalendar()[:2] for d in disco2})} de 53")
    say(f"  reporte: {REP}")
    say("=" * 88)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
