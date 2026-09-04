#!/usr/bin/env python3
"""FASE 1 -- Ingesta del piloto.  Baja OPRA + equities y arma el lakehouse.

USO
    python3 run_ingesta.py --dry-run          # solo muestra el costo, no baja nada
    python3 run_ingesta.py --scope pilot      # 1 ano   (~$451)
    python3 run_ingesta.py --scope medium     # 26 meses (~$992)
    python3 run_ingesta.py --scope full       # 100 meses (~$3.743)

Es REANUDABLE: cada dia se guarda en su propio parquet y los dias ya bajados se
saltan. Si se corta la conexion, vuelve a correr el mismo comando.

Pide confirmacion explicita del costo antes de descargar nada.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from datetime import date, datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from concurrent.futures import ThreadPoolExecutor, as_completed

from gex.ingest.opra import (OPT_DATASET, OI_WIN, QUOTE_WIN, DayResult,
                             dataset_condition, fetch_equities_daily, ingest_day)

NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
INDICES = ["SPY", "QQQ"]
TODOS = NUCLEO + DISPERSION + INDICES
PARENTS = [f"{s}.OPT" for s in TODOS]

SCOPES = {
    "pilot":     (date(2025, 9, 2), date(2026, 8, 31), "1 ano (piloto original)"),
    "extension": (date(2024, 9, 1), date(2025, 8, 31),
                  "1 ano adicional hacia atras -- validado sin overlap ni hueco "
                  "con 'pilot': ultima semana 2025-08-29, primera de pilot 2025-09-05"),
    "extension2": (date(2023, 9, 1), date(2024, 8, 31),
                   "1 ano adicional hacia atras -- contiguo con 'extension', "
                   "que arranca 2024-09-01"),
    "extension3": (date(2022, 9, 1), date(2023, 8, 31),
                   "1 ano adicional hacia atras -- contiguo con 'extension2', "
                   "que arranca 2023-09-01"),
    "extension4": (date(2021, 9, 1), date(2022, 8, 31),
                   "1 ano adicional hacia atras -- contiguo con 'extension3', "
                   "que arranca 2022-09-01"),
    "medium": (date(2024, 7, 1),  date(2026, 8, 31), "26 meses"),
    "full":   (date(2018, 5, 1),  date(2026, 8, 31), "100 meses"),
}
EQ_DATASET = {"pilot": "EQUS.SUMMARY", "extension": "EQUS.SUMMARY",
              "extension2": "EQUS.SUMMARY", "extension3": "EQUS.SUMMARY",
              "extension4": "EQUS.SUMMARY",
              "medium": "EQUS.SUMMARY", "full": "XNAS.ITCH"}

OUT_OPT = ROOT / "data" / "raw" / "opra_chain"
OUT_EQ = ROOT / "data" / "raw" / "equities"
LOG = ROOT / "reports" / f"ingesta_{datetime.now():%Y%m%d_%H%M}.log"


def load_key() -> str:
    k = os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("DATABENTO_API_KEY"):
                return line.split("=", 1)[1].strip()
    sys.exit("No encuentro la API key. Ponla en .env o en DATABENTO_API_KEY.")


def business_days(a: date, b: date):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def weekly_days(a: date, b: date) -> list:
    """Un dia por semana: el viernes de cada semana ISO dentro del rango.

    Si un viernes es feriado no habra `definition` ese dia y el script lo
    reintenta con el jueves y luego el miercoles (segunda pasada), asi que
    ninguna semana se pierde por un feriado.
    """
    semanas = {}
    for d in business_days(a, b):
        semanas.setdefault(d.isocalendar()[:2], []).append(d)
    return [max(v) for _, v in sorted(semanas.items())]


def week_fallbacks(a: date, b: date) -> dict:
    """Candidatos alternativos por semana, del mas tarde al mas temprano."""
    semanas = {}
    for d in business_days(a, b):
        semanas.setdefault(d.isocalendar()[:2], []).append(d)
    return {k: sorted(v, reverse=True) for k, v in semanas.items()}


def estimate_cost(c, start: date, end: date, days: list) -> float:
    """Costo real sumando por schema, muestreando un dia habil representativo."""
    probe = days[len(days) // 2]
    p = probe.isoformat()
    total = 0.0
    print(f"\n  {'schema':<14}{'ventana':<26}{'$/dia':>10}{f'x{len(days)} dias':>16}")
    print(f"  {'-'*14}{'-'*26}{'-'*10}{'-'*16}")
    for sch, s, e, win in [
        ("definition", p, (probe + timedelta(days=1)).isoformat(), "dia completo"),
        ("statistics", f"{p}T{OI_WIN[0]}", f"{p}T{OI_WIN[1]}", "10:00-12:00 UTC (OI)"),
        ("cbbo-1m", f"{p}T{QUOTE_WIN[0]}", f"{p}T{QUOTE_WIN[1]}", "19:55-20:00 UTC (cierre)"),
    ]:
        try:
            v = c.metadata.get_cost(dataset=OPT_DATASET, symbols=PARENTS, stype_in="parent",
                                    schema=sch, start=s, end=e, mode="historical-streaming")
            total += v * len(days)
            print(f"  {sch:<14}{win:<26}{'$'+format(v,',.2f'):>10}{'$'+format(v*len(days),',.2f'):>16}")
        except Exception as ex:
            print(f"  {sch:<14}{win:<26}{'ERROR':>10}  {type(ex).__name__}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=list(SCOPES), default="pilot")
    ap.add_argument("--freq", choices=["weekly", "daily"], default="weekly",
                    help="weekly = un dia por semana (viernes). ~5x mas barato.")
    ap.add_argument("--max-cost", type=float, default=100.0,
                    help="tope de gasto en USD. Aborta si la estimacion lo supera.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N dias (prueba)")
    ap.add_argument("--only-equities", action="store_true",
                    help="solo re-baja las equities (barato, ~$0.01)")
    ap.add_argument("--workers", type=int, default=4,
                    help="dias en paralelo (1 = secuencial). 4-6 es seguro.")
    a = ap.parse_args()

    import databento as db
    c = db.Historical(load_key())
    start, end, etiqueta = SCOPES[a.scope]
    if a.freq == "weekly":
        days = weekly_days(start, end)
        fallbacks = week_fallbacks(start, end)
    else:
        days = list(business_days(start, end))
        fallbacks = {}
    if a.limit:
        days = days[:a.limit]

    print("=" * 84)
    print(f"FASE 1 -- INGESTA  ·  alcance '{a.scope}' ({etiqueta})".center(84))
    print(f"{start} -> {end}  ·  frecuencia {a.freq.upper()}  ·  {len(days)} fechas  ·  "
          f"{len(TODOS)} subyacentes".center(84))
    print("=" * 84)

    ya = sum(1 for d in days if (OUT_OPT / f"date={d.isoformat()}" / "chain.parquet").exists())
    if ya:
        print(f"\n  {ya} dias ya estan en disco y se van a saltar "
              f"({len(days)-ya} por bajar).")

    total = estimate_cost(c, start, end, days)
    print(f"\n  COSTO ESTIMADO TOTAL: ${total:,.2f}")
    if ya:
        print(f"  (dias pendientes: aprox ${total*(len(days)-ya)/len(days):,.2f})")

    pendiente = total * (len(days) - ya) / len(days) if len(days) else 0.0
    if a.dry_run:
        print("\n  --dry-run: no se descarga nada.")
        return 0

    if pendiente > a.max_cost:
        print(f"\n  ABORTADO: la estimacion pendiente (${pendiente:,.2f}) supera el")
        print(f"  tope de --max-cost (${a.max_cost:,.2f}). No se descargo nada.")
        print("  Opciones: --freq weekly, un --scope menor, o subir --max-cost")
        print("  explicitamente si de verdad quieres gastar mas.")
        return 2

    if a.only_equities:
        print("\n  --only-equities: solo se re-baja el historico de equities (~$0.01).")
    print(f"\n  Se van a hacer ~{(len(days)-ya)*3} llamadas a la API, {a.workers} dias en paralelo.")
    print("  La primera linea de la tabla tarda 1-4 min: cada dia baja ~2M registros")
    print("  de open interest y no se imprime nada hasta que el dia termina.")
    print("  Escribe EXACTAMENTE 'DESCARGAR' para confirmar el gasto.")
    if input("  > ").strip() != "DESCARGAR":
        print("  Cancelado. No se gasto nada.")
        return 0

    OUT_OPT.mkdir(parents=True, exist_ok=True)
    OUT_EQ.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = LOG.open("w")

    def w(s):
        print(s)
        fh.write(s + "\n")
        fh.flush()

    w(f"# ingesta {a.scope}  {start} -> {end}  {len(days)} dias  inicio {datetime.now():%H:%M:%S}")

    # ---------------------------------------------------------- equities
    w("\n[equities] cierre y volumen del subyacente...")
    try:
        eq = fetch_equities_daily(c, TODOS, start, end, EQ_DATASET[a.scope])
        if eq.is_empty():
            w("  ATENCION: equities vino vacio. Revisar dataset/symbology.")
        else:
            eq.write_parquet(OUT_EQ / f"daily_{a.scope}.parquet")
            w(f"  {eq.height:,} filas guardadas en {OUT_EQ.name}/daily_{a.scope}.parquet")
    except Exception as ex:
        w(f"  ERROR: {type(ex).__name__}: {ex}")

    # ------------------------------------------------------------ opciones
    if a.only_equities:
        w("\n--only-equities: listo, no se toca OPRA.")
        fh.close()
        return 0

    # dias con calidad reducida: se marcan, no se descartan
    cond = dataset_condition(c, OPT_DATASET, start, end)
    if cond:
        w(f"\n[calidad] {len(cond)} dias con calidad reducida segun Databento:")
        for d_, v_ in sorted(cond.items())[:15]:
            w(f"    {d_}  {v_}")
        if len(cond) > 15:
            w(f"    ... y {len(cond)-15} mas")
        w("    (se ingestan igual, quedan marcados para excluirlos en la puerta P2)")

    pend = [d for d in days
            if a.overwrite or not (OUT_OPT / f"date={d.isoformat()}" / "chain.parquet").exists()]
    w(f"\n[opciones] {len(pend)} dias por bajar, {a.workers} en paralelo")
    w(f"  {'fecha':<12}{'contratos':>10}{'OI':>10}{'quotes':>9}{'unidos':>9}"
      f"{'pub':>5}{'desc%':>7}{'def':>6}{'oi':>6}{'cbbo':>6}  notas")
    t0 = time.time()
    hechos = fallidos = 0
    done = 0

    def trabajo(d):
        return ingest_day(c, PARENTS, d, OUT_OPT, overwrite=a.overwrite)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(trabajo, d): d for d in pend}
        for fut in as_completed(futs):
            d = futs[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                w(f"  {d.isoformat():<12}  EXCEPCION {type(e).__name__}: {e}")
                fallidos += 1
                continue
            if r.skipped:
                continue
            nota = "; ".join(r.notes) if r.notes else ""
            if d.isoformat() in cond:
                nota = (nota + "; " if nota else "") + f"calidad:{cond[d.isoformat()]}"
            if r.error:
                nota = f"ERROR {r.error}"
                fallidos += 1
            elif r.n_joined:
                hechos += 1
            w(f"  {d.isoformat():<12}{r.n_contracts:>10,}{r.n_oi:>10,}{r.n_quotes:>9,}"
              f"{r.n_joined:>9,}{r.publishers_per_contract:>5}"
              f"{r.quote_discard_rate*100:>6.1f}%{r.t_defs:>6.0f}{r.t_oi:>6.0f}"
              f"{r.t_quotes:>6.0f}  {nota}")
            if done % 20 == 0 or done == len(pend):
                el = time.time() - t0
                w(f"  ... {done}/{len(pend)} · {el/60:.1f} min transcurridos · "
                  f"faltan ~{el/done*(len(pend)-done)/60:.0f} min")

    # segunda pasada: semanas sin datos (viernes feriado) -> jueves, miercoles
    if fallbacks:
        vacias = []
        for wk, cands in sorted(fallbacks.items()):
            if any((OUT_OPT / f"date={d.isoformat()}" / "chain.parquet").exists()
                   for d in cands):
                continue
            vacias.append((wk, cands))
        if vacias:
            w(f"\n[feriados] {len(vacias)} semanas sin datos en su ultimo dia habil;"
              f" probando dias anteriores")
            for wk, cands in vacias:
                for d in cands[1:3]:
                    r = ingest_day(c, PARENTS, d, OUT_OPT)
                    if r.n_joined:
                        w(f"  semana {wk[0]}-W{wk[1]:02d} resuelta con {d.isoformat()}"
                          f"  ({r.n_joined:,} contratos)")
                        hechos += 1
                        break
                    w(f"  semana {wk[0]}-W{wk[1]:02d}: {d.isoformat()} tampoco tiene datos"
                      f" ({'; '.join(r.notes) if r.notes else r.error})")
                else:
                    w(f"  semana {wk[0]}-W{wk[1]:02d} SIN DATOS tras 3 intentos")

    w(f"\n# fin {datetime.now():%H:%M:%S}  ·  {hechos} dias OK  ·  {fallidos} con error")
    w(f"# duracion total {(time.time()-t0)/60:.1f} min")
    w(f"# log: {LOG}")
    fh.close()
    print(f"\nLog completo en: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
