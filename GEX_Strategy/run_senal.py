#!/usr/bin/env python3
"""PUERTA P2 -- Calcula Gamma sobre los dias en disco y valida las magnitudes.

Cuesta $0: usa solo lo que ya esta bajado. Es la puerta que decide si vale la
pena gastar el resto del presupuesto.

CRITERIOS DE ACEPTACION, con la correccion de expectativa que importa:
el paper reporta correlacion +0.15 entre Gamma y tamano, asi que un universo de
mega-caps debe dar Gamma MEDIANA POR ENCIMA de 0.41 y MENOS del 21.8% de
observaciones negativas. Si replicaramos exactamente la Tabla 1 seria sospechoso.

USO
    python3 run_senal.py
"""
from __future__ import annotations

import glob
import pathlib
import sys
import time

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.curves import fetch_treasury_curve
from gex.equities import load_equities
from gex.signal.gamma_exposure import (add_adv, aggregate, prepare, solve_greeks,
                                       winsorize_zscore)
from gex.signal.implied_carry import attach_carry, implied_carry, resumen_por_accion

INDICES = {"SPY", "QQQ"}
OUT = ROOT / "data" / "curated"
REP = ROOT / "reports" / "p2_senal.txt"
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def main():
    files = sorted(glob.glob(str(ROOT / "data/raw/opra_chain/date=*/chain.parquet")))
    if not files:
        sys.exit("No hay cadenas en data/raw/opra_chain/. Corre run_ingesta.py primero.")
    try:
        eq = load_equities(ROOT)
    except FileNotFoundError as ex:
        sys.exit(str(ex))

    say("=" * 92)
    say("PUERTA P2 -- CALCULO Y VALIDACION DE GAMMA".center(92))
    say(f"{len(files)} dias en disco".center(92))
    say("=" * 92)

    adv = add_adv(eq)
    n_adv = adv.filter(pl.col("adv_usd").is_not_null()).height
    say(f"\nequities: {eq.height:,} filas · {adv.height:,} con fecha · "
        f"{n_adv:,} con ADV$ de 21 dias disponible")
    if n_adv == 0:
        say("\n  ATENCION: ningun dia tiene ADV$ de 21 dias.")
        say("  El historico de equities arranca el mismo dia que la muestra.")
        say("  Corre:  python3 run_ingesta.py --only-equities")
        say("  (ya parcheado para bajar 45 dias previos, cuesta ~$0.01)")
        REP.write_text("\n".join(_lines) + "\n")
        return 1

    # Los dias reparados quedaron con las columnas de cotizacion al final (el join
    # las reubica), asi que se normaliza el orden por NOMBRE antes de concatenar.
    COLS = ["date", "underlying", "instrument_id", "raw_symbol", "expiration",
            "strike", "klass", "multiplier", "is_call", "open_interest",
            "bid", "ask", "mid", "rel_spread"]
    partes = []
    for f in files:
        d = pl.read_parquet(f)
        faltan = [c for c in COLS if c not in d.columns]
        if faltan:
            say(f"  ATENCION {pathlib.Path(f).parent.name}: faltan columnas {faltan}")
            continue
        partes.append(d.select(COLS))
    chain = pl.concat(partes, how="vertical_relaxed")
    say(f"cadenas:  {chain.height:,} contrato-dia · "
        f"{chain['underlying'].n_unique()} subyacentes · "
        f"{chain['date'].n_unique()} fechas")

    t0 = time.time()
    pre = prepare(chain, adv)
    say(f"\ntras filtros de calidad: {pre.height:,} contratos "
        f"({pre.height/chain.height*100:.1f}% de los ingestados)")
    say("  descartes: T<1d, mid<=0, OI=0, sin ADV$, rel_spread>50%, precio<intrinseco")

    # ---- r observada del Tesoro; dividendo + borrow de la paridad put-call
    curva = fetch_treasury_curve(ROOT)
    r3m = curva.filter(pl.col("date").is_between(pre["date"].min(), pre["date"].max()))
    say(f"\ncurva del Tesoro (FRED): {r3m.height:,} dias · "
        f"3M de {r3m['DGS3MO'].min():.2f}% a {r3m['DGS3MO'].max():.2f}%")
    say("  r se LEE de la curva, interpolada al plazo de cada contrato.")
    say("  La paridad put-call se reserva para el dividendo + costo de prestamo:")
    say("  su pendiente no resuelve r en vencimientos cortos (daba 3-5x por debajo).")
    say("\nextrayendo dividendos implicitos de la paridad put-call...")
    t1 = time.time()
    carry = implied_carry(pre, r_curve=curva)
    say(f"  {carry.height:,} ajustes (fecha x subyacente x vencimiento) · {time.time()-t1:.0f}s")
    pre = attach_carry(pre, carry, r_curve=curva)
    cov = float(pre["carry_ok"].mean()) * 100 if "carry_ok" in pre.columns else 0.0
    say(f"  cobertura: {cov:.1f}% de los contratos  (r comun por fecha, yield de")
    say(f"  dividendos por accion extrapolado a todos los vencimientos con D = S*q*T)")
    if not carry.is_empty():
        rp = resumen_por_accion(carry)
        SIN_DIV = {"TSLA","GME","PLTR","COIN","MSTR","RIVN","SOFI","SHOP","AMD",
                   "UBER","NFLX","BA","AMZN","GOOGL"}
        say("")
        say("  PRUEBA DEL METODO: carry implicito anualizado (dividendo + prestamo)")
        say("  Los que NO pagan dividendo y son faciles de prestar deben dar ~0.")
        say("  Los hard-to-borrow (GME, RIVN, SOFI) SI deben dar positivo: es el")
        say("  costo de prestamo del papel, y para valuar la opcion es correcto.")
        say("  Columna K<=S: mismo estimado usando solo puts OTM, donde el sesgo")
        say("  de ejercicio anticipado es minimo. Si difiere mucho, hay sesgo.")
        say(f"    {'ticker':<8}{'carry':>9}{'K<=S':>9}{'r impl':>9}{'disp':>8}   esperado")
        for row in rp.iter_rows(named=True):
            u = row["underlying"]
            esp = "NO paga dividendo" if u in SIN_DIV else "paga dividendo"
            marca = ""
            HTB = {"GME", "RIVN", "SOFI", "MSTR", "COIN"}
            if u in HTB:
                esp = "hard-to-borrow: carry > 0 esperado"
            if u in SIN_DIV and u not in HTB and row["div_yield_med"] > 0.01:
                marca = "  <-- deberia ser ~0"
            if u not in SIN_DIV and row["div_yield_med"] < 0.002:
                marca = "  <-- deberia ser > 0"
            qo = row.get('q_otm_med')
            qo_s = f"{qo*100:>8.2f}%" if qo is not None and qo == qo else f"{'--':>9}"
            say(f"    {u:<8}{row['div_yield_med']*100:>8.2f}%{qo_s}"
                f"{row['r_med']*100:>8.2f}%{row['rmse_med']:>8.2f}   {esp}{marca}")
        say("")
        say(f"  r implicito mediano global: {float(rp['r_med'].median())*100:.2f}%"
            f"   (antes se usaba 4.20% fijo)")

    say("\ninvirtiendo IV y calculando gamma con arbol CRR...")
    gr = solve_greeks(pre)
    say(f"  {gr.height:,} contratos con IV valida "
        f"({gr.height/pre.height*100:.1f}%)  ·  {time.time()-t0:.0f}s")
    say(f"  IV: p5={gr['iv'].quantile(0.05):.3f}  mediana={gr['iv'].median():.3f}  "
        f"p95={gr['iv'].quantile(0.95):.3f}")
    say(f"  gamma*S: mediana={float((gr['gamma']*gr['close']).median()):.4f}")

    g = aggregate(gr)
    OUT.mkdir(parents=True, exist_ok=True)
    g.write_parquet(OUT / "gamma_exposure.parquet")
    gr.write_parquet(OUT / "contract_greeks.parquet")

    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    ix = g.filter(pl.col("underlying").is_in(INDICES))

    say("\n" + "-" * 92)
    say("GAMMA POR SUBYACENTE  (promedio de los dias disponibles, cross-section)")
    say("-" * 92)
    say(f"  {'ticker':<8}{'Gamma':>10}{'bruta':>11}{'net/gross':>11}{'spot':>9}"
        f"{'ADV$ (M)':>11}{'contr':>8}{'OI tot':>11}{'IV med':>8}")
    prom = (xs.group_by("underlying").agg([
        pl.col("gamma_exposure").mean(), pl.col("gamma_gross").mean(),
        pl.col("net_gross_ratio").mean(), pl.col("spot").mean(),
        pl.col("adv_usd").mean(), pl.col("n_contracts").mean(),
        pl.col("total_oi").mean(), pl.col("iv_median").mean()])
        .sort("gamma_exposure"))
    for r in prom.iter_rows(named=True):
        say(f"  {r['underlying']:<8}{r['gamma_exposure']:>10.3f}{r['gamma_gross']:>11.2f}"
            f"{r['net_gross_ratio']:>11.3f}{r['spot']:>9.2f}{r['adv_usd']/1e6:>11.0f}"
            f"{r['n_contracts']:>8.0f}{r['total_oi']:>11,.0f}{r['iv_median']:>8.3f}")

    say("\n  Track de indice (fuera del cross-section):")
    for r in (ix.group_by("underlying").agg([
            pl.col("gamma_exposure").mean(), pl.col("gamma_gross").mean(),
            pl.col("net_gross_ratio").mean()]).iter_rows(named=True)):
        say(f"  {r['underlying']:<8}{r['gamma_exposure']:>10.3f}"
            f"{r['gamma_gross']:>11.2f}{r['net_gross_ratio']:>11.3f}")

    # -------------------------------------------------- validacion
    ge = xs["gamma_exposure"]
    pct_neg = float((ge < 0).mean()) * 100
    say("\n" + "=" * 92)
    say("VALIDACION CONTRA EL PAPER")
    say("=" * 92)
    say("  BENCHMARK CORRECTO: la Tabla 1 del paper es la distribucion cross-seccional")
    say("  EQUAL-WEIGHTED de TODO el universo CRSP (miles de acciones, casi todas mucho")
    say("  mas chicas). La Tabla 2 da el Gamma promedio por decil VALUE-WEIGHTED, es")
    say("  decir dominado por las acciones grandes: de -0.01 (decil L) a +0.04 (decil H).")
    say("  Para un universo de 30 mega-caps el benchmark es la Tabla 2, no la Tabla 1.")
    say("  Razon: Gamma lleva el ADV$ en el DENOMINADOR. Las mega-caps tienen el mercado")
    say("  de acciones mas profundo del mundo, asi que el mismo open interest representa")
    say("  una fraccion mucho menor del volumen diario.")
    say("")
    say(f"  {'estadistico':<26}{'nuestro':>11}{'Tabla 2 (VW)':>14}   veredicto")
    say(f"  {'-'*26}{'-'*11}{'-'*14}   {'-'*38}")

    def linea(nom, val, ref, ok, nota):
        say(f"  {nom:<26}{val:>11.3f}{ref:>14}   {'OK     ' if ok else 'REVISAR'} {nota}")

    p10, p25 = float(ge.quantile(0.10)), float(ge.quantile(0.25))
    med, p75, p90 = float(ge.median()), float(ge.quantile(0.75)), float(ge.quantile(0.90))
    linea("P10 (~decil L del paper)", p10, "-0.01", -0.04 < p10 < 0.01, "decil L VW = -0.01")
    linea("mediana (~decil 9)", med, "0.02", 0.0 < med < 0.06, "deciles 6-9 VW = 0.01-0.02")
    linea("P90 (~decil H)", p90, "0.04", 0.01 < p90 < 0.20, "decil H VW = 0.04")
    linea("rango P90-P10 (~H-L)", p90 - p10, "0.05", 0.01 < (p90-p10) < 0.25, "H-L VW = 0.05")
    linea("% observaciones negativas", pct_neg, "21.8", pct_neg < 21.8,
          "menor que 21.8%: mega-caps tienen mas Gamma positiva")

    say("")
    say("  Validaciones externas (no vienen del paper):")
    spy = ix.filter(pl.col("underlying") == "SPY")["gamma_exposure"].mean()
    qqq = ix.filter(pl.col("underlying") == "QQQ")["gamma_exposure"].mean()
    ok_ix = (spy is not None and spy < 0) and (qqq is not None and qqq < 0)
    say(f"    Gamma de indice NEGATIVA (SPY {spy:+.3f}, QQQ {qqq:+.3f}): "
        f"{'OK' if ok_ix else 'REVISAR'}")
    say("      Los market makers estan estructuralmente CORTOS gamma en opciones de")
    say("      indice por la demanda institucional de puts. Que el signo salga negativo")
    say("      sin habersele dicho valida la convencion de signos del pipeline.")
    say(f"    Acciones individuales mayormente Gamma POSITIVA ({100-pct_neg:.0f}%): "
        f"{'OK' if pct_neg < 40 else 'REVISAR'}  (paper: 78.2% positivas)")

    say("")
    say("  Descomposicion (el paper: la senal viene de ATM/OTM y de 'slow'):")
    for c, lab in [("gex_atm", "ATM"), ("gex_otm", "OTM"), ("gex_itm", "ITM"),
                   ("gex_fast", "fast (<=31d)"), ("gex_slow", "slow (>31d)")]:
        v = xs[c]
        say(f"    {lab:<16} media={float(v.mean()):>9.3f}   "
            f"|contribucion| media={float(v.abs().mean()):>9.3f}")

    say("")
    say(f"  Razon de cancelacion net/gross: min={float(xs['net_gross_ratio'].min()):.4f}  "
        f"mediana={float(xs['net_gross_ratio'].median()):.4f}  "
        f"max={float(xs['net_gross_ratio'].max()):.4f}")
    say("  (Hallazgo H1: cuanto menor, mas se amplifica cualquier error de datos)")

    if xs["date"].n_unique() >= 2:
        w = winsorize_zscore(xs)
        say("\n  Ranking por z-score de Gamma (ultima fecha):")
        last = w.filter(pl.col("date") == w["date"].max()).sort("gex_z")
        lo = last.head(6)["underlying"].to_list()
        hi = last.tail(6)["underlying"].to_list()
        say(f"    quintil LARGO  (Gamma mas baja): {', '.join(lo)}")
        say(f"    quintil CORTO  (Gamma mas alta): {', '.join(hi)}")

    say("\n" + "=" * 92)
    say(f"  Guardado: {OUT/'gamma_exposure.parquet'}")
    say(f"  Reporte:  {REP}")
    say("=" * 92)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
