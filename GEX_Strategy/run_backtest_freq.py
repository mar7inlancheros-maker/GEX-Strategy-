#!/usr/bin/env python3
"""Comparacion de disenos de rebalanceo. El semanal rotaba 150% por semana.

El paper rebalancea MENSUAL. Aqui se comparan cuatro disenos sobre exactamente
la misma senal, para aislar cuanto del resultado era la estrategia y cuanto era
el diseno de implementacion.

Nota de medicion: aunque se rebalancee mensual, los retornos se miden SEMANALES
(la cartera se mantiene y los pesos derivan entre rebalanceos). Eso da 53
observaciones en vez de 13 para el mismo diseno, asi que el drawdown y la
volatilidad quedan mejor medidos sin cambiar la estrategia.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.backtest.engine import (fechas_rebalanceo, metricas, semanal_returns,
                                 simular_periodica)
from gex.curves import fetch_treasury_curve
from gex.equities import load_equities
from gex.signal.implied_carry import implied_carry

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "backtest_freq.txt"
_lines = []


def say(s=""):
    print(s, flush=True); _lines.append(s)


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    gr = pl.read_parquet(ROOT / "data/curated/contract_greeks.parquet")
    eq = load_equities(ROOT)
    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    fechas = sorted(xs["date"].unique().to_list())
    rets = semanal_returns(eq, fechas)

    curva = fetch_treasury_curve(ROOT)
    carry = implied_carry(gr.select(["date", "underlying", "expiration", "strike",
                                     "mid", "close", "T", "is_call"]), r_curve=curva)
    borrow = {}
    if not carry.is_empty():
        b = (carry.filter(pl.col("T") > 0.05)
             .with_columns((pl.col("div_pv_impl") / pl.col("spot") / pl.col("T")).alias("q"))
             .group_by("underlying").agg(pl.col("q").median()))
        borrow = dict(zip(b["underlying"].to_list(), b["q"].to_list()))

    mens = fechas_rebalanceo(fechas, "mensual")
    say("=" * 96)
    say("DISENO DE REBALANCEO -- misma senal, cuatro implementaciones".center(96))
    say("=" * 96)
    say(f"\nfechas de senal: {len(fechas)} semanales")
    say(f"fechas de rebalanceo mensual: {len(mens)}  ({mens[0]} ... {mens[-1]})")
    say("Los retornos se miden semanales en los cuatro casos; lo que cambia es")
    say("cada cuanto se re-arma la cartera.\n")

    disenos = [
        ("Semanal (lo que corrimos)", dict(freq="semanal", banda=0)),
        ("MENSUAL (el diseno del paper)", dict(freq="mensual", banda=0)),
        ("Mensual + banda 3", dict(freq="mensual", banda=3)),
        ("Mensual + banda 6", dict(freq="mensual", banda=6)),
    ]
    say(f"  {'diseno':<30}{'ret anual':>11}{'vol':>8}{'Sharpe':>8}{'max DD':>9}"
        f"{'turnover':>10}{'costo/año':>11}")
    say(f"  {'-'*30}{'-'*11}{'-'*8}{'-'*8}{'-'*9}{'-'*10}{'-'*11}")
    guardado = {}
    for nombre, kw in disenos:
        s = simular_periodica(xs, rets, borrow=borrow, n_pata=6, **kw)
        guardado[nombre] = s
        m = metricas(s, "ret_neto")
        tn = float(s["turnover"].mean())
        cost = float((s["costo_tx"] + s["costo_borrow"]).sum())
        say(f"  {nombre:<30}{m['ret_anual']*100:>10.2f}%{m['vol_anual']*100:>7.1f}%"
            f"{m['sharpe']:>8.2f}{m['max_dd']*100:>8.1f}%{tn*100:>9.0f}%"
            f"{cost*100:>10.2f}%")

    say("")
    say("-" * 96)
    say("DETALLE DEL DISENO MENSUAL (el del paper)")
    say("-" * 96)
    s = guardado["MENSUAL (el diseno del paper)"]
    mb, mn = metricas(s, "ret_bruto"), metricas(s, "ret_neto")
    say(f"  {'':26}{'BRUTO':>12}{'NETO':>12}")
    for lab, k in [("retorno total", "ret_total"), ("retorno anualizado", "ret_anual"),
                   ("volatilidad anual", "vol_anual"), ("max drawdown", "max_dd"),
                   ("semanas ganadoras", "hit_rate")]:
        say(f"  {lab:<26}{mb[k]*100:>11.2f}%{mn[k]*100:>11.2f}%")
    say(f"  {'Sharpe':<26}{mb['sharpe']:>12.2f}{mn['sharpe']:>12.2f}")
    say(f"  {'t-stat':<26}{mb['t_stat']:>12.2f}{mn['t_stat']:>12.2f}")
    say(f"  IC95 del retorno anual NETO: [{mn['ic95_lo']*100:+.1f}%, {mn['ic95_hi']*100:+.1f}%]")
    say(f"  rebalanceos efectivos: {int(s['rebalanceo'].sum())}  ·  "
        f"turnover medio en fecha de rebalanceo: "
        f"{float(s.filter(pl.col('rebalanceo'))['turnover'].mean())*100:.0f}%")

    say("")
    say("-" * 96)
    say("CURVA DE CAPITAL -- mensual, neto (base 100)")
    say("-" * 96)
    c = 100 * np.cumprod(1 + s["ret_neto"].to_numpy())
    f = s["date"].to_list()
    lo, hi = float(c.min()), float(c.max())
    for i in range(0, len(c), max(1, len(c) // 24)):
        say(f"  {f[i]}  {c[i]:>7.1f}  {'.' * int((c[i]-lo)/max(hi-lo,1e-9)*42)}o")
    say(f"\n  final {c[-1]:.1f}")

    bench = (rets.filter(~pl.col("symbol").is_in(INDICES)).group_by("date")
             .agg(pl.col("ret").mean().alias("ret_neto")).sort("date"))
    mbe = metricas(bench, "ret_neto")
    spy = rets.filter(pl.col("symbol") == "SPY").sort("date").rename({"ret": "ret_neto"})
    ms = metricas(spy, "ret_neto")
    say("")
    say(f"  referencia · benchmark 30 nombres: {mbe['ret_anual']*100:.1f}% anual, "
        f"Sharpe {mbe['sharpe']:.2f}, DD {mbe['max_dd']*100:.1f}%")
    say(f"  referencia · SPY: {ms['ret_anual']*100:.1f}% anual, "
        f"Sharpe {ms['sharpe']:.2f}, DD {ms['max_dd']*100:.1f}%")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
