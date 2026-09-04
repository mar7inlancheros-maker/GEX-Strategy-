#!/usr/bin/env python3
"""BACKTEST -- cuanto habria rendido el portafolio.

Tres variantes, rebalanceo semanal, point-in-time y costos reales:
  A. Long-Short: largo el quintil de Gamma mas baja, corto el mas alta
  B. Long-only tilt: solo la pata larga
  C. Benchmark: 30 nombres equiponderados, y SPY

El IC 95% va PEGADO al retorno. Con ~50 semanas es ancho por aritmetica, no
por defecto del codigo, y el retorno puntual no es evidencia de nada.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.backtest.engine import metricas, semanal_returns, simular
from gex.curves import fetch_treasury_curve
from gex.equities import load_equities
from gex.signal.implied_carry import implied_carry

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "backtest.txt"
_lines = []


def say(s=""):
    print(s, flush=True); _lines.append(s)


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    gr = pl.read_parquet(ROOT / "data/curated/contract_greeks.parquet")
    eq = load_equities(ROOT)
    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    fechas = sorted(xs["date"].unique().to_list())

    say("=" * 92)
    say("BACKTEST -- NET GAMMA EXPOSURE".center(92))
    say(f"{fechas[0]} -> {fechas[-1]}  ·  {len(fechas)} rebalanceos semanales".center(92))
    say("=" * 92)

    rets = semanal_returns(eq, fechas)
    say(f"\nretornos semanales: {rets.height:,} obs · {rets['date'].n_unique()} semanas")

    curva = fetch_treasury_curve(ROOT)
    carry = implied_carry(gr.select(["date", "underlying", "expiration", "strike",
                                     "mid", "close", "T", "is_call"]), r_curve=curva)
    borrow = {}
    if not carry.is_empty():
        b = (carry.filter(pl.col("T") > 0.05)
             .with_columns((pl.col("div_pv_impl") / pl.col("spot") / pl.col("T")).alias("q"))
             .group_by("underlying").agg(pl.col("q").median()))
        borrow = dict(zip(b["underlying"].to_list(), b["q"].to_list()))
        top = sorted(borrow.items(), key=lambda kv: -kv[1])[:5]
        say("costo de prestamo (del carry implicito), los mas caros: " +
            ", ".join(f"{k} {v*100:.2f}%" for k, v in top))

    say("\n" + "-" * 92)
    say("RESULTADOS")
    say("-" * 92)
    res = {}
    for nombre, kw in [("A. Long-Short (6+6)", dict(n_pata=6)),
                       ("B. Long-only tilt", dict(n_pata=6, solo_largo=True))]:
        s = simular(xs, rets, borrow=borrow, **kw)
        res[nombre] = s
        mb, mn = metricas(s, "ret_bruto"), metricas(s, "ret_neto")
        say(f"\n  {nombre}")
        say(f"    {'':24}{'BRUTO':>12}{'NETO':>12}")
        for lab, k in [("retorno total", "ret_total"), ("retorno anualizado", "ret_anual"),
                       ("volatilidad anual", "vol_anual"), ("max drawdown", "max_dd"),
                       ("semanas ganadoras", "hit_rate")]:
            say(f"    {lab:<24}{mb[k]*100:>11.2f}%{mn[k]*100:>11.2f}%")
        say(f"    {'Sharpe':<24}{mb['sharpe']:>12.2f}{mn['sharpe']:>12.2f}")
        say(f"    {'t-stat':<24}{mb['t_stat']:>12.2f}{mn['t_stat']:>12.2f}")
        say(f"    IC95 del retorno anual NETO: "
            f"[{mn['ic95_lo']*100:+.1f}%, {mn['ic95_hi']*100:+.1f}%]")
        say(f"    turnover {float(s['turnover'].mean())*100:.0f}%/sem · "
            f"tx {float(s['costo_tx'].mean())*1e4:.1f} bps · "
            f"prestamo {float(s['costo_borrow'].mean())*1e4:.1f} bps")

    bench = (rets.filter(~pl.col("symbol").is_in(INDICES)).group_by("date")
             .agg(pl.col("ret").mean().alias("ret_neto")).sort("date"))
    mbe = metricas(bench, "ret_neto")
    say(f"\n  C. Benchmark equiponderado (30 nombres): anual {mbe['ret_anual']*100:.2f}% · "
        f"vol {mbe['vol_anual']*100:.2f}% · Sharpe {mbe['sharpe']:.2f} · "
        f"DD {mbe['max_dd']*100:.1f}%")
    spy = rets.filter(pl.col("symbol") == "SPY").sort("date").rename({"ret": "ret_neto"})
    if spy.height > 5:
        ms = metricas(spy, "ret_neto")
        say(f"     SPY: anual {ms['ret_anual']*100:.2f}% · vol {ms['vol_anual']*100:.2f}% "
            f"· Sharpe {ms['sharpe']:.2f} · DD {ms['max_dd']*100:.1f}%")

    say("\n" + "-" * 92)
    say("CURVA DE CAPITAL -- Long-Short neto de costos (base 100)")
    say("-" * 92)
    s = res["A. Long-Short (6+6)"]
    c = 100 * np.cumprod(1 + s["ret_neto"].to_numpy())
    f = s["date"].to_list()
    lo, hi = float(c.min()), float(c.max())
    for i in range(0, len(c), max(1, len(c) // 26)):
        say(f"  {f[i]}  {c[i]:>7.1f}  {'.' * int((c[i]-lo)/max(hi-lo,1e-9)*44)}o")
    mn = metricas(s, "ret_neto")
    say(f"\n  final {c[-1]:.1f}  ·  retorno anual neto {mn['ret_anual']*100:+.1f}%")
    say(f"  IC95 [{mn['ic95_lo']*100:+.1f}%, {mn['ic95_hi']*100:+.1f}%]  "
        f"-> ancho {(mn['ic95_hi']-mn['ic95_lo'])*100:.0f} puntos con {mn['n']} semanas")
    say("=" * 92)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    s.write_parquet(ROOT / "data/curated/backtest_ls.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
