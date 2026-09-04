#!/usr/bin/env python3
"""Metricas de presentacion de la estrategia optima (cartera neutral).

Reconstruye la cartera neutral (beta + sector, rebalanceo mensual), la
descompone en TRADES individuales y calcula el cuadro completo de metricas:
actividad, rendimiento, riesgo, eficiencia y friccion operativa.

Un TRADE = una posicion en un nombre, desde que se abre (o cambia de signo)
hasta que se cierra (o cambia de signo). Con pesos continuos es la unica
definicion honesta: no hay 6v6 discreto.

Exporta JSON para el informe. Costo: $0, solo lee parquets locales.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.backtest.engine import fechas_rebalanceo, semanal_returns
from gex.backtest.neutral import SECTORES, calcular_beta, formar_carteras_neutral
from gex.curves import fetch_treasury_curve, rate_lookup
from gex.equities import load_equities

INDICES = ["SPY", "QQQ"]
OUT = pathlib.Path("/private/tmp/claude-501/-Users-sant-Finance2026-GEX-PROJECT-"
                   "/5f199f88-d292-4fc8-8e72-4ec96140a59f/scratchpad/metricas.json")
CAPITAL = 100_000.0
SEM = 52


def mdd_info(c):
    """Max drawdown, y la duracion (en periodos) del drawdown mas largo."""
    peak = np.maximum.accumulate(c)
    dd = c / peak - 1
    i_min = int(np.argmin(dd))
    # duracion: desde el pico previo hasta recuperarlo (o hasta el final)
    i_pico = int(np.argmax(c[:i_min + 1])) if i_min > 0 else 0
    rec = np.where(c[i_min:] >= c[i_pico])[0]
    i_fin = i_min + int(rec[0]) if len(rec) else len(c) - 1
    # racha bajo el agua mas larga
    bajo, mejor, actual = dd < -1e-9, 0, 0
    for b in bajo:
        actual = actual + 1 if b else 0
        mejor = max(mejor, actual)
    return float(dd.min()), i_fin - i_pico, mejor


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)
    curva = fetch_treasury_curve(ROOT)
    nt = pl.read_parquet(ROOT / "data/curated/backtest_neutral.parquet")

    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    fechas = sorted(xs["date"].unique().to_list())
    rets = semanal_returns(eq, fechas)
    rebal = fechas_rebalanceo(fechas, "mensual")

    # ---------- posiciones de la cartera neutral
    beta_df = calcular_beta(eq, fechas)
    pos = formar_carteras_neutral(xs, beta_df, SECTORES)
    pos_r = pos.filter(pl.col("date").is_in(rebal)).select(
        ["date", "underlying", "w", "sector", "beta", "gamma_exposure"])

    # ---------- serie de la estrategia y del benchmark
    spy = (rets.filter(pl.col("symbol") == "SPY").sort("date")
              .select(["date", pl.col("ret").alias("spy")]))
    df = (nt.select(["date", pl.col("ret_neto").alias("est"), "ret_bruto",
                     "costo_tx", "costo_borrow", "turnover", "rebalanceo"])
            .join(spy, on="date", how="inner").sort("date"))
    n = df.height
    rf_a = rate_lookup(curva, df["date"].to_list(), np.full(n, 0.25))
    rf = np.expm1(rf_a / SEM)
    est = df["est"].to_numpy(); bmk = df["spy"].to_numpy()

    def cuadro(r, nombre):
        ex = r - rf
        c = np.cumprod(1 + r)
        cagr = c[-1] ** (SEM / len(r)) - 1
        vol = r.std(ddof=1) * np.sqrt(SEM)
        dn = ex[ex < 0]
        dd, dur, bajo = mdd_info(c)
        import statsmodels.api as sm
        m = sm.OLS(ex, sm.add_constant(bmk - rf)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        gan, per = r[r > 0], r[r < 0]
        # racha perdedora maxima (periodos)
        mejor = act = 0
        for x in r:
            act = act + 1 if x < 0 else 0
            mejor = max(mejor, act)
        return dict(
            nombre=nombre, n_periodos=len(r),
            ret_total=float(c[-1] - 1), cagr=float(cagr), vol=float(vol),
            sharpe=float(ex.mean() * SEM / vol),
            sortino=float(ex.mean() * SEM / (dn.std(ddof=1) * np.sqrt(SEM))) if len(dn) > 2 else None,
            calmar=float(cagr / abs(dd)) if dd < 0 else None,
            mdd=float(dd), mdd_usd=float(CAPITAL * dd),
            mdd_dur_sem=int(dur), bajo_agua_sem=int(bajo),
            alpha_anual=float((1 + m.params[0]) ** SEM - 1), alpha_t=float(m.tvalues[0]),
            beta=float(m.params[1]), r2=float(m.rsquared),
            win_rate=float((r > 0).mean()),
            profit_factor=float(gan.sum() / abs(per.sum())) if len(per) else None,
            payoff=float(gan.mean() / abs(per.mean())) if len(per) else None,
            racha_perdedora=int(mejor),
            capital_final=float(CAPITAL * c[-1]),
            curva=[round(float(x) * CAPITAL, 2) for x in c],
        )

    est_m = cuadro(est, "GEX Neutral")
    bmk_m = cuadro(bmk, "SPY")

    # ---------- TRADES: cada posicion por nombre entre cambios de signo
    wmap = {(r["date"], r["underlying"]): r["w"] for r in pos_r.iter_rows(named=True)}
    rmap = {(r["date"], r["symbol"]): r["ret"] for r in rets.iter_rows(named=True)}
    nombres = sorted(xs["underlying"].unique().to_list())
    idx_f = {f: i for i, f in enumerate(fechas)}

    trades, abiertos = [], {}
    for k, t in enumerate(rebal):
        for nm in nombres:
            w = wmap.get((t, nm), 0.0)
            sg = 0 if abs(w) < 1e-6 else (1 if w > 0 else -1)
            ab = abiertos.get(nm)
            if ab and ab["signo"] != sg:
                ab["fin"] = t; trades.append(ab); abiertos.pop(nm)
                ab = None
            if sg != 0 and not ab:
                abiertos[nm] = dict(nombre=nm, signo=sg, ini=t, w=w,
                                    sector=SECTORES.get(nm, "Otro"))
    for nm, ab in abiertos.items():
        ab["fin"] = fechas[-1]; trades.append(ab)

    for tr in trades:
        i0, i1 = idx_f[tr["ini"]], idx_f[tr["fin"]]
        acum = 1.0
        for j in range(i0, i1):
            r = rmap.get((fechas[j + 1], tr["nombre"]))
            if r is not None and np.isfinite(r):
                acum *= (1 + r)
        bruto = (acum - 1) * tr["signo"]
        tr["semanas"] = i1 - i0
        tr["ret_bruto"] = float(bruto)
        tr["pnl"] = float(bruto * abs(tr["w"]))
        tr["ini"] = str(tr["ini"]); tr["fin"] = str(tr["fin"])

    tr_ok = [t for t in trades if t["semanas"] > 0]
    pnl = np.array([t["pnl"] for t in tr_ok])
    dur = np.array([t["semanas"] for t in tr_ok])
    gan, per = pnl[pnl > 0], pnl[pnl < 0]
    trade_stats = dict(
        n_trades=len(tr_ok),
        n_largos=sum(1 for t in tr_ok if t["signo"] > 0),
        n_cortos=sum(1 for t in tr_ok if t["signo"] < 0),
        win_rate=float((pnl > 0).mean()),
        profit_factor=float(gan.sum() / abs(per.sum())) if len(per) else None,
        payoff=float(gan.mean() / abs(per.mean())) if len(per) else None,
        expectativa=float(pnl.mean()),
        dur_media_sem=float(dur.mean()),
        dur_ganadoras=float(dur[pnl > 0].mean()),
        dur_perdedoras=float(dur[pnl < 0].mean()),
        mejor=float(pnl.max()), peor=float(pnl.min()),
    )

    # ---------- friccion
    costos = dict(
        tx_total=float(df["costo_tx"].sum()),
        borrow_total=float(df["costo_borrow"].sum()),
        total=float(df["costo_tx"].sum() + df["costo_borrow"].sum()),
        total_usd=float((df["costo_tx"].sum() + df["costo_borrow"].sum()) * CAPITAL),
        turnover_medio_rebal=float(df.filter(pl.col("rebalanceo"))["turnover"].mean()),
        n_rebalanceos=int(df["rebalanceo"].sum()),
        ret_bruto_total=float(np.prod(1 + df["ret_bruto"].to_numpy()) - 1),
    )
    # exposicion: la cartera es long-short, siempre invertida en fechas activas
    exp_bruta = (pos_r.group_by("date").agg(pl.col("w").abs().sum().alias("gross"))
                      .sort("date"))
    costos["exposicion_bruta_media"] = float(exp_bruta["gross"].mean())
    costos["pct_tiempo_en_mercado"] = 100.0

    # ---------- tabla mensual de posiciones
    mensual = []
    for t in rebal:
        s = pos_r.filter(pl.col("date") == t).sort("w", descending=True)
        lg = s.filter(pl.col("w") > 0.001)
        ct = s.filter(pl.col("w") < -0.001)
        mensual.append(dict(
            fecha=str(t),
            largos=[[r["underlying"], round(r["w"] * 100, 1)] for r in lg.head(6).iter_rows(named=True)],
            cortos=[[r["underlying"], round(abs(r["w"]) * 100, 1)] for r in ct.tail(6).iter_rows(named=True)],
            n_largos=lg.height, n_cortos=ct.height))

    fechas_ser = [str(d) for d in df["date"].to_list()]
    reb_flags = df["rebalanceo"].to_list()

    out = dict(
        capital_inicial=CAPITAL,
        periodo=[fechas_ser[0], fechas_ser[-1]],
        estrategia=est_m, benchmark=bmk_m,
        trades=trade_stats, costos=costos, mensual=mensual,
        serie=dict(fechas=fechas_ser, rebalanceo=reb_flags),
        top_trades=sorted(tr_ok, key=lambda t: -t["pnl"])[:8],
        peores_trades=sorted(tr_ok, key=lambda t: t["pnl"])[:8],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))

    print(f"periodo {out['periodo'][0]} -> {out['periodo'][1]} · {n} semanas")
    print(f"\n{'metrica':<26}{'GEX Neutral':>16}{'SPY':>14}")
    print("-" * 56)
    for k, lab, f in [("ret_total", "Retorno total", "pct"), ("cagr", "CAGR", "pct"),
                      ("vol", "Volatilidad", "pct"), ("sharpe", "Sharpe", "num"),
                      ("sortino", "Sortino", "num"), ("calmar", "Calmar", "num"),
                      ("mdd", "Max drawdown", "pct"), ("alpha_anual", "Alfa anual", "pct"),
                      ("beta", "Beta", "num"), ("win_rate", "Win rate", "pct")]:
        a, b = est_m[k], bmk_m[k]
        if f == "pct":
            print(f"{lab:<26}{a*100:>15.2f}%{b*100:>13.2f}%")
        else:
            print(f"{lab:<26}{a:>16.2f}{b:>14.2f}")
    print(f"\nTRADES: {trade_stats['n_trades']} "
          f"({trade_stats['n_largos']} largos / {trade_stats['n_cortos']} cortos)")
    print(f"  win rate {trade_stats['win_rate']*100:.1f}% · "
          f"profit factor {trade_stats['profit_factor']:.2f} · "
          f"payoff {trade_stats['payoff']:.2f}")
    print(f"  duracion media {trade_stats['dur_media_sem']:.1f} sem "
          f"(ganadoras {trade_stats['dur_ganadoras']:.1f} · "
          f"perdedoras {trade_stats['dur_perdedoras']:.1f})")
    print(f"\nCOSTOS: {costos['total']*100:.2f}% del capital "
          f"(${costos['total_usd']:,.0f}) · {costos['n_rebalanceos']} rebalanceos")
    print(f"\nJSON -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
