#!/usr/bin/env python3
"""Backtest neutralizado por beta y por sector -- vs. la version original 6v6.

Compara, sobre la MISMA señal de Gamma y el MISMO diseño de rebalanceo mensual
(el del paper), dos construcciones de cartera:
  A. Original: +1/6 a los 6 Gamma mas bajos, -1/6 a los 6 mas altos, sin mirar
     sector ni beta.
  B. Neutral: peso proporcional al z-score de Gamma DENTRO de cada sector,
     escalado por 1/beta. Sector-neutral por construccion, beta-neutral por
     escalado.

El punto no es que B tenga que ganarle a A en retorno -- el punto es ver si
el -35% de drawdown que vimos en A se explica por el sesgo a tech (y por lo
tanto se reduce en B) o si persiste igual (evidencia de que el problema no
era el sesgo sectorial).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.backtest.engine import (fechas_rebalanceo, formar_carteras, metricas,
                                 semanal_returns, simular_periodica)
from gex.backtest.neutral import (SECTORES, calcular_beta, exposicion_sectorial,
                                  formar_carteras_neutral, simular_neutral)
from gex.curves import fetch_treasury_curve
from gex.equities import load_equities
from gex.signal.implied_carry import implied_carry

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "backtest_neutral.txt"
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

    say("=" * 96)
    say("NEUTRALIZACION POR BETA Y SECTOR -- comparacion vs. cartera original".center(96))
    say("=" * 96)

    # --- beta point-in-time, ventana expansiva ---
    say("\nCalculando beta diario contra SPY (ventana expansiva, point-in-time)...")
    beta_df = calcular_beta(eq, fechas)
    say(f"  {len(fechas)} fechas x {xs['underlying'].n_unique()} nombres = "
        f"{beta_df.height} estimaciones de beta")
    prom_obs_final = beta_df.filter(pl.col("date") == fechas[-1])["n_obs"].mean()
    say(f"  observaciones diarias disponibles en la ultima fecha: {prom_obs_final:.0f}")

    # --- A. cartera original ---
    pos_a = formar_carteras(xs, n_pata=6)
    pos_a = pos_a.join(beta_df, left_on=["date", "underlying"],
                       right_on=["date", "symbol"], how="left")
    s_a = simular_periodica(xs, rets, freq="mensual", n_pata=6, borrow=borrow)
    m_a = metricas(s_a, "ret_neto")

    beta_a = (pos_a.filter(pl.col("w") != 0)
              .group_by("date")
              .agg((pl.col("w") * pl.col("beta")).filter(pl.col("w") > 0).sum().alias("bl"),
                   (pl.col("w") * pl.col("beta")).filter(pl.col("w") < 0).sum().alias("bc")))
    beta_a = beta_a.with_columns((pl.col("bl") + pl.col("bc")).alias("bn"))

    # --- B. cartera neutral ---
    pos_b = formar_carteras_neutral(xs, beta_df, SECTORES)
    s_b = simular_neutral(pos_b, rets, freq="mensual", borrow=borrow)
    m_b = metricas(s_b, "ret_neto")
    expo_sec = exposicion_sectorial(pos_b)

    say("")
    say("-" * 96)
    say(f"  {'':30}{'A. ORIGINAL (6v6)':>22}{'B. NEUTRAL':>22}")
    say("-" * 96)
    for lab, k in [("retorno anual neto", "ret_anual"), ("vol anual", "vol_anual"),
                   ("Sharpe neto", "sharpe"), ("max drawdown", "max_dd"),
                   ("hit rate", "hit_rate"), ("t-stat", "t_stat")]:
        va = m_a.get(k, float("nan")); vb = m_b.get(k, float("nan"))
        if k in ("sharpe", "t_stat"):
            say(f"  {lab:<30}{va:>22.2f}{vb:>22.2f}")
        else:
            say(f"  {lab:<30}{va*100:>21.2f}%{vb*100:>21.2f}%")
    say(f"  {'turnover medio/rebal':<30}"
        f"{float(s_a.filter(pl.col('rebalanceo'))['turnover'].mean())*100:>21.0f}%"
        f"{float(s_b.filter(pl.col('rebalanceo'))['turnover'].mean())*100:>21.0f}%")
    say(f"  {'costo total periodo':<30}"
        f"{float((s_a['costo_tx']+s_a['costo_borrow']).sum())*100:>21.2f}%"
        f"{float((s_b['costo_tx']+s_b['costo_borrow']).sum())*100:>21.2f}%")

    say("")
    say("-" * 96)
    say("DIAGNOSTICO -- beta ponderada de cada pata (el numero que prueba si el fix funciono)")
    say("-" * 96)
    say(f"  {'':30}{'A. ORIGINAL':>22}{'B. NEUTRAL':>22}")
    say(f"  {'beta pata larga (prom)':<30}{float(beta_a['bl'].mean()):>22.2f}"
        f"{float(s_b['beta_largo'].mean()):>22.2f}")
    say(f"  {'beta pata corta (prom)':<30}{float(beta_a['bc'].mean()):>22.2f}"
        f"{float(s_b['beta_corto'].mean()):>22.2f}")
    say(f"  {'beta NETA del L-S (prom)':<30}{float(beta_a['bn'].mean()):>22.2f}"
        f"{float(s_b['beta_neta'].mean()):>22.2f}")
    say(f"  {'|beta neta| maxima':<30}{float(beta_a['bn'].abs().max()):>22.2f}"
        f"{float(s_b['beta_neta'].abs().max()):>22.2f}")
    say("\n  beta neta cerca de 0 = el mercado en general ya no mueve el L-S.")
    say("  Si B esta mucho mas cerca de 0 que A, el ajuste esta cumpliendo su proposito.")

    say("")
    say("-" * 96)
    say("EXPOSICION NETA POR SECTOR -- cartera B, ultima fecha de rebalanceo")
    say("-" * 96)
    ultimo_rebal = [f for f in fechas_rebalanceo(fechas, "mensual")][-1]
    tabla = expo_sec.filter(pl.col("date") == ultimo_rebal).sort("sector")
    say(f"  {'sector':<18}{'largo':>10}{'corto':>10}{'neta':>10}")
    for row in tabla.iter_rows(named=True):
        say(f"  {row['sector']:<18}{row['largo']:>10.3f}{row['corto']:>10.3f}{row['expo_neta']:>10.3f}")
    expo_prom = (expo_sec.group_by("sector").agg(pl.col("expo_neta").abs().mean().alias("m"))
                 .sort("m", descending=True))
    say(f"\n  |exposicion neta| promedio por sector, todas las fechas (0 = perfectamente neutral):")
    for row in expo_prom.iter_rows(named=True):
        say(f"    {row['sector']:<18}{row['m']:>8.3f}")

    say("")
    say("-" * 96)
    say("CURVAS DE CAPITAL -- A vs B, neto, base 100")
    say("-" * 96)
    ca = 100 * np.cumprod(1 + s_a["ret_neto"].to_numpy())
    cb = 100 * np.cumprod(1 + s_b["ret_neto"].to_numpy())
    fa = s_a["date"].to_list()
    say(f"  {'fecha':<12}{'A original':>12}{'B neutral':>12}")
    for i in range(0, len(ca), max(1, len(ca) // 20)):
        say(f"  {str(fa[i]):<12}{ca[i]:>12.1f}{cb[i]:>12.1f}")
    say(f"  {'final':<12}{ca[-1]:>12.1f}{cb[-1]:>12.1f}")

    say("")
    say(f"  IC95% ret. anual NETO -- A: [{m_a['ic95_lo']*100:+.1f}%, {m_a['ic95_hi']*100:+.1f}%]"
        f"   B: [{m_b['ic95_lo']*100:+.1f}%, {m_b['ic95_hi']*100:+.1f}%]")
    say("  (con ~50 observaciones semanales el intervalo sigue siendo muy ancho en ambos casos --")
    say("   esto compara si el SESGO estructural se corrigio, no da significancia estadistica.)")
    say("=" * 96)

    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    s_b.write_parquet(ROOT / "data/curated/backtest_neutral.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
