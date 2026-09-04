#!/usr/bin/env python3
"""PUERTA P2b -- Estabilidad del ranking de Gamma ante la incertidumbre de datos.

POR QUE ESTE TEST ES EL QUE IMPORTA
El hallazgo H1 dice que Gamma es un residuo entre la gamma de las calls y la de
los puts, que casi se cancelan (net/gross mediano 0.24). Cualquier error en las
entradas se amplifica por el inverso de esa razon. Y sabemos que quedan tres
fuentes de incertidumbre sin cerrar:

  1. La tasa. El estimador por paridad da r = 1.53%, que es demasiado bajo para
     2025-26. La causa esta diagnosticada: en opciones AMERICANAS el put ITM carga
     prima de ejercicio anticipado que crece con el strike, lo que hace la recta
     C-P vs K mas inclinada, sobreestima el factor de descuento y subestima r.
     La columna K<=S salio en -2%/-3% sistematico, que es la firma de ese sesgo.
  2. Los dividendos. Con la tasa sesgada, el carry se empuja a cero incluso en
     pagadores conocidos (AAPL, MSFT, JPM, WMT dieron 0.00%).
  3. El ruido de la IV y del open interest.

La pregunta que decide si el proyecto sigue NO es "cual es el valor exacto de r".
Es: **¿el ORDENAMIENTO de las acciones cambia?** La estrategia solo usa el ranking.
Si el quintil largo y el corto son los mismos bajo supuestos muy distintos, la
incertidumbre es de segundo orden y se puede avanzar declarandola. Si el ranking
se reordena, la estrategia no es implementable con estos datos, por mas que el
paper la respalde.

USO
    python3 run_sensibilidad.py            # 14 fechas (rapido, ~8 min)
    python3 run_sensibilidad.py --todas    # las 56 fechas
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import time

import numpy as np
import polars as pl
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.curves import fetch_treasury_curve
from gex.equities import load_equities
from gex.signal.gamma_exposure import add_adv, aggregate, prepare, solve_greeks
from gex.signal.implied_carry import attach_carry, implied_carry

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "p2b_sensibilidad.txt"
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


# Yields de dividendo de referencia para el escenario "manual". NO son cifras
# declaradas que yo pueda garantizar: son ordenes de magnitud tipicos del sector
# para estos nombres, y se usan SOLO para estresar el supuesto, no como dato.
Q_MANUAL = {"XOM": 0.035, "CVX": 0.045, "KO": 0.030, "PG": 0.025, "JPM": 0.022,
            "BAC": 0.025, "WMT": 0.010, "CAT": 0.015, "GS": 0.020, "DIS": 0.008,
            "AAPL": 0.005, "MSFT": 0.008, "NVDA": 0.0003, "META": 0.003,
            "MU": 0.005, "CRM": 0.006, "GME": 0.030, "RIVN": 0.020,
            "SOFI": 0.015, "MSTR": 0.010, "COIN": 0.005}


def escenario(pre, nombre, r=None, q_map=None, carry=None, r_curve=None,
              iv_noise=0.0, oi_noise=0.0, seed=0):
    df = pre
    if carry is not None:
        df = attach_carry(df, carry, r_curve=r_curve)
    else:
        df = df.with_columns(pl.lit(r).alias("r"))
        if q_map:
            qexpr = pl.col("underlying").replace_strict(q_map, default=0.0)
            df = df.with_columns((pl.col("close") * qexpr * pl.col("T")).alias("div_pv"))
        else:
            df = df.with_columns(pl.lit(0.0).alias("div_pv"))
    if oi_noise:
        rng = np.random.default_rng(seed)
        m = rng.normal(1.0, oi_noise, df.height)
        df = df.with_columns((pl.col("open_interest") * pl.Series(m)).alias("open_interest"))
    gr = solve_greeks(df)
    if iv_noise:
        rng = np.random.default_rng(seed + 1)
        sh = rng.normal(0.0, iv_noise, gr.height)
        gr = gr.with_columns((pl.col("iv") + pl.Series(sh)).clip(0.02, 5.0).alias("iv"))
        from gex.pricing.crr import crr_vec
        n = gr.height
        _, _, g2 = crr_vec(gr["close"].to_numpy().astype(float),
                           gr["strike"].to_numpy().astype(float),
                           gr["T"].to_numpy().astype(float),
                           gr["r"].to_numpy().astype(float),
                           gr["iv"].to_numpy().astype(float),
                           gr["div_pv"].to_numpy().astype(float),
                           gr["is_call"].to_numpy().astype(bool),
                           np.ones(n, dtype=np.bool_), 400)
        gr = gr.with_columns(pl.Series("gamma", g2)).filter(
            pl.col("gamma").is_finite() & (pl.col("gamma") >= 0))
    g = aggregate(gr).filter(~pl.col("underlying").is_in(INDICES))
    return nombre, g.select(["date", "underlying", "gamma_exposure"])


def comparar(base, otro, k=6):
    """Spearman promedio y solapamiento de quintiles, fecha por fecha."""
    j = base.join(otro, on=["date", "underlying"], how="inner",
                  suffix="_b")
    rs, ov_lo, ov_hi = [], [], []
    for _, g in j.group_by("date"):
        if g.height < 12:
            continue
        a = g["gamma_exposure"].to_numpy()
        b = g["gamma_exposure_b"].to_numpy()
        r = spearmanr(a, b).statistic
        if r == r:
            rs.append(r)
        ga = g.sort("gamma_exposure")
        gb = g.sort("gamma_exposure_b")
        lo_a, lo_b = set(ga.head(k)["underlying"]), set(gb.head(k)["underlying"])
        hi_a, hi_b = set(ga.tail(k)["underlying"]), set(gb.tail(k)["underlying"])
        ov_lo.append(len(lo_a & lo_b) / k)
        ov_hi.append(len(hi_a & hi_b) / k)
    return (float(np.mean(rs)), float(np.mean(ov_lo)), float(np.mean(ov_hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "data/raw/opra_chain/date=*/chain.parquet")))
    if not a.todas:
        files = files[::4]
    COLS = ["date", "underlying", "instrument_id", "raw_symbol", "expiration",
            "strike", "klass", "multiplier", "is_call", "open_interest",
            "bid", "ask", "mid", "rel_spread"]
    chain = pl.concat([pl.read_parquet(f).select(COLS) for f in files],
                      how="vertical_relaxed")
    eq = load_equities(ROOT)
    pre = prepare(chain, add_adv(eq))

    say("=" * 92)
    say("PUERTA P2b -- ESTABILIDAD DEL RANKING DE GAMMA".center(92))
    say(f"{len(files)} fechas · {pre.height:,} contratos".center(92))
    say("=" * 92)

    # base = la configuracion de produccion: r observada del Tesoro y dividendo
    # + borrow de la paridad. B conserva el metodo viejo para medir cuanto movio.
    curva = fetch_treasury_curve(ROOT)
    carry = implied_carry(pre, r_curve=curva)
    carry_viejo = implied_carry(pre)
    base_kw = dict(carry=carry, r_curve=curva)
    escenarios = [
        ("A base:  curva Tesoro + div. de paridad", dict(**base_kw)),
        ("B metodo viejo: r de la paridad",         dict(carry=carry_viejo)),
        ("C r=3.0%, dividendos manuales",           dict(r=0.030, q_map=Q_MANUAL)),
        ("D r=5.0%, dividendos manuales x2",        dict(r=0.050,
              q_map={k: v * 2 for k, v in Q_MANUAL.items()})),
        ("E base + ruido IV 0.5 pt vol",            dict(**base_kw, iv_noise=0.005)),
        ("F base + ruido OI 1%",                    dict(**base_kw, oi_noise=0.01)),
        ("G base + ruido OI 5%",                    dict(**base_kw, oi_noise=0.05)),
    ]
    res = {}
    for nombre, kw in escenarios:
        t0 = time.time()
        n, g = escenario(pre, nombre, **kw)
        res[n] = g
        say(f"  calculado: {n:<42} {time.time()-t0:>5.0f}s   "
            f"Gamma mediana {float(g['gamma_exposure'].median()):.4f}")

    base = res[escenarios[0][0]]
    say("")
    say("-" * 92)
    say("COMPARACION CONTRA EL ESCENARIO BASE")
    say("-" * 92)
    say("  Spearman = correlacion de rangos de Gamma entre los 30 nombres, promedio")
    say("  por fecha. Solapamiento = fraccion de los 6 nombres del quintil que")
    say("  coinciden. 1.00 = ranking identico.")
    say("")
    say(f"  {'escenario':<42}{'Spearman':>10}{'quintil L':>11}{'quintil H':>11}")
    say(f"  {'-'*42}{'-'*10}{'-'*11}{'-'*11}")
    peor = 1.0
    for nombre, _ in escenarios[1:]:
        r, lo, hi = comparar(base, res[nombre])
        peor = min(peor, r)
        say(f"  {nombre:<42}{r:>10.4f}{lo*100:>10.0f}%{hi*100:>10.0f}%")

    say("")
    say("=" * 92)
    say("VEREDICTO")
    say("=" * 92)
    if peor > 0.98:
        say(f"  Spearman minimo {peor:.4f} > 0.98")
        say("  El ranking es INSENSIBLE a los supuestos de tasa, dividendos y al ruido")
        say("  de IV y open interest en los rangos probados. La incertidumbre que queda")
        say("  en el carry es de segundo orden PARA LA SENAL, y se puede avanzar al")
        say("  test de mecanismo declarandola.")
    elif peor > 0.90:
        say(f"  Spearman minimo {peor:.4f}: el ranking se mueve poco pero se mueve.")
        say("  Se puede avanzar, pero todo resultado debe reportarse con esta")
        say("  sensibilidad al lado, y hay que cerrar la tasa con una curva externa.")
    else:
        say(f"  Spearman minimo {peor:.4f}: EL RANKING NO ES ESTABLE.")
        say("  La estrategia no es implementable con esta calidad de datos. Antes de")
        say("  seguir hay que cerrar la tasa y los dividendos con fuentes externas.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 92)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
