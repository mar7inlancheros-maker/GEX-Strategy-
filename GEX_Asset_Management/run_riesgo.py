#!/usr/bin/env python3
"""ANALISIS DE RIESGO -- VaR, CVaR, bootstrap y comportamiento en recesion.

QUE PUEDE Y QUE NO PUEDE ESTE ANALISIS
El bootstrap remuestrea las 266 semanas observadas. Eso mide la INCERTIDUMBRE
MUESTRAL -- cuan ancha es la distribucion alrededor del retorno estimado -- y es
informacion real. Lo que NO hace es inventar regimenes que no estan en la
muestra: si en 2021-2026 no hubo una crisis tipo 2008, ningun remuestreo la va
a producir. Por eso el bloque C mira el unico episodio de estres que SI esta en
los datos (2022, S&P -19%) y el bloque D condiciona a las semanas en que el
mercado cayo. Eso vale mas que simular recesiones imaginarias.

Se usa bootstrap por BLOQUES (estacionario, Politis-Romano): remuestrear
semanas sueltas destruiria la autocorrelacion y los clusters de volatilidad,
que son justo lo que produce los drawdowns. Con bloques se conservan.

USO
    python3 run_riesgo.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.backtest.engine import semanal_returns
from gex.equities import load_equities

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "riesgo.txt"
N_SIM = 10_000
BLOQUE_MEDIO = 8          # semanas; ~2 meses de memoria
SEM = 52
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def max_dd(r):
    c = np.cumprod(1 + r)
    return float((c / np.maximum.accumulate(c) - 1).min())


def metricas(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    ret = float(np.prod(1 + r) ** (SEM / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(SEM))
    return ret, vol, (ret / vol if vol > 0 else np.nan), max_dd(r)


def bootstrap_bloques(r, n_sim=N_SIM, p=1.0 / BLOQUE_MEDIO, seed=0):
    """Bootstrap estacionario: bloques de largo geometrico, indice circular."""
    rng = np.random.default_rng(seed)
    n = len(r)
    out = np.empty((n_sim, n))
    for s in range(n_sim):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[s] = r[idx]
    return out


def var_cvar(r, q):
    """VaR y CVaR historicos al nivel q (perdida en la cola izquierda)."""
    v = float(np.quantile(r, 1 - q))
    c = float(r[r <= v].mean()) if (r <= v).any() else v
    return v, c


def main():
    eq = load_equities(ROOT)
    ls = pl.read_parquet(ROOT / "data/curated/backtest_ls.parquet")
    nt = pl.read_parquet(ROOT / "data/curated/backtest_neutral.parquet")
    fechas = ls["date"].to_list()
    rets = semanal_returns(eq, fechas)

    spy = (rets.filter(pl.col("symbol") == "SPY").sort("date")
              .select(["date", "ret"]).rename({"ret": "spy"}))
    bench = (rets.filter(~pl.col("symbol").is_in(INDICES)).group_by("date")
                 .agg(pl.col("ret").mean().alias("bench")).sort("date"))

    df = (ls.select(["date", pl.col("ret_neto").alias("ls")])
            .join(nt.select(["date", pl.col("ret_neto").alias("neutral")]), on="date", how="inner")
            .join(spy, on="date", how="inner")
            .join(bench, on="date", how="inner").sort("date"))
    carteras = ["ls", "neutral", "bench", "spy"]
    ETIQ = {"ls": "L-S semanal", "neutral": "L-S neutral",
            "bench": "30 nombres EW", "spy": "SPY"}

    say("=" * 100)
    say("ANALISIS DE RIESGO -- VaR, CVaR, bootstrap y estres".center(100))
    say(f"{df.height} semanas · {df['date'].min()} -> {df['date'].max()}".center(100))
    say("=" * 100)

    # ------------------------------------------------------------ A. VaR/CVaR
    say("")
    say("-" * 100)
    say("A. VaR y CVaR HISTORICOS (semanal, sobre la distribucion observada)")
    say("-" * 100)
    say("   VaR 95% = la perdida que solo se supera 1 de cada 20 semanas.")
    say("   CVaR 95% = la perdida MEDIA en esas peores semanas. Es la que importa:")
    say("   el VaR te dice donde empieza la cola, el CVaR cuanto duele dentro.")
    say("")
    say(f"  {'cartera':<16}{'VaR 95%':>10}{'CVaR 95%':>11}{'VaR 99%':>10}"
        f"{'CVaR 99%':>11}{'peor sem':>10}{'vol anual':>11}")
    say(f"  {'-'*16}{'-'*10}{'-'*11}{'-'*10}{'-'*11}{'-'*10}{'-'*11}")
    for c in carteras:
        r = df[c].to_numpy()
        v95, c95 = var_cvar(r, 0.95)
        v99, c99 = var_cvar(r, 0.99)
        say(f"  {ETIQ[c]:<16}{v95*100:>9.2f}%{c95*100:>10.2f}%{v99*100:>9.2f}%"
            f"{c99*100:>10.2f}%{r.min()*100:>9.2f}%{r.std(ddof=1)*np.sqrt(SEM)*100:>10.1f}%")

    # -------------------------------------------------------- B. bootstrap
    say("")
    say("-" * 100)
    say(f"B. BOOTSTRAP POR BLOQUES ({N_SIM:,} simulaciones, bloque medio {BLOQUE_MEDIO} semanas)")
    say("-" * 100)
    say("   Cada simulacion re-arma 5 anos remuestreando bloques de semanas reales.")
    say("   Mide la incertidumbre muestral: si repitieras el experimento, que rango")
    say("   de resultados verias. NO simula crisis que no estan en los datos.")
    say("")
    say(f"  {'cartera':<16}{'ret medio':>11}{'IC90 retorno':>22}"
        f"{'P(ret<0)':>10}{'DD mediano':>12}{'DD p95':>10}")
    say(f"  {'-'*16}{'-'*11}{'-'*22}{'-'*10}{'-'*12}{'-'*10}")
    boot = {}
    for c in carteras:
        r = df[c].to_numpy()
        sims = bootstrap_bloques(r, seed=hash(c) % 2**31)
        m = np.array([metricas(s) for s in sims])
        boot[c] = m
        ret, dd = m[:, 0], m[:, 3]
        say(f"  {ETIQ[c]:<16}{ret.mean()*100:>10.2f}%"
            f"{f'[{np.percentile(ret,5)*100:+.1f}%, {np.percentile(ret,95)*100:+.1f}%]':>22}"
            f"{(ret<0).mean()*100:>9.1f}%{np.median(dd)*100:>11.1f}%"
            f"{np.percentile(dd,5)*100:>9.1f}%")

    say("")
    say("  Probabilidad de que cada cartera le gane a la alternativa pasiva:")
    for c in ["ls", "neutral", "bench"]:
        p_spy = (boot[c][:, 0] > boot["spy"][:, 0]).mean() * 100
        p_sh = (boot[c][:, 2] > boot["spy"][:, 2]).mean() * 100
        say(f"    {ETIQ[c]:<16} P(retorno > SPY) = {p_spy:>5.1f}%    "
            f"P(Sharpe > SPY) = {p_sh:>5.1f}%")
    say("  (50% = indistinguible de tirar una moneda)")

    # ----------------------------------------------------- C. estres 2022
    say("")
    say("-" * 100)
    say("C. EL UNICO EPISODIO DE ESTRES REAL EN LA MUESTRA: 2022")
    say("-" * 100)
    say("   S&P -19% en el ano. Es la prueba honesta de comportamiento en recesion,")
    say("   y vale mas que cualquier recesion simulada.")
    say("")
    periodos = [("2022 completo", "2022-01-01", "2022-12-31"),
                ("caida ene-oct 2022", "2022-01-01", "2022-10-14"),
                ("resto de la muestra", "2023-01-01", "2026-12-31")]
    say(f"  {'periodo':<22}{'cartera':<16}{'retorno':>10}{'vol':>9}"
        f"{'Sharpe':>9}{'max DD':>10}{'peor sem':>10}")
    say(f"  {'-'*22}{'-'*16}{'-'*10}{'-'*9}{'-'*9}{'-'*10}{'-'*10}")
    for nom, a, b in periodos:
        sub = df.filter(pl.col("date").is_between(pl.lit(a).str.to_date(),
                                                  pl.lit(b).str.to_date()))
        if sub.height < 5:
            continue
        for i, c in enumerate(carteras):
            r = sub[c].to_numpy()
            ret, vol, sh, dd = metricas(r)
            say(f"  {nom if i==0 else '':<22}{ETIQ[c]:<16}{ret*100:>9.1f}%{vol*100:>8.1f}%"
                f"{sh:>9.2f}{dd*100:>9.1f}%{r.min()*100:>9.2f}%")
        say("")

    # ------------------------------------------ D. condicional a mercado abajo
    say("-" * 100)
    say("D. COMPORTAMIENTO CONDICIONAL AL MERCADO -- la pregunta que importa para cubrir")
    say("-" * 100)
    say("   Una estrategia vale como diversificador si gana (o pierde poco) justo")
    say("   cuando el mercado cae. Aqui se separa por lo que hizo SPY esa semana.")
    say("")
    cortes = [("SPY < -3%", pl.col("spy") < -0.03),
              ("SPY entre -3% y 0%", (pl.col("spy") >= -0.03) & (pl.col("spy") < 0)),
              ("SPY > 0%", pl.col("spy") >= 0)]
    say(f"  {'regimen':<22}{'n sem':>7}{'L-S semanal':>14}{'L-S neutral':>14}"
        f"{'30 nombres':>13}{'SPY':>10}")
    say(f"  {'-'*22}{'-'*7}{'-'*14}{'-'*14}{'-'*13}{'-'*10}")
    for nom, cond in cortes:
        sub = df.filter(cond)
        if sub.is_empty():
            continue
        say(f"  {nom:<22}{sub.height:>7}"
            f"{float(sub['ls'].mean())*100:>13.2f}%{float(sub['neutral'].mean())*100:>13.2f}%"
            f"{float(sub['bench'].mean())*100:>12.2f}%{float(sub['spy'].mean())*100:>9.2f}%")
    say("")
    for c in ["ls", "neutral"]:
        r, s = df[c].to_numpy(), df["spy"].to_numpy()
        beta = float(np.polyfit(s, r, 1)[0])
        dn = s < 0
        beta_dn = float(np.polyfit(s[dn], r[dn], 1)[0])
        say(f"  {ETIQ[c]:<16} beta total {beta:+.2f}   beta en semanas de caida {beta_dn:+.2f}"
            f"   correlacion {float(np.corrcoef(r, s)[0,1]):+.2f}")
    say("  (beta negativa en caidas = cubre; positiva = amplifica la perdida)")

    # ----------------------------------------------------------- veredicto
    say("")
    say("=" * 100)
    say("LECTURA")
    say("=" * 100)
    p_ret = (boot["neutral"][:, 0] > boot["spy"][:, 0]).mean() * 100
    cv_n = var_cvar(df["neutral"].to_numpy(), 0.95)[1]
    cv_s = var_cvar(df["spy"].to_numpy(), 0.95)[1]
    say(f"  La version neutral tiene CVaR 95% de {cv_n*100:.2f}% semanal contra "
        f"{cv_s*100:.2f}% de SPY.")
    say(f"  El bootstrap le da {p_ret:.0f}% de probabilidad de superar a SPY en retorno.")
    say("")
    say("  Recorda que esto NO es un test de la senal: el analisis de riesgo describe")
    say("  la distribucion de una estrategia cuyo mecanismo ya quedo sin confirmar")
    say("  en P3, P3b y el test de ortogonalizacion. Un perfil de riesgo atractivo")
    say("  sobre una senal sin mecanismo sigue sin ser razon para operarla.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 100)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
