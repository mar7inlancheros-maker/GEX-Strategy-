#!/usr/bin/env python3
"""Momentum intradia condicionado por gamma -- test pre-registrado.

COSTO: $0. Solo lee parquets locales. No importa databento, no hace llamadas.

BASE TEORICA
Baltussen, Da, Lammers & Martens (2021), JFE, "Hedging Demand and Market
Intraday Momentum": el retorno del ultimo tramo del dia es predicho por el
retorno del resto del dia, via la cobertura de gamma de los dealers. El efecto
REVIERTE en los dias siguientes -- por eso nunca aparecio en nuestros tests
semanales. Lo mediamos despues de que se lavara.

HIPOTESIS
Dealers cortos gamma -> cubren a favor -> el gap de apertura CONTINUA.
Dealers largos gamma -> cubren en contra -> el gap REVIERTE.

Pre-registro congelado en scratchpad/preregistro_intradia.md. Umbral z +-0.5,
ventana 26 semanas, costos 5 pb (SPY) y 10 pb (acciones). Nada se toca.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.equities import load_equities

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "intradia.txt"
Z_TH, WIN, DIA = 0.5, 26, 252
COST_IDX, COST_ACC = 0.0005, 0.0010
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def nw(x, lags=5):
    import statsmodels.api as sm
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan
    r = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0])


def reg(y, x, lags=5):
    """OLS y ~ x con Newey-West. Devuelve (beta, t, n)."""
    import statsmodels.api as sm
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 20:
        return np.nan, np.nan, int(m.sum())
    r = sm.OLS(y[m], sm.add_constant(x[m])).fit(cov_type="HAC",
                                                cov_kwds={"maxlags": lags})
    return float(r.params[1]), float(r.tvalues[1]), int(m.sum())


def metricas(r, per=DIA):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std() == 0:
        return dict(ret=np.nan, vol=np.nan, sharpe=np.nan, dd=np.nan, t=np.nan, n=n)
    ret = np.prod(1 + r) ** (per / n) - 1
    vol = r.std(ddof=1) * np.sqrt(per)
    c = np.cumprod(1 + r)
    dd = float((c / np.maximum.accumulate(c) - 1).min())
    _, t = nw(r)
    return dict(ret=ret, vol=vol, sharpe=r.mean() * per / vol, dd=dd, t=t, n=n)


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)

    say("=" * 100)
    say("MOMENTUM INTRADIA CONDICIONADO POR GAMMA -- test pre-registrado".center(100))
    say("Baltussen, Da, Lammers & Martens (2021, JFE) · costo del test: $0".center(100))
    say("=" * 100)

    # ---------- tramos diarios
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("d"))
           .sort(["symbol", "d"])
           .with_columns(pl.col("close").shift(1).over("symbol").alias("prev"))
           .drop_nulls("prev")
           .with_columns([(pl.col("open") / pl.col("prev") - 1).alias("gap"),
                          (pl.col("close") / pl.col("open") - 1).alias("intra")])
           .filter(pl.col("gap").abs() < 0.25))
    say(f"\ndias-accion con gap e intradia: {d.height:,} · "
        f"{d['d'].n_unique()} dias · {d['symbol'].n_unique()} simbolos")

    # ---------- gamma semanal -> propagado a dias POSTERIORES (sin look-ahead)
    fechas = sorted(g["date"].unique().to_list())
    idx = (g.filter(pl.col("underlying").is_in(INDICES))
             .group_by("date").agg(pl.col("gamma_exposure").sum().alias("G")).sort("date"))
    G, fx = idx["G"].to_numpy(), idx["date"].to_list()
    z = np.full(len(G), np.nan)
    for i in range(len(G)):
        lo = max(0, i - WIN)
        if i - lo >= 8:
            w = G[lo:i]
            z[i] = (G[i] - w.mean()) / (w.std() + 1e-12)
    zidx = dict(zip(fx, z))

    dias = sorted(d["d"].unique().to_list())
    def propagar(zmap_por_fecha):
        """A cada dia habil le asigna el z de la ULTIMA fecha de snapshot ANTERIOR."""
        out, j, cur = {}, 0, np.nan
        for dd in dias:
            while j < len(fechas) and fechas[j] < dd:
                cur = zmap_por_fecha.get(fechas[j], np.nan); j += 1
            out[dd] = cur
        return out
    zdia = propagar(zidx)

    def regimen(zv):
        if not np.isfinite(zv):
            return "neutral"
        return "corto" if zv <= -Z_TH else ("largo" if zv >= Z_TH else "neutral")

    # ================================================================= INDICE
    say("")
    say("=" * 100)
    say("NIVEL INDICE -- SPY")
    say("=" * 100)
    spy = d.filter(pl.col("symbol") == "SPY").sort("d")
    fs = spy["d"].to_list()
    gap = spy["gap"].to_numpy(); intra = spy["intra"].to_numpy()
    regs = np.array([regimen(zdia.get(f, np.nan)) for f in fs])
    from collections import Counter
    say(f"\ndias operables: {len(fs)} · {dict(Counter(regs))}")

    say("")
    say("-" * 100)
    say("A. LA REGRESION -- predice el gap el movimiento intradia?")
    say("-" * 100)
    say("   Hipotesis: beta POSITIVO en gamma corto (continua), NEGATIVO en largo (revierte).")
    say("")
    say(f"  {'regimen':<24}{'beta':>10}{'t (NW)':>10}{'n':>8}")
    say(f"  {'-'*24}{'-'*10}{'-'*10}{'-'*8}")
    betas = {}
    for rg in ["corto", "neutral", "largo", "TODOS"]:
        m = np.ones(len(fs), bool) if rg == "TODOS" else (regs == rg)
        b, t, n = reg(intra[m], gap[m])
        betas[rg] = b
        say(f"  {rg:<24}{b:>10.4f}{t:>10.2f}{n:>8}")
    # diferencia corto - largo con interaccion
    import statsmodels.api as sm
    mm = np.isin(regs, ["corto", "largo"])
    dum = (regs[mm] == "corto").astype(float)
    X = np.column_stack([gap[mm], dum, gap[mm] * dum])
    rr = sm.OLS(intra[mm], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    dif_b, dif_t = float(rr.params[3]), float(rr.tvalues[3])
    say(f"\n  interaccion gap x (corto gamma): beta {dif_b:+.4f}   t = {dif_t:+.2f}")
    say(f"  (= cuanto MAS positivo es el efecto del gap en regimen corto)")

    say("")
    say("-" * 100)
    say("B. LA ESTRATEGIA -- entrar en apertura, salir en cierre, solo spot")
    say("-" * 100)
    sg = np.sign(gap)
    pos = {"A": np.where(regs == "corto", sg, np.where(regs == "largo", -sg, 0.0)),
           "B": sg, "C": -sg,
           "D": np.where(regs == "corto", -sg, np.where(regs == "largo", sg, 0.0))}
    ser = {k: v * intra - np.abs(v) * COST_IDX for k, v in pos.items()}
    M = {k: metricas(v) for k, v in ser.items()}
    say(f"  {'estrategia':<42}{'ret anual':>10}{'vol':>8}{'Sharpe':>8}{'max DD':>9}{'t (NW)':>9}")
    say(f"  {'-'*42}{'-'*10}{'-'*8}{'-'*8}{'-'*9}{'-'*9}")
    for k, lab in [("A", "A. Condicionada por gamma"), ("B", "B. Continuacion incondicional"),
                   ("C", "C. Reversion incondicional"), ("D", "D. A invertida (placebo)")]:
        m = M[k]
        say(f"  {lab:<42}{m['ret']*100:>9.2f}%{m['vol']*100:>7.1f}%"
            f"{m['sharpe']:>8.2f}{m['dd']*100:>8.1f}%{m['t']:>9.2f}")

    rng = np.random.default_rng(20260903)
    sh = []
    for _ in range(1000):
        rp = rng.permutation(regs)
        p = np.where(rp == "corto", sg, np.where(rp == "largo", -sg, 0.0))
        sh.append(metricas(p * intra - np.abs(p) * COST_IDX)["sharpe"])
    sh = np.array(sh); pctl_i = float((sh < M["A"]["sharpe"]).mean() * 100)
    say(f"\n  placebo aleatorio (1000): media {np.nanmean(sh):.3f} · "
        f"p95 {np.nanpercentile(sh,95):.3f} · A en percentil {pctl_i:.1f}")

    yrs = np.array([f.year for f in fs])
    say(f"\n  {'periodo':<20}{'ret A':>10}{'Sharpe A':>10}")
    for y in sorted(set(yrs)):
        m = yrs == y
        if m.sum() < 40:
            continue
        mm2 = metricas(ser["A"][m])
        say(f"  {y:<20}{mm2['ret']*100:>9.1f}%{mm2['sharpe']:>10.2f}")
    m_no22 = metricas(ser["A"][yrs != 2022])
    say(f"  {'SIN 2022':<20}{m_no22['ret']*100:>9.1f}%{m_no22['sharpe']:>10.2f}")

    # ================================================================ ACCIONES
    say("")
    say("=" * 100)
    say("NIVEL ACCION -- las 30, gamma cross-seccional")
    say("=" * 100)
    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    zx = (xs.with_columns(
            ((pl.col("gamma_exposure") - pl.col("gamma_exposure").mean().over("date"))
             / (pl.col("gamma_exposure").std().over("date") + 1e-12)).alias("zc"))
            .select(["date", "underlying", "zc"]))
    # a cada dia-accion, el z de la ultima fecha de snapshot anterior
    zmap = {(r["date"], r["underlying"]): r["zc"] for r in zx.iter_rows(named=True)}
    prev_f = {}
    j, cur = 0, None
    for dd in dias:
        while j < len(fechas) and fechas[j] < dd:
            cur = fechas[j]; j += 1
        prev_f[dd] = cur
    da = d.filter(~pl.col("symbol").is_in(INDICES))
    zz = np.array([zmap.get((prev_f.get(r0), r1), np.nan)
                   for r0, r1 in zip(da["d"].to_list(), da["symbol"].to_list())])
    da = da.with_columns(pl.Series("zc", zz)).drop_nulls("zc")
    gp = da["gap"].to_numpy(); it = da["intra"].to_numpy()
    rg2 = np.where(da["zc"].to_numpy() <= -Z_TH, "corto",
                   np.where(da["zc"].to_numpy() >= Z_TH, "largo", "neutral"))
    say(f"\ndias-accion con gamma: {da.height:,} · {dict(Counter(rg2))}")
    say("")
    say(f"  {'regimen':<24}{'beta':>10}{'t (NW)':>10}{'n':>8}")
    say(f"  {'-'*24}{'-'*10}{'-'*10}{'-'*8}")
    for rgx in ["corto", "neutral", "largo", "TODOS"]:
        m = np.ones(len(gp), bool) if rgx == "TODOS" else (rg2 == rgx)
        b, t, n = reg(it[m], gp[m])
        say(f"  {rgx:<24}{b:>10.4f}{t:>10.2f}{n:>8}")
    mm = np.isin(rg2, ["corto", "largo"])
    dum = (rg2[mm] == "corto").astype(float)
    X = np.column_stack([gp[mm], dum, gp[mm] * dum])
    rr2 = sm.OLS(it[mm], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    dif_b2, dif_t2 = float(rr2.params[3]), float(rr2.tvalues[3])
    say(f"\n  interaccion gap x (corto gamma): beta {dif_b2:+.4f}   t = {dif_t2:+.2f}")

    # cartera diaria equiponderada por accion
    sg2 = np.sign(gp)
    pa = np.where(rg2 == "corto", sg2, np.where(rg2 == "largo", -sg2, 0.0))
    dd_df = da.with_columns([pl.Series("p", pa), pl.Series("r", pa * it - np.abs(pa) * COST_ACC)])
    port = (dd_df.filter(pl.col("p") != 0).group_by("d")
                 .agg(pl.col("r").mean().alias("ret")).sort("d"))
    pb = (da.with_columns(pl.Series("r", sg2 * it - COST_ACC)).group_by("d")
            .agg(pl.col("r").mean().alias("ret")).sort("d"))
    pc = (da.with_columns(pl.Series("r", -sg2 * it - COST_ACC)).group_by("d")
            .agg(pl.col("r").mean().alias("ret")).sort("d"))
    say("")
    say(f"  {'cartera diaria equiponderada':<42}{'ret anual':>10}{'vol':>8}{'Sharpe':>8}{'max DD':>9}{'t (NW)':>9}")
    say(f"  {'-'*42}{'-'*10}{'-'*8}{'-'*8}{'-'*9}{'-'*9}")
    MA = {}
    for k, lab, s in [("A", "A. Condicionada por gamma", port), ("B", "B. Continuacion", pb),
                      ("C", "C. Reversion", pc)]:
        m = metricas(s["ret"].to_numpy()); MA[k] = m
        say(f"  {lab:<42}{m['ret']*100:>9.2f}%{m['vol']*100:>7.1f}%"
            f"{m['sharpe']:>8.2f}{m['dd']*100:>8.1f}%{m['t']:>9.2f}")
    ya = np.array([x.year for x in port["d"].to_list()])
    ra = port["ret"].to_numpy()
    m_no22a = metricas(ra[ya != 2022])
    say(f"\n  A sin 2022: ret {m_no22a['ret']*100:.2f}%  Sharpe {m_no22a['sharpe']:.2f}")

    # ---------------------------------------------------------------- veredicto
    say("")
    say("=" * 100)
    say("VEREDICTO CONTRA EL CRITERIO PRE-REGISTRADO")
    say("=" * 100)
    for nivel, dt_, sh_a, sh_b, sh_c, sh_d, pctl, no22 in [
            ("INDICE (SPY)", dif_t, M["A"]["sharpe"], M["B"]["sharpe"], M["C"]["sharpe"],
             M["D"]["sharpe"], pctl_i, m_no22["ret"]),
            ("ACCION (30)", dif_t2, MA["A"]["sharpe"], MA["B"]["sharpe"], MA["C"]["sharpe"],
             np.nan, np.nan, m_no22a["ret"])]:
        say(f"\n  {nivel}")
        cs = [(dt_ >= 2.0, f"1. interaccion gap x corto-gamma con t >= 2.0   (t = {dt_:+.2f})"),
              ((sh_a - sh_b >= 0.15) and (sh_a - sh_c >= 0.15),
               f"2. Sharpe A supera a B y C por >= 0.15   (A {sh_a:.2f} · B {sh_b:.2f} · C {sh_c:.2f})"),
              (sh_d < sh_a if np.isfinite(sh_d) else None,
               f"3. placebo invertido D peor que A   (D {sh_d:.2f} < A {sh_a:.2f})"),
              (pctl >= 95 if np.isfinite(pctl) else None,
               f"4. A sobre p95 del aleatorio   (percentil {pctl:.0f})" if np.isfinite(pctl) else "4. placebo aleatorio (solo indice)"),
              (no22 > 0, f"5. A positivo sin 2022   (ret {no22*100:+.2f}%)")]
        for ok, txt in cs:
            marca = "n/a " if ok is None else ("PASA" if ok else "FALLA")
            say(f"    [{marca}] {txt}")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 100)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
