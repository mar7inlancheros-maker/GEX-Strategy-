#!/usr/bin/env python3
"""GEX de indice como switch momentum / reversal -- test pre-registrado.

Pre-registro congelado el 2026-09-03 (scratchpad/preregistro_gex_regimen.md).
Los parametros NO se tocan: umbral z +-0.5, ventana 26 semanas, 6 por pata,
horizonte 1 semana, 5 bps por lado.

HIPOTESIS
El GEX agregado de SPY+QQQ dice si los dealers estan largos o cortos gamma.
Largos -> cubren contra tendencia -> el mercado revierte. Cortos -> cubren a
favor -> el mercado trend-ea. Una estrategia que hace reversal en regimen
largo y momentum en regimen corto deberia ganarle al momentum y al reversal
incondicionales.

GEX NO predice el retorno. Predice QUE señal funciona esa semana.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.curves import fetch_treasury_curve, rate_lookup
from gex.equities import load_equities

INDICES = ["SPY", "QQQ"]
REP = ROOT / "reports" / "regimen.txt"
Z_TH = 0.5
WIN = 26
N_PATA = 6
COST = 0.0005          # por lado
SEM = 52
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def nw_t(x, lags=4):
    import statsmodels.api as sm
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    r = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0])


def metricas(r, rf):
    r = np.asarray(r, float); n = len(r)
    ex = r - rf[:n]
    ret_a = np.prod(1 + r) ** (SEM / n) - 1
    vol = r.std(ddof=1) * np.sqrt(SEM)
    sharpe = ex.mean() * SEM / vol if vol > 0 else np.nan
    c = np.cumprod(1 + r)
    dd = float((c / np.maximum.accumulate(c) - 1).min())
    m, t = nw_t(r)
    return dict(ret=ret_a, vol=vol, sharpe=sharpe, dd=dd, t=t, n=n)


def sim(sig_por_fecha, fwd, fechas, costo=COST):
    """sig_por_fecha: dict fecha -> ('mom'|'rev'|'flat'). Devuelve serie de retornos."""
    rr, prev = [], None
    for i in range(1, len(fechas) - 1):
        t = fechas[i]
        modo = sig_por_fecha.get(t, "flat")
        if modo == "flat":
            rr.append(0.0); prev = None; continue
        # ranking por retorno t-1 -> t
        r_pasado = fwd.get((fechas[i - 1], t))
        r_fut = fwd.get((t, fechas[i + 1]))
        if r_pasado is None or r_fut is None:
            rr.append(0.0); prev = None; continue
        nombres = sorted(r_pasado, key=r_pasado.get)
        lo, hi = nombres[:N_PATA], nombres[-N_PATA:]
        if modo == "mom":
            largo, corto = hi, lo
        else:
            largo, corto = lo, hi
        ret = (np.mean([r_fut.get(s, 0) for s in largo])
               - np.mean([r_fut.get(s, 0) for s in corto]))
        # costo de turnover
        cur = set(largo) | set(corto)
        tn = 1.0 if prev is None else len(cur ^ prev) / len(cur)
        ret -= costo * 2 * tn
        rr.append(ret); prev = cur
    return np.array(rr)


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)
    curva = fetch_treasury_curve(ROOT)

    xs = g.filter(~pl.col("underlying").is_in(INDICES))
    fechas = sorted(g["date"].unique().to_list())

    say("=" * 96)
    say("GEX DE INDICE COMO SWITCH MOMENTUM / REVERSAL -- test pre-registrado".center(96))
    say(f"{len(fechas)} fechas semanales · umbral z +-{Z_TH} · ventana {WIN} sem".center(96))
    say("=" * 96)

    # ---- indicador de regimen: GEX(SPY) + GEX(QQQ)
    idx = (g.filter(pl.col("underlying").is_in(INDICES))
             .group_by("date").agg(pl.col("gamma_exposure").sum().alias("G"))
             .sort("date"))
    G = idx["G"].to_numpy()
    fx = idx["date"].to_list()
    z = np.full(len(G), np.nan)
    for i in range(len(G)):
        lo = max(0, i - WIN)
        if i - lo >= 8:
            w = G[lo:i]
            z[i] = (G[i] - w.mean()) / (w.std() + 1e-12)
    zmap = {fx[i]: z[i] for i in range(len(G))}

    reg = {}
    for f in fechas:
        zz = zmap.get(f, np.nan)
        if not np.isfinite(zz):
            reg[f] = "neutral"
        elif zz <= -Z_TH:
            reg[f] = "corto"      # dealers cortos gamma -> momentum
        elif zz >= Z_TH:
            reg[f] = "largo"      # dealers largos gamma -> reversal
        else:
            reg[f] = "neutral"
    from collections import Counter
    cc = Counter(reg.values())
    say(f"\nregimen: {cc['corto']} sem corto gamma · {cc['largo']} sem largo gamma · "
        f"{cc['neutral']} neutral")
    say(f"G(SPY+QQQ): media {G.mean():.3f}  ·  siempre negativo: {(G<0).all()}")

    # ---- retornos forward por par de fechas
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .filter(~pl.col("symbol").is_in(INDICES))
           .sort(["symbol", "date"]))
    close = {(r["symbol"], r["date"]): r["close"] for r in d.iter_rows(named=True)}
    syms = xs["underlying"].unique().to_list()
    fwd = {}
    for i in range(len(fechas) - 1):
        a, b = fechas[i], fechas[i + 1]
        dd = {}
        for s in syms:
            ca, cb = close.get((s, a)), close.get((s, b))
            if ca and cb and ca > 0:
                dd[s] = cb / ca - 1
        if dd:
            fwd[(a, b)] = dd

    # ---- rf semanal alineado
    rf_a = rate_lookup(curva, fechas[1:-1], np.full(len(fechas) - 2, 0.25))
    rf = np.expm1(rf_a / SEM)

    # ---- estrategias
    A = sim({f: {"corto": "mom", "largo": "rev", "neutral": "flat"}[reg[f]] for f in fechas}, fwd, fechas)
    B = sim({f: "mom" for f in fechas}, fwd, fechas)
    C = sim({f: "rev" for f in fechas}, fwd, fechas)
    D = sim({f: {"corto": "rev", "largo": "mom", "neutral": "flat"}[reg[f]] for f in fechas}, fwd, fechas)

    say("")
    say("-" * 96)
    say("RESULTADOS -- retorno semanal, neto de 5 bps por lado")
    say("-" * 96)
    say(f"  {'estrategia':<40}{'ret anual':>10}{'vol':>8}{'Sharpe':>8}{'max DD':>9}{'t (NW)':>9}")
    say(f"  {'-'*40}{'-'*10}{'-'*8}{'-'*8}{'-'*9}{'-'*9}")
    res = {}
    for nom, r in [("A. GEX-switch (mom si corto, rev si largo)", A),
                   ("B. momentum incondicional", B),
                   ("C. reversal incondicional", C),
                   ("D. GEX-switch INVERTIDO (placebo)", D)]:
        m = metricas(r, rf); res[nom[:1]] = m
        say(f"  {nom:<40}{m['ret']*100:>9.2f}%{m['vol']*100:>7.1f}%"
            f"{m['sharpe']:>8.2f}{m['dd']*100:>8.1f}%{m['t']:>9.2f}")

    # ---- placebo aleatorio
    say("")
    say("-" * 96)
    say("PLACEBO ALEATORIO -- 1000 barajados del vector de regimen")
    say("-" * 96)
    rng = np.random.default_rng(20260903)
    modos = [reg[f] for f in fechas]
    sh_rand = []
    for _ in range(1000):
        perm = list(rng.permutation(modos))
        rmap = {fechas[i]: {"corto": "mom", "largo": "rev", "neutral": "flat"}[perm[i]]
                for i in range(len(fechas))}
        rr = sim(rmap, fwd, fechas)
        sh_rand.append(metricas(rr, rf)["sharpe"])
    sh_rand = np.array(sh_rand)
    pctl = (sh_rand < res["A"]["sharpe"]).mean() * 100
    say(f"  Sharpe de A: {res['A']['sharpe']:.3f}")
    say(f"  distribucion aleatoria: media {sh_rand.mean():.3f}  "
        f"p95 {np.percentile(sh_rand,95):.3f}  max {sh_rand.max():.3f}")
    say(f"  A esta en el percentil {pctl:.1f} de los switches aleatorios")

    # ---- por año
    say("")
    say("-" * 96)
    say("A POR AÑO")
    say("-" * 96)
    fa = fechas[1:-1]
    say(f"  {'año':<8}{'ret A':>10}{'ret B (mom)':>13}{'ret C (rev)':>13}")
    aos = sorted(set(f.year for f in fa))
    a_pos = 0
    for yr in aos:
        idxy = [i for i, f in enumerate(fa) if f.year == yr]
        if len(idxy) < 5:
            continue
        ra = np.prod(1 + A[idxy]) - 1
        rb = np.prod(1 + B[idxy]) - 1
        rc = np.prod(1 + C[idxy]) - 1
        if ra > 0:
            a_pos += 1
        say(f"  {yr:<8}{ra*100:>9.1f}%{rb*100:>12.1f}%{rc*100:>12.1f}%")

    # ---- veredicto contra el criterio congelado
    say("")
    say("=" * 96)
    say("VEREDICTO CONTRA EL CRITERIO PRE-REGISTRADO")
    say("=" * 96)
    shA, shB, shC, shD = res['A']['sharpe'], res['B']['sharpe'], res['C']['sharpe'], res['D']['sharpe']
    c1 = (shA - shB >= 0.15) and (shA - shC >= 0.15)
    c2 = res['A']['t'] >= 2.0
    c3 = shD < shA
    c4 = pctl >= 95
    c5 = a_pos >= 3
    for ok, txt in [(c1, f"1. Sharpe A supera a B y a C por >= 0.15  (A {shA:.2f} · B {shB:.2f} · C {shC:.2f})"),
                    (c2, f"2. t de Newey-West de A >= 2.0  (t = {res['A']['t']:.2f})"),
                    (c3, f"3. placebo invertido D peor que A  (D {shD:.2f} < A {shA:.2f})"),
                    (c4, f"4. A sobre el p95 del placebo aleatorio  (percentil {pctl:.0f})"),
                    (c5, f"5. A positivo en >= 3 de 5 años  ({a_pos} años)")]:
        say(f"  [{'PASA' if ok else 'FALLA'}] {txt}")
    say("")
    if all([c1, c2, c3, c4, c5]):
        say("  HIPOTESIS RESPALDADA. GEX de indice funciona como switch de regimen.")
        say("  Siguiente: validar los parametros en train/test partido, NO aqui.")
    else:
        say("  HIPOTESIS NO RESPALDADA. El switch por GEX no dispara el retorno de forma")
        say("  robusta. Se reporta y se cierra, segun el pre-registro.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
