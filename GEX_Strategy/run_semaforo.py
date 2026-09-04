#!/usr/bin/env python3
"""GEX de indice como semaforo de exposicion -- test pre-registrado.

Pre-registro congelado el 2026-09-03 (scratchpad/preregistro_semaforo.md).
Umbral z +-0.5, ventana 26 semanas, pesos 100/50/0, horizonte 1 semana,
5 bps por lado sobre el CAMBIO de exposicion. Nada de eso se toca.

LA IDEA
GEX es el indicador (sale de datos de opciones). Las ACCIONES son el vehiculo.
No se operan opciones. Una sola decision por semana: cuanta exposicion larga
tener al mercado.

Dealers largos gamma -> cubren contra tendencia -> mercado estable -> invertido.
Dealers cortos gamma -> cubren a favor -> caidas amplificadas -> fuera.

El comparador que importa es E: una exposicion FIJA con el mismo promedio que A.
Si A no le gana a E, la mejora venia de tener menos beta, no del semaforo.
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
REP = ROOT / "reports" / "semaforo.txt"
Z_TH, WIN, COST, SEM = 0.5, 26, 0.0005, 52
PESOS = {"verde": 1.0, "amarillo": 0.5, "rojo": 0.0}
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def nw_t(x, lags=4):
    import statsmodels.api as sm
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues[0])


def aplicar(pesos, act, rf):
    """Retorno neto de una serie de pesos sobre el activo, con cash al rf."""
    w = np.asarray(pesos, float)
    r = w * act + (1 - w) * rf
    cambio = np.abs(np.diff(np.concatenate([[0.0], w])))
    return r - cambio * COST


def metricas(r, rf):
    r = np.asarray(r, float); n = len(r)
    ex = r - rf[:n]
    ret = np.prod(1 + r) ** (SEM / n) - 1
    vol = r.std(ddof=1) * np.sqrt(SEM)
    c = np.cumprod(1 + r)
    dd = float((c / np.maximum.accumulate(c) - 1).min())
    return dict(ret=ret, vol=vol, sharpe=ex.mean() * SEM / vol if vol > 0 else np.nan,
                dd=dd, t=nw_t(r))


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)
    curva = fetch_treasury_curve(ROOT)
    fechas = sorted(g["date"].unique().to_list())

    say("=" * 96)
    say("GEX COMO SEMAFORO DE EXPOSICION -- test pre-registrado".center(96))
    say(f"indicador: GEX(SPY)+GEX(QQQ) · z +-{Z_TH} sobre {WIN} sem · pesos 100/50/0".center(96))
    say("=" * 96)

    # ---- indicador y semaforo (sin look-ahead)
    idx = (g.filter(pl.col("underlying").is_in(INDICES))
             .group_by("date").agg(pl.col("gamma_exposure").sum().alias("G")).sort("date"))
    G, fx = idx["G"].to_numpy(), idx["date"].to_list()
    z = np.full(len(G), np.nan)
    for i in range(len(G)):
        lo = max(0, i - WIN)
        if i - lo >= 8:
            w = G[lo:i]
            z[i] = (G[i] - w.mean()) / (w.std() + 1e-12)
    zmap = dict(zip(fx, z))

    def estado(f):
        zz = zmap.get(f, np.nan)
        if not np.isfinite(zz):
            return "amarillo"
        return "verde" if zz >= Z_TH else ("rojo" if zz <= -Z_TH else "amarillo")

    # ---- retornos semanales del activo
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date")).sort(["symbol", "date"]))
    close = {(r["symbol"], r["date"]): r["close"] for r in d.iter_rows(named=True)}
    nombres = [s for s in g["underlying"].unique().to_list() if s not in INDICES]

    fs, act_spy, act_eq = [], [], []
    for i in range(len(fechas) - 1):
        a, b = fechas[i], fechas[i + 1]
        ca, cb = close.get(("SPY", a)), close.get(("SPY", b))
        rr = [close[(s, b)] / close[(s, a)] - 1 for s in nombres
              if close.get((s, a)) and close.get((s, b))]
        if ca and cb and rr:
            fs.append(a); act_spy.append(cb / ca - 1); act_eq.append(float(np.mean(rr)))
    act_spy, act_eq = np.array(act_spy), np.array(act_eq)
    rf = np.expm1(rate_lookup(curva, fs, np.full(len(fs), 0.25)) / SEM)

    est = [estado(f) for f in fs]
    from collections import Counter
    cc = Counter(est)
    say(f"\nsemanas operables: {len(fs)}")
    say(f"semaforo: {cc['verde']} verde · {cc['amarillo']} amarillo · {cc['rojo']} rojo")
    w_A = np.array([PESOS[e] for e in est])
    say(f"exposicion media de A: {w_A.mean()*100:.1f}%")

    resumen = {}
    for vname, act in [("V1 · SPY", act_spy), ("V2 · 30 nombres equiponderados", act_eq)]:
        say("")
        say("-" * 96)
        say(f"{vname}")
        say("-" * 96)
        w_B = np.ones(len(fs))
        w_C = np.array([{"verde": 0.0, "amarillo": 0.5, "rojo": 1.0}[e] for e in est])
        w_E = np.full(len(fs), w_A.mean())
        series = {"A": aplicar(w_A, act, rf), "B": aplicar(w_B, act, rf),
                  "C": aplicar(w_C, act, rf), "E": aplicar(w_E, act, rf)}
        M = {k: metricas(v, rf) for k, v in series.items()}
        say(f"  {'estrategia':<42}{'ret anual':>10}{'vol':>8}{'Sharpe':>8}{'max DD':>9}{'t (NW)':>9}")
        say(f"  {'-'*42}{'-'*10}{'-'*8}{'-'*8}{'-'*9}{'-'*9}")
        for k, lab in [("A", "A. Semaforo GEX (100/50/0%)"),
                       ("B", "B. Buy & hold (100%)"),
                       ("E", f"E. Exposicion fija {w_A.mean()*100:.0f}% (control clave)"),
                       ("C", "C. Semaforo INVERTIDO (placebo)")]:
            m = M[k]
            say(f"  {lab:<42}{m['ret']*100:>9.2f}%{m['vol']*100:>7.1f}%"
                f"{m['sharpe']:>8.2f}{m['dd']*100:>8.1f}%{m['t']:>9.2f}")

        # placebo aleatorio
        rng = np.random.default_rng(20260903)
        sh = []
        for _ in range(1000):
            wp = np.array([PESOS[e] for e in rng.permutation(est)])
            sh.append(metricas(aplicar(wp, act, rf), rf)["sharpe"])
        sh = np.array(sh)
        pctl = (sh < M["A"]["sharpe"]).mean() * 100
        say(f"\n  placebo aleatorio (1000): media {sh.mean():.3f} · p95 {np.percentile(sh,95):.3f}"
            f" · A en percentil {pctl:.1f}")
        resumen[vname] = dict(M=M, pctl=pctl)

    # ---- veredicto
    say("")
    say("=" * 96)
    say("VEREDICTO CONTRA EL CRITERIO PRE-REGISTRADO")
    say("=" * 96)
    todos = []
    for vname, r in resumen.items():
        M, pctl = r["M"], r["pctl"]
        c = [M["A"]["sharpe"] - M["B"]["sharpe"] >= 0.15,
             M["A"]["sharpe"] - M["E"]["sharpe"] >= 0.10,
             abs(M["A"]["dd"]) < abs(M["B"]["dd"]),
             M["C"]["sharpe"] < M["A"]["sharpe"],
             pctl >= 95]
        todos.append(all(c))
        say(f"\n  {vname}")
        for ok, txt in zip(c, [
                f"1. Sharpe A - B >= 0.15   ({M['A']['sharpe']:.2f} - {M['B']['sharpe']:.2f} = {M['A']['sharpe']-M['B']['sharpe']:+.2f})",
                f"2. Sharpe A - E >= 0.10   ({M['A']['sharpe']:.2f} - {M['E']['sharpe']:.2f} = {M['A']['sharpe']-M['E']['sharpe']:+.2f})",
                f"3. drawdown de A menor que B   ({M['A']['dd']*100:.1f}% vs {M['B']['dd']*100:.1f}%)",
                f"4. placebo invertido C peor que A   ({M['C']['sharpe']:.2f} < {M['A']['sharpe']:.2f})",
                f"5. A sobre p95 del aleatorio   (percentil {pctl:.0f})"]):
            say(f"    [{'PASA' if ok else 'FALLA'}] {txt}")
    say(f"\n  6. [{'PASA' if all(todos) else 'FALLA'}] Se cumple en AMBAS versiones")
    say("")
    if all(todos):
        say("  HIPOTESIS RESPALDADA. El GEX funciona como semaforo de exposicion.")
        say("  Siguiente: validar los parametros en train/test partido, NO aqui.")
    else:
        say("  HIPOTESIS NO RESPALDADA segun el criterio congelado.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
