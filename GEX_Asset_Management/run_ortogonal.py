#!/usr/bin/env python3
"""TEST DE ORTOGONALIZACION -- que queda de Gamma cuando le quitas la volatilidad?

P3 y P3b dejaron el efecto crudo significativo (t = -2.51 en Fama-MacBeth, y el
componente de cobertura en t = -4.05, replicando la estructura del paper) pero
muerto en cuanto entra la IV como control (t = +0.08).

La sospecha es un confundido MECANICO, no economico: la gamma de Black-Scholes
lleva 1/(S*sigma*sqrt(T)) dentro, asi que Gamma es una funcion decreciente de
sigma POR CONSTRUCCION. Como sigma es persistente, Gamma "predice" la
volatilidad futura sin que medie ningun canal de cobertura.

Este script separa las dos cosas. En cada fecha regresa Gamma contra la IV
cross-seccionalmente y se queda con el RESIDUO: la parte de Gamma que no se
explica por el nivel de volatilidad. Si el residuo sigue prediciendo, hay senal
mas alla del artefacto. Si no, el efecto del paper es la mecanica de la formula.

Es una hipotesis pre-especificada, no una busqueda de parametros: se decide
ANTES de mirar, y el resultado se reporta salga como salga.

USO
    python3 run_ortogonal.py
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
REP = ROOT / "reports" / "ortogonal.txt"
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def newey_west_t(x, lags=4):
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), len(x)


def fama_macbeth(df, y, xs, lags=4):
    import statsmodels.api as sm
    slopes = []
    for _, g in df.group_by("date"):
        sub = g.select([y] + xs).drop_nulls()
        if sub.height < len(xs) + 6:
            continue
        X = sm.add_constant(sub.select(xs).to_numpy())
        try:
            slopes.append(sm.OLS(sub[y].to_numpy(), X).fit().params[1])
        except Exception:
            continue
    return newey_west_t(slopes, lags), slopes


def residualizar(df: pl.DataFrame, y: str, contra: list, nombre: str) -> pl.DataFrame:
    """Residuo cross-seccional de `y` contra `contra`, fecha por fecha."""
    import statsmodels.api as sm
    partes = []
    for _, g in df.group_by("date", maintain_order=True):
        sub = g.select(["date", "underlying", y] + contra)
        ok = sub.drop_nulls()
        if ok.height < len(contra) + 6:
            continue
        X = sm.add_constant(ok.select(contra).to_numpy())
        r = sm.OLS(ok[y].to_numpy(), X).fit()
        partes.append(ok.select(["date", "underlying"])
                        .with_columns(pl.Series(nombre, r.resid)))
    return pl.concat(partes) if partes else pl.DataFrame()


def linea(lab, res, ancho=52):
    coef, se, t, n = res
    say(f"  {lab:<{ancho}}{coef:>10.4f}{se:>10.4f}{t:>9.2f}{n:>7}")


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)
    xs = g.filter(~pl.col("underlying").is_in(INDICES))

    say("=" * 96)
    say("TEST DE ORTOGONALIZACION -- Gamma sin su componente de volatilidad".center(96))
    say("=" * 96)

    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .sort(["symbol", "date"])
           .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol"))
                         .log().alias("ret")))
    fechas = sorted(xs["date"].unique().to_list())

    # ---------------------------------------------- panel con futuros y pasado
    filas = []
    for i, t in enumerate(fechas):
        if i + 1 >= len(fechas) or i == 0:
            continue
        nxt, prv = fechas[i + 1], fechas[i - 1]
        fwd = (d.filter((pl.col("date") > t) & (pl.col("date") <= nxt))
                 .group_by("symbol").agg([
                     (pl.col("ret").std() * np.sqrt(252)).alias("rv_fwd"),
                     (pl.col("close").last() / pl.col("close").first() - 1).alias("ret_fwd")]))
        pas = (d.filter((pl.col("date") > prv) & (pl.col("date") <= t))
                 .group_by("symbol").agg(
                     (pl.col("ret").std() * np.sqrt(252)).alias("rv_prev")))
        cur = xs.filter(pl.col("date") == t)
        filas.append(cur.join(fwd, left_on="underlying", right_on="symbol", how="inner")
                        .join(pas, left_on="underlying", right_on="symbol", how="left"))
    pan = pl.concat(filas).with_columns([
        pl.col("adv_usd").log().alias("l_adv"),
        pl.col("spot").log().alias("l_px"),
        pl.col("iv_median").alias("iv"),
    ]).drop_nulls(["rv_fwd", "iv", "gamma_exposure"])
    say(f"\npanel: {pan.height:,} obs · {pan['date'].n_unique()} semanas · "
        f"{pan['underlying'].n_unique()} acciones")

    # ------------------------------------------ cuanto de Gamma explica la IV
    say("")
    say("-" * 96)
    say("A. CUANTO DE GAMMA ES VOLATILIDAD  (R2 de la regresion Gamma ~ IV, por fecha)")
    say("-" * 96)
    import statsmodels.api as sm
    r2s, rhos = [], []
    for _, gg in pan.group_by("date"):
        sub = gg.select(["gamma_exposure", "iv"]).drop_nulls()
        if sub.height < 8:
            continue
        X = sm.add_constant(sub["iv"].to_numpy())
        r2s.append(sm.OLS(sub["gamma_exposure"].to_numpy(), X).fit().rsquared)
        rhos.append(np.corrcoef(sub["gamma_exposure"], sub["iv"])[0, 1])
    r2s, rhos = np.array(r2s), np.array(rhos)
    say(f"  R2 de Gamma ~ IV:  mediana {np.median(r2s):.3f}   "
        f"p25 {np.percentile(r2s,25):.3f}   p75 {np.percentile(r2s,75):.3f}")
    say(f"  correlacion Gamma-IV: mediana {np.median(rhos):+.3f}   "
        f"% de fechas con signo negativo: {(rhos<0).mean()*100:.0f}%")
    say("  (el paper reporta -0.11; una correlacion mucho mas fuerte que esa es")
    say("   senal de que en mega-caps el vinculo mecanico domina)")

    # ------------------------------------------------------ ortogonalizacion
    orto = residualizar(pan, "gamma_exposure", ["iv"], "gex_orth")
    orto2 = residualizar(pan, "gamma_exposure", ["iv", "rv_prev"], "gex_orth2")
    pan = pan.join(orto, on=["date", "underlying"], how="left")
    pan = pan.join(orto2, on=["date", "underlying"], how="left")

    say("")
    say("-" * 96)
    say("B. EL TEST -- predice el RESIDUO la volatilidad de la semana siguiente?")
    say("-" * 96)
    say("   Prediccion del paper: coeficiente NEGATIVO y significativo.")
    say("   Fama-MacBeth, t de Newey-West (4 lags).")
    say("")
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>10}{'t (NW)':>9}{'sem':>7}")
    say(f"  {'-'*52}{'-'*10}{'-'*10}{'-'*9}{'-'*7}")
    linea("1. Gamma cruda (referencia, = P3b col 1)",
          fama_macbeth(pan, "rv_fwd", ["gamma_exposure"])[0])
    linea("2. Gamma ORTOGONAL a IV  <-- el test",
          fama_macbeth(pan, "rv_fwd", ["gex_orth"])[0])
    linea("3. Gamma ORTOGONAL a IV y vol previa",
          fama_macbeth(pan, "rv_fwd", ["gex_orth2"])[0])
    say("")
    linea("4. control: la IV sola predice la vol futura",
          fama_macbeth(pan, "rv_fwd", ["iv"])[0])
    linea("5. control: la vol previa sola",
          fama_macbeth(pan, "rv_fwd", ["rv_prev"])[0])

    # -------------------------------------------- la razon de cancelacion
    say("")
    say("-" * 96)
    say("C. LA RAZON DE CANCELACION -- la unica variable que aparecio con senal propia")
    say("-" * 96)
    say("   Salio con t = 2.26 en P3 (spec 5). El paper no la usa. Se testea")
    say("   honestamente: si no sobrevive sus propios controles, era ruido.")
    say("")
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>10}{'t (NW)':>9}{'sem':>7}")
    say(f"  {'-'*52}{'-'*10}{'-'*10}{'-'*9}{'-'*7}")
    linea("6. razon net/gross sola", fama_macbeth(pan, "rv_fwd", ["net_gross_ratio"])[0])
    linea("7. + IV", fama_macbeth(pan, "rv_fwd", ["net_gross_ratio", "iv"])[0])
    linea("8. + IV + vol previa",
          fama_macbeth(pan, "rv_fwd", ["net_gross_ratio", "iv", "rv_prev"])[0])

    # ----------------------------------------- la condicion del propio paper
    say("")
    say("-" * 96)
    say("D. LA CONDICION DEL PAPER -- la senal debe venir de 'slow' (>31d)")
    say("-" * 96)
    say("   En P2 salio fast ~ slow, que es una desviacion. Se testea por separado.")
    say("")
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>10}{'t (NW)':>9}{'sem':>7}")
    say(f"  {'-'*52}{'-'*10}{'-'*10}{'-'*9}{'-'*7}")
    for col, lab in [("gex_slow", "9.  Gamma slow (>31d) -- donde el paper dice"),
                     ("gex_fast", "10. Gamma fast (<=31d)"),
                     ("gex_atm", "11. Gamma ATM"),
                     ("gex_otm", "12. Gamma OTM")]:
        linea(lab, fama_macbeth(pan, "rv_fwd", [col])[0])
    say("")
    orto_slow = residualizar(pan.drop_nulls(["gex_slow"]), "gex_slow", ["iv"], "slow_orth")
    p2 = pan.join(orto_slow, on=["date", "underlying"], how="inner")
    linea("13. Gamma slow ORTOGONAL a IV", fama_macbeth(p2, "rv_fwd", ["slow_orth"])[0])

    # ------------------------------------------------------------- retornos
    say("")
    say("-" * 96)
    say("E. RETORNOS (indicativo -- 60 meses no dan potencia para una prima chica)")
    say("-" * 96)
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>10}{'t (NW)':>9}{'sem':>7}")
    say(f"  {'-'*52}{'-'*10}{'-'*10}{'-'*9}{'-'*7}")
    linea("14. retorno ~ Gamma cruda", fama_macbeth(pan, "ret_fwd", ["gamma_exposure"])[0])
    linea("15. retorno ~ Gamma ORTOGONAL", fama_macbeth(pan, "ret_fwd", ["gex_orth"])[0])

    # ------------------------------------------------------------- veredicto
    (c1, _, t1, _) = fama_macbeth(pan, "rv_fwd", ["gamma_exposure"])[0]
    (c2, _, t2, _) = fama_macbeth(pan, "rv_fwd", ["gex_orth"])[0]
    say("")
    say("=" * 96)
    say("VEREDICTO")
    say("=" * 96)
    say(f"  Gamma cruda:      coef {c1:+.4f}   t = {t1:+.2f}")
    say(f"  Gamma ortogonal:  coef {c2:+.4f}   t = {t2:+.2f}")
    say("")
    if abs(t2) >= 2.0 and c2 < 0:
        say("  QUEDA SENAL. El residuo -- la parte de Gamma que NO es volatilidad --")
        say("  sigue prediciendo con el signo correcto. El efecto no se reduce al")
        say("  artefacto mecanico y vale la pena seguir investigando.")
    elif abs(t2) >= 2.0:
        say("  QUEDA SENAL, PERO CON EL SIGNO CONTRARIO al que predice el paper.")
        say("  Eso no respalda la estrategia: pide explicacion antes de usarse.")
    else:
        say("  NO QUEDA SENAL. Al quitarle la volatilidad, Gamma deja de predecir.")
        say("  El efecto documentado es, en esta muestra, la mecanica de la formula:")
        say("  gamma ~ 1/sigma por construccion, y sigma es persistente. No hay")
        say("  evidencia de un canal de cobertura separable en mega-caps liquidas.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
