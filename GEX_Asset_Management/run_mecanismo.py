#!/usr/bin/env python3
"""PUERTA P3 -- TEST DE MECANISMO. El objetivo real del piloto.

LA HIPOTESIS DEL PAPER, ESLABON POR ESLABON
  Gamma negativa -> los market makers deben COMPRAR cuando el precio sube y VENDER
  cuando baja -> amplifican el movimiento -> mayor volatilidad realizada futura ->
  los inversionistas exigen prima de riesgo -> mayor retorno esperado.
Soebhag (2023, seccion 5) verifica el eslabon de volatilidad y encuentra que Gamma
predice NEGATIVAMENTE la volatilidad realizada del mes siguiente.

POR QUE ESTE TEST Y NO EL DE RETORNOS
Con 52 semanas y 30 nombres, el spread de retornos tiene ~52 observaciones de serie
temporal: no alcanza significancia para una prima de riesgo pequena, pase lo que
pase. El eslabon de volatilidad, en cambio, es un PANEL de 30 x 52 = ~1.560
observaciones con un efecto grande. Es lo unico concluyente que este piloto puede
producir, y es un resultado presentable: "el canal de hedging de gamma opera en
estos nombres en 2025-26, con el signo y la magnitud del paper".

ESPECIFICACION
  RV_{t+1} = a_i + b*Gamma_t + controles_t + e
  efectos fijos por accion (absorben el nivel de volatilidad de cada nombre)
  errores estandar clusterizados por FECHA (las acciones co-mueven dentro de una
  semana; ignorarlo infla los t-stats de forma severa)
  Prediccion del paper: b < 0

Se reporta ademas el test de retornos como INDICATIVO, nunca como conclusion.

USO
    python3 run_mecanismo.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.equities import load_equities

REP = ROOT / "reports" / "p3_mecanismo.txt"
INDICES = ["SPY", "QQQ"]
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def main():
    import statsmodels.api as sm

    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)

    say("=" * 92)
    say("PUERTA P3 -- TEST DE MECANISMO: Gamma predice la volatilidad futura?".center(92))
    say("=" * 92)

    # ---- retornos diarios y volatilidad realizada por ventana
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .sort(["symbol", "date"])
           .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol"))
                         .log().alias("ret")))
    say(f"\nequities: {d.height:,} filas · {d['symbol'].n_unique()} simbolos")

    fechas = sorted(g["date"].unique().to_list())
    say(f"fechas de senal: {len(fechas)} (semanales)")

    # RV en la ventana [t, t+1) y en la ventana previa [t-1, t)
    filas = []
    for i, t in enumerate(fechas):
        nxt = fechas[i + 1] if i + 1 < len(fechas) else None
        prv = fechas[i - 1] if i > 0 else None
        if nxt is None or prv is None:
            continue
        fwd = (d.filter((pl.col("date") > t) & (pl.col("date") <= nxt))
                 .group_by("symbol")
                 .agg([(pl.col("ret").std() * np.sqrt(252)).alias("rv_fwd"),
                       pl.len().alias("n_fwd"),
                       (pl.col("close").last() / pl.col("close").first() - 1).alias("ret_fwd")]))
        pas = (d.filter((pl.col("date") > prv) & (pl.col("date") <= t))
                 .group_by("symbol")
                 .agg((pl.col("ret").std() * np.sqrt(252)).alias("rv_lag")))
        sub = (g.filter(pl.col("date") == t)
                 .join(fwd, left_on="underlying", right_on="symbol", how="inner")
                 .join(pas, left_on="underlying", right_on="symbol", how="inner"))
        filas.append(sub)
    panel = pl.concat(filas, how="vertical_relaxed").filter(
        (pl.col("n_fwd") >= 3) & pl.col("rv_fwd").is_finite()
        & pl.col("rv_lag").is_finite())

    xs = panel.filter(~pl.col("underlying").is_in(INDICES))
    say(f"panel: {xs.height:,} observaciones accion-semana · "
        f"{xs['underlying'].n_unique()} acciones · {xs['date'].n_unique()} semanas")

    # z-score cross-seccional de Gamma por fecha (comparable entre semanas)
    xs = xs.with_columns([
        ((pl.col("gamma_exposure") - pl.col("gamma_exposure").mean().over("date"))
         / pl.col("gamma_exposure").std().over("date")).alias("gex_z"),
        pl.col("adv_usd").log().alias("log_adv"),
        pl.col("spot").log().alias("log_spot"),
    ]).drop_nulls(["gex_z", "rv_fwd", "rv_lag", "iv_median"])

    df = xs.to_pandas()
    say(f"tras limpiar: {len(df):,} observaciones\n")

    def panel_ols(y, xcols, etiqueta, fe_accion=True):
        X = df[xcols].copy()
        if fe_accion:
            D = pd_dummies(df["underlying"])
            X = np.hstack([X.values, D])
            nombres = xcols + [f"FE_{c}" for c in range(D.shape[1])]
        else:
            X = X.values
            nombres = list(xcols)
        X = sm.add_constant(X, has_constant="add")
        m = sm.OLS(df[y].values, X)
        # errores clusterizados por fecha
        res = m.fit(cov_type="cluster",
                    cov_kwds={"groups": df["date"].astype(str).values})
        i = 1  # const es 0, gex_z es 1
        b, se = res.params[i], res.bse[i]
        t = b / se
        say(f"  {etiqueta:<52}{b:>10.4f}{se:>9.4f}{t:>9.2f}{res.rsquared:>9.3f}")
        return b, t

    def pd_dummies(s):
        cats = sorted(s.unique())[1:]
        return np.column_stack([(s == c).astype(float).values for c in cats])

    say("-" * 92)
    say("A. PANEL PRINCIPAL -- Gamma predice la volatilidad realizada de la semana siguiente")
    say("-" * 92)
    say("   Prediccion del paper: coeficiente NEGATIVO.")
    say("   Errores clusterizados por fecha. Efectos fijos por accion.\n")
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>9}{'t':>9}{'R2':>9}")
    say(f"  {'-'*52}{'-'*10}{'-'*9}{'-'*9}{'-'*9}")
    b1, t1 = panel_ols("rv_fwd", ["gex_z"], "1. Gamma sola (con FE de accion)")
    b2, t2 = panel_ols("rv_fwd", ["gex_z", "rv_lag"], "2. + volatilidad realizada previa")
    b3, t3 = panel_ols("rv_fwd", ["gex_z", "rv_lag", "iv_median"], "3. + IV mediana")
    b4, t4 = panel_ols("rv_fwd", ["gex_z", "rv_lag", "iv_median", "log_adv", "log_spot"],
                       "4. + log(ADV$) y log(precio)")
    b5, t5 = panel_ols("rv_fwd", ["gex_z", "rv_lag", "iv_median", "log_adv", "log_spot",
                                  "net_gross_ratio"], "5. + razon de cancelacion")

    say("")
    say("-" * 92)
    say("B. TEST DE RETORNOS -- INDICATIVO, NO CONCLUYENTE")
    say("-" * 92)
    say("   Con ~52 semanas no hay potencia para una prima de riesgo pequena.")
    say("   Se reporta el signo, no se interpreta la significancia.\n")
    say(f"  {'especificacion':<52}{'coef':>10}{'se':>9}{'t':>9}{'R2':>9}")
    say(f"  {'-'*52}{'-'*10}{'-'*9}{'-'*9}{'-'*9}")
    br, tr = panel_ols("ret_fwd", ["gex_z"], "retorno semana siguiente ~ Gamma")

    # quintiles
    say("")
    q = (xs.with_columns(
            ((pl.col("gex_z").rank("ordinal").over("date") - 1)
             / pl.col("gex_z").count().over("date") * 5).floor().alias("q"))
           .group_by("q").agg([pl.col("ret_fwd").mean().alias("ret"),
                               pl.col("rv_fwd").mean().alias("rv"),
                               pl.len().alias("n")]).sort("q"))
    say("  Quintiles por Gamma (Q0 = Gamma mas baja = pata LARGA del paper):")
    say(f"    {'quintil':<10}{'ret sem. sig.':>16}{'vol realizada':>16}{'n':>8}")
    for r in q.iter_rows(named=True):
        say(f"    Q{int(r['q']):<9}{r['ret']*100:>15.3f}%{r['rv']*100:>15.1f}%{r['n']:>8}")
    lo = q.filter(pl.col("q") == 0); hi = q.filter(pl.col("q") == 4)
    if lo.height and hi.height:
        say(f"    {'L - H':<10}{(lo['ret'][0]-hi['ret'][0])*100:>15.3f}%"
            f"{(lo['rv'][0]-hi['rv'][0])*100:>15.1f}%")
        say("    (el paper predice: retorno L-H POSITIVO, volatilidad L-H POSITIVA)")

    say("")
    say("=" * 92)
    say("VEREDICTO DEL MECANISMO")
    say("=" * 92)
    signo_ok = b4 < 0
    sig = abs(t4) > 1.96
    if signo_ok and sig:
        say(f"  CONFIRMADO. Coeficiente {b4:+.4f} con t = {t4:.2f} en la especificacion")
        say("  completa. Gamma predice NEGATIVAMENTE la volatilidad realizada futura,")
        say("  con el signo del paper y significancia al 5%.")
        say("  El canal de hedging de gamma opera en estos 30 nombres en 2025-26.")
    elif signo_ok:
        say(f"  SIGNO CORRECTO, NO SIGNIFICATIVO. Coeficiente {b4:+.4f}, t = {t4:.2f}.")
        say("  Consistente con el paper pero sin poder rechazar la hipotesis nula.")
    else:
        say(f"  NO CONFIRMADO. Coeficiente {b4:+.4f} (positivo), t = {t4:.2f}.")
        say("  En esta muestra Gamma NO predice negativamente la volatilidad futura.")
        say("  No refuta el paper -- es otro regimen y otro universo -- pero el")
        say("  mecanismo no aparece aqui, y sin mecanismo no hay razon para esperar")
        say("  la prima de riesgo.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 92)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
