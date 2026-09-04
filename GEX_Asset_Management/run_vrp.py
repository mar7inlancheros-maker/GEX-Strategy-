#!/usr/bin/env python3
"""TEST DE LA PRIMA DE VARIANZA -- predice Gamma el spread RV - IV?

POR QUE ESTE ES EL TEST CORRECTO PARA LA VERSION CON OPCIONES
El piloto testeo si Gamma predice la volatilidad REALIZADA futura, pero opero
spot -- direccion, no volatilidad. Hay un desajuste entre hipotesis e
instrumento. Si la tesis del canal de cobertura es cierta, el instrumento
natural son las opciones.

Pero hay una trampa. Como gamma ~ 1/sigma por construccion, Gamma baja == IV
alta. Comprar volatilidad donde Gamma es baja es comprar las opciones MAS caras
y vender las mas baratas: ponerse corto de la prima de riesgo de varianza de
forma sistematica, y esa prima es justamente mayor en los nombres de IV alta.

La pregunta que decide no es "predice Gamma la volatilidad realizada" (esta
confundida por el vinculo mecanico) sino:

    predice Gamma el SPREAD entre realizada e implicita?

Ese spread es lo que cobra un straddle delta-cubierto. Restar la IV elimina el
artefacto 1/sigma de la variable dependiente, asi que es una ortogonalizacion
mas limpia que residualizar el regresor. Si Gamma no predice RV - IV, el
backtest de opciones pierde dinero y no hay que construirlo.

IV emparejada al horizonte: solo contratos cercanos al dinero con vencimiento
entre 7 y 45 dias, que es la ventana comparable a una volatilidad realizada
semanal. Usar la IV mediana de toda la cadena mezclaria plazos.

USO
    python3 run_vrp.py
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
REP = ROOT / "reports" / "vrp.txt"
ATM = 0.10          # |ln(S/K)| < 0.10
T_LO, T_HI = 7/365, 45/365
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
    r = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.bse[0]), float(r.tvalues[0]), len(x)


def fama_macbeth(df, y, xs, lags=4):
    import statsmodels.api as sm
    sl = []
    for _, g in df.group_by("date"):
        sub = g.select([y] + xs).drop_nulls()
        if sub.height < len(xs) + 6:
            continue
        X = sm.add_constant(sub.select(xs).to_numpy())
        try:
            sl.append(sm.OLS(sub[y].to_numpy(), X).fit().params[1])
        except Exception:
            continue
    return newey_west_t(sl, lags)


def linea(lab, res, w=48):
    c, se, t, n = res
    say(f"  {lab:<{w}}{c:>10.4f}{se:>10.4f}{t:>9.2f}{n:>7}")


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    eq = load_equities(ROOT)
    xs = g.filter(~pl.col("underlying").is_in(INDICES))

    say("=" * 96)
    say("PRIMA DE VARIANZA -- predice Gamma el spread entre realizada e implicita?".center(96))
    say("=" * 96)

    # ---- IV emparejada al horizonte (ATM, 7-45 dias)
    say("\nconstruyendo IV emparejada al horizonte (ATM, 7-45 dias)...")
    gr = (pl.scan_parquet(ROOT / "data/curated/contract_greeks.parquet")
            .filter((pl.col("T") >= T_LO) & (pl.col("T") <= T_HI)
                    & (pl.col("ln_moneyness").abs() < ATM)
                    & pl.col("iv").is_not_null())
            .group_by(["date", "underlying"])
            .agg([pl.col("iv").median().alias("iv_atm"), pl.len().alias("n_iv")])
            .collect())
    gr = gr.filter(pl.col("n_iv") >= 4)
    say(f"  {gr.height:,} pares fecha-accion con IV ATM de plazo corto")

    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .sort(["symbol", "date"])
           .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol"))
                         .log().alias("ret")))
    fechas = sorted(xs["date"].unique().to_list())

    filas = []
    for i, t in enumerate(fechas):
        if i + 1 >= len(fechas) or i == 0:
            continue
        nxt, prv = fechas[i + 1], fechas[i - 1]
        fwd = (d.filter((pl.col("date") > t) & (pl.col("date") <= nxt))
                 .group_by("symbol").agg([
                     (pl.col("ret").std() * np.sqrt(252)).alias("rv_fwd"),
                     pl.len().alias("nd")]))
        fwd = fwd.filter(pl.col("nd") >= 4)
        pas = (d.filter((pl.col("date") > prv) & (pl.col("date") <= t))
                 .group_by("symbol").agg((pl.col("ret").std() * np.sqrt(252)).alias("rv_prev")))
        cur = xs.filter(pl.col("date") == t)
        filas.append(cur.join(fwd, left_on="underlying", right_on="symbol", how="inner")
                        .join(pas, left_on="underlying", right_on="symbol", how="left"))
    pan = (pl.concat(filas).join(gr, on=["date", "underlying"], how="inner")
             .with_columns((pl.col("rv_fwd") - pl.col("iv_atm")).alias("vrp"))
             .drop_nulls(["vrp", "gamma_exposure"]))
    say(f"panel: {pan.height:,} obs · {pan['date'].n_unique()} semanas · "
        f"{pan['underlying'].n_unique()} acciones")

    # ---- linea base: existe la prima?
    v = pan["vrp"]
    say("")
    say("-" * 96)
    say("A. LINEA BASE -- cuanto vale la prima de varianza en esta muestra")
    say("-" * 96)
    say(f"  RV futura - IV ATM:  media {float(v.mean()):+.4f}   mediana {float(v.median()):+.4f}")
    say(f"  % de observaciones con RV > IV: {float((v > 0).mean())*100:.1f}%")
    say(f"  IV ATM media {float(pan['iv_atm'].mean()):.3f}  ·  "
        f"RV futura media {float(pan['rv_fwd'].mean()):.3f}")
    neg = float(v.mean()) < 0
    say(f"  -> la implicita {'SUPERA' if neg else 'NO supera'} a la realizada en promedio: "
        f"{'prima positiva para el VENDEDOR de opciones' if neg else 'anomalo'}")

    # ---- el test
    say("")
    say("-" * 96)
    say("B. EL TEST -- predice Gamma la prima? (Fama-MacBeth, Newey-West 4 rezagos)")
    say("-" * 96)
    say("   Si el canal de cobertura opera y el mercado no lo tiene precificado,")
    say("   Gamma baja deberia anticipar RV > IV. Coeficiente NEGATIVO.")
    say("")
    say(f"  {'especificacion':<48}{'coef':>10}{'se':>10}{'t (NW)':>9}{'sem':>7}")
    say(f"  {'-'*48}{'-'*10}{'-'*10}{'-'*9}{'-'*7}")
    r1 = fama_macbeth(pan, "vrp", ["gamma_exposure"])
    linea("1. prima ~ Gamma", r1)
    linea("2. + vol realizada previa", fama_macbeth(pan, "vrp", ["gamma_exposure", "rv_prev"]))
    linea("3. + IV ATM", fama_macbeth(pan, "vrp", ["gamma_exposure", "iv_atm"]))
    linea("4. + ambas", fama_macbeth(pan, "vrp", ["gamma_exposure", "rv_prev", "iv_atm"]))
    say("")
    linea("5. solo componente slow (>31d)", fama_macbeth(pan, "vrp", ["gex_slow"]))
    linea("6. control: la IV sola predice la prima",
          fama_macbeth(pan, "vrp", ["iv_atm"]))

    # ---- lo que se operaria
    say("")
    say("-" * 96)
    say("C. LO QUE SE OPERARIA -- prima media por quintil de Gamma")
    say("-" * 96)
    say("   Q0 = Gamma mas baja = donde el paper dice comprar volatilidad.")
    say("   Un straddle delta-cubierto cobra aproximadamente esta prima.")
    say("")
    q = (pan.with_columns(
            pl.col("gamma_exposure").qcut(5, labels=[f"Q{i}" for i in range(5)])
              .over("date").alias("q"))
           .group_by("q").agg([
               pl.col("vrp").mean().alias("vrp"), pl.col("iv_atm").mean().alias("iv"),
               pl.col("rv_fwd").mean().alias("rv"), pl.len().alias("n")]).sort("q"))
    say(f"  {'quintil':<10}{'IV ATM':>10}{'RV futura':>12}{'prima RV-IV':>14}{'n':>8}")
    say(f"  {'-'*10}{'-'*10}{'-'*12}{'-'*14}{'-'*8}")
    for r in q.iter_rows(named=True):
        say(f"  {str(r['q']):<10}{r['iv']:>10.3f}{r['rv']:>12.3f}{r['vrp']:>+14.4f}{r['n']:>8,}")
    v0 = float(q.filter(pl.col("q") == "Q0")["vrp"][0])
    v4 = float(q.filter(pl.col("q") == "Q4")["vrp"][0])
    say(f"\n  spread Q0 - Q4: {v0-v4:+.4f} puntos de volatilidad anualizada")
    say("  (positivo = comprar vol en Gamma baja y venderla en Gamma alta gana)")

    # significancia del spread, por fecha
    sp = (pan.with_columns(
            pl.col("gamma_exposure").qcut(5, labels=[f"Q{i}" for i in range(5)])
              .over("date").alias("q"))
            .filter(pl.col("q").is_in(["Q0", "Q4"]))
            .group_by(["date", "q"]).agg(pl.col("vrp").mean())
            .pivot(on="q", index="date", values="vrp").drop_nulls())
    dif = (sp["Q0"] - sp["Q4"]).to_numpy()
    c, se, t, n = newey_west_t(dif)
    say(f"  serie temporal del spread: media {c:+.4f}  se {se:.4f}  "
        f"t = {t:+.2f}  ({n} semanas)")

    # ---- veredicto
    say("")
    say("=" * 96)
    say("VEREDICTO")
    say("=" * 96)
    say(f"  Gamma -> prima:   coef {r1[0]:+.4f}   t = {r1[2]:+.2f}")
    say(f"  spread Q0-Q4:     {c:+.4f}            t = {t:+.2f}")
    say("")
    if abs(r1[2]) >= 2 and r1[0] < 0 and t > 0:
        say("  HAY SENAL. Gamma predice la prima con el signo correcto: comprar")
        say("  volatilidad donde Gamma es baja cobraria mas de lo que paga. El")
        say("  backtest con opciones vale la pena construirse.")
    elif abs(r1[2]) >= 2:
        say("  HAY SENAL PERO CON EL SIGNO CONTRARIO. Operar la estrategia como la")
        say("  describe el paper PERDERIA la prima de forma sistematica.")
    else:
        say("  NO HAY SENAL. Gamma no predice el spread entre realizada e implicita.")
        say("  El mercado ya tiene precificado en la IV lo que Gamma pueda decir")
        say("  sobre la volatilidad futura. Un straddle guiado por esta senal")
        say("  cobraria la prima media del quintil y nada mas -- y como Gamma baja")
        say("  == IV alta, seria comprar las opciones mas caras del universo.")
        say("  NO construir el backtest de opciones.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
