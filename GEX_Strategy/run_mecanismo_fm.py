#!/usr/bin/env python3
"""PUERTA P3b -- Test de mecanismo con la especificacion REAL del paper.

POR QUE EL PRIMER TEST NO ERA EL TEST DEL PAPER
Yo corri un PANEL CON EFECTOS FIJOS POR ACCION. La seccion 5 del paper no hace eso:
dice "the average Gamma coefficient equals -12.92 with a Newey-West t-statistic of
-3.58". "Coeficiente promedio" + "Newey-West" = FAMA-MACBETH: una regresion
CROSS-SECCIONAL por periodo, y despues se promedian las pendientes.

La diferencia no es cosmetica. Los efectos fijos por accion absorben precisamente
la variacion CROSS-SECCIONAL que el paper usa como su fuente de identificacion. Mi
panel preguntaba "cuando la Gamma de GME sube respecto a su propio promedio, baja
su volatilidad?"; el paper pregunta "las acciones con Gamma mas baja QUE OTRAS
tienen mas volatilidad despues?". Son dos preguntas distintas y yo probe la que el
paper no hace. Los quintiles ya mostraban el patron cross-seccional con fuerza:
40.9% de volatilidad en el quintil de Gamma mas baja contra 26.9% en el mas alta.

EL CONFUNDIDO QUE HAY QUE TOMAR EN SERIO
La gamma de Black-Scholes es proporcional a 1/(S*sigma*raiz(T)). Es decir: a MAYOR
volatilidad, MENOR gamma por contrato, y por tanto MENOR Gamma agregada. Existe una
relacion negativa MECANICA entre Gamma y volatilidad, metida en la propia formula,
que no tiene nada que ver con el canal de hedging.
El paper lo enfrenta controlando por volatilidad implicita (columna 2 de su Tabla
13) y reporta que la pendiente sobrevive. Aqui se hace lo mismo, y ademas se corre
su descomposicion de identificacion:

    Gamma(t) = Gamma_viejo(OI de t-1, precio de hoy)  +  Gamma_info(cambio en OI)

Las posiciones que YA EXISTIAN no pueden venir de informacion privada adquirida
despues. Si el efecto vive en el componente VIEJO, es re-balanceo de cobertura
(el mecanismo del paper). Si vive en el componente de INFORMACION, es que alguien
sabia algo. El paper encuentra el primero (t = -4.57) y no el segundo.

USO
    python3 run_mecanismo_fm.py
"""
from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gex.equities import load_equities

REP = ROOT / "reports" / "p3b_mecanismo_fm.txt"
INDICES = ["SPY", "QQQ"]
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(s)


def newey_west_t(x, lags=4):
    """t de Newey-West para la media de una serie (el t de Fama-MacBeth)."""
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), len(x)


def fama_macbeth(df, y, xs, lags=4):
    """Regresion cross-seccional por fecha; devuelve la media de la pendiente de xs[0]."""
    import statsmodels.api as sm
    slopes = []
    for _, g in df.group_by("date"):
        sub = g.select([y] + xs).drop_nulls()
        if sub.height < len(xs) + 6:
            continue
        X = sm.add_constant(sub.select(xs).to_numpy())
        try:
            r = sm.OLS(sub[y].to_numpy(), X).fit()
            slopes.append(r.params[1])
        except Exception:
            continue
    return newey_west_t(slopes, lags), slopes


def main():
    g = pl.read_parquet(ROOT / "data/curated/gamma_exposure.parquet")
    gr = pl.read_parquet(ROOT / "data/curated/contract_greeks.parquet")
    eq = load_equities(ROOT)

    say("=" * 96)
    say("PUERTA P3b -- MECANISMO CON LA ESPECIFICACION DEL PAPER (Fama-MacBeth)".center(96))
    say("=" * 96)

    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .sort(["symbol", "date"])
           .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol"))
                         .log().alias("ret")))
    fechas = sorted(g["date"].unique().to_list())

    # ---- Gamma "vieja": OI de la fecha anterior, precio y greeks de HOY.
    #      Se empareja por raw_symbol (el simbolo OSI), NUNCA por instrument_id,
    #      que Databento recicla entre dias.
    say("\nconstruyendo la descomposicion de identificacion del paper...")
    oi_prev = (gr.select(["date", "underlying", "raw_symbol", "open_interest"])
                 .rename({"open_interest": "oi_prev", "date": "date_prev"}))
    mapa = {fechas[i]: fechas[i - 1] for i in range(1, len(fechas))}
    gr2 = gr.with_columns(pl.col("date").replace_strict(mapa, default=None)
                          .alias("date_prev"))
    j = gr2.join(oi_prev, on=["date_prev", "underlying", "raw_symbol"], how="inner")
    say(f"  contratos con OI de la semana previa: {j.height:,} "
        f"({j.height/gr.height*100:.0f}% de los greeks)")

    gam = pl.col("gamma") * pl.col("multiplier")
    sgn = pl.when(pl.col("is_call")).then(1.0).otherwise(-1.0)
    desc = (j.group_by(["date", "underlying"]).agg([
                (sgn * gam * pl.col("open_interest")).sum().alias("num_now"),
                (sgn * gam * pl.col("oi_prev")).sum().alias("num_old"),
                pl.col("close").first().alias("spot"),
                pl.col("adv_usd").first().alias("adv_usd")])
             .with_columns((0.01 * pl.col("spot") ** 2 / pl.col("adv_usd")).alias("sc"))
             .with_columns([(pl.col("num_now") * pl.col("sc")).alias("gex_sub"),
                            (pl.col("num_old") * pl.col("sc")).alias("gex_old")])
             .with_columns((pl.col("gex_sub") - pl.col("gex_old")).alias("gex_info"))
             .select(["date", "underlying", "gex_old", "gex_info"]))

    # ---- panel con volatilidad y retorno futuros
    filas = []
    for i, t in enumerate(fechas):
        if i + 1 >= len(fechas) or i == 0:
            continue
        nxt, prv = fechas[i + 1], fechas[i - 1]
        fwd = (d.filter((pl.col("date") > t) & (pl.col("date") <= nxt))
                 .group_by("symbol").agg([
                     (pl.col("ret").std() * np.sqrt(252)).alias("rv_fwd"),
                     pl.len().alias("n_fwd"),
                     (pl.col("close").last() / pl.col("close").first() - 1).alias("ret_fwd")]))
        pas = (d.filter((pl.col("date") > prv) & (pl.col("date") <= t))
                 .group_by("symbol")
                 .agg((pl.col("ret").std() * np.sqrt(252)).alias("rv_lag")))
        filas.append(g.filter(pl.col("date") == t)
                      .join(fwd, left_on="underlying", right_on="symbol", how="inner")
                      .join(pas, left_on="underlying", right_on="symbol", how="inner"))
    panel = (pl.concat(filas, how="vertical_relaxed")
               .filter((pl.col("n_fwd") >= 3) & pl.col("rv_fwd").is_finite())
               .filter(~pl.col("underlying").is_in(INDICES))
               .join(desc, on=["date", "underlying"], how="left")
               .with_columns([pl.col("adv_usd").log().alias("log_adv"),
                              pl.col("spot").log().alias("log_spot"),
                              pl.col("total_oi").log().alias("log_oi")])
               .drop_nulls(["rv_fwd", "rv_lag", "iv_median", "gamma_exposure"]))
    say(f"  panel: {panel.height:,} obs · {panel['date'].n_unique()} semanas · "
        f"{panel['underlying'].n_unique()} acciones")

    say("")
    say("-" * 96)
    say("A. FAMA-MACBETH: volatilidad realizada de la semana siguiente ~ Gamma")
    say("-" * 96)
    say("   Gamma en NIVELES, comparable con el -12.92 del paper (Tabla 13, col 1).")
    say("   El paper usa volatilidad MENSUAL; aqui es semanal anualizada, asi que la")
    say("   magnitud no es directamente comparable, pero el SIGNO y el t si.\n")
    say(f"  {'especificacion':<50}{'coef':>11}{'se':>10}{'t (NW)':>9}{'sem':>6}")
    say(f"  {'-'*50}{'-'*11}{'-'*10}{'-'*9}{'-'*6}")
    specs = [
        ("1. Gamma sola (col 1 del paper)", ["gamma_exposure"]),
        ("2. + vol realizada previa", ["gamma_exposure", "rv_lag"]),
        ("3. + IV mediana  <-- el control que importa", ["gamma_exposure", "rv_lag", "iv_median"]),
        ("4. + log(OI), log(ADV$), log(precio)", ["gamma_exposure", "rv_lag", "iv_median",
                                                  "log_oi", "log_adv", "log_spot"]),
    ]
    out = {}
    for nombre, xs in specs:
        (b, se, t, n), _ = fama_macbeth(panel, "rv_fwd", xs)
        out[nombre] = (b, t)
        say(f"  {nombre:<50}{b:>11.3f}{se:>10.3f}{t:>9.2f}{n:>6}")

    say("")
    say("-" * 96)
    say("B. IDENTIFICACION DEL PAPER: cobertura vs informacion privada")
    say("-" * 96)
    say("   Gamma_viejo usa el OI de la semana ANTERIOR con el precio de hoy: son")
    say("   posiciones que ya existian, no pueden venir de informacion posterior.")
    say("   Gamma_info es el cambio por nuevo open interest.")
    say("   Paper: viejo significativo (t=-4.57), info NO significativo.\n")
    say(f"  {'componente':<50}{'coef':>11}{'se':>10}{'t (NW)':>9}{'sem':>6}")
    say(f"  {'-'*50}{'-'*11}{'-'*10}{'-'*9}{'-'*6}")
    sub = panel.drop_nulls(["gex_old", "gex_info"])
    for nombre, xs in [("Gamma_viejo (re-balanceo de cobertura)", ["gex_old", "gex_info"]),
                       ("Gamma_info (informacion privada)", ["gex_info", "gex_old"]),
                       ("Gamma_viejo + IV", ["gex_old", "gex_info", "iv_median", "rv_lag"])]:
        (b, se, t, n), _ = fama_macbeth(sub, "rv_fwd", xs)
        say(f"  {nombre:<50}{b:>11.3f}{se:>10.3f}{t:>9.2f}{n:>6}")

    say("")
    say("-" * 96)
    say("C. RETORNOS (indicativo: ~50 semanas no dan potencia)")
    say("-" * 96)
    say(f"  {'especificacion':<50}{'coef':>11}{'se':>10}{'t (NW)':>9}{'sem':>6}")
    say(f"  {'-'*50}{'-'*11}{'-'*10}{'-'*9}{'-'*6}")
    for nombre, xs in [("retorno ~ Gamma", ["gamma_exposure"]),
                       ("retorno ~ Gamma + IV + vol previa",
                        ["gamma_exposure", "iv_median", "rv_lag"])]:
        (b, se, t, n), sl = fama_macbeth(panel, "ret_fwd", xs)
        say(f"  {nombre:<50}{b:>11.4f}{se:>10.4f}{t:>9.2f}{n:>6}")

    say("")
    say("=" * 96)
    say("VEREDICTO")
    say("=" * 96)
    b1, t1 = out["1. Gamma sola (col 1 del paper)"]
    b3, t3 = out["3. + IV mediana  <-- el control que importa"]
    say(f"  Gamma sola:        coef {b1:+.3f}  t = {t1:+.2f}")
    say(f"  Controlando IV:    coef {b3:+.3f}  t = {t3:+.2f}")
    say("")
    if b1 < 0 and abs(t1) > 1.96 and b3 < 0 and abs(t3) > 1.96:
        say("  MECANISMO CONFIRMADO. El efecto sobrevive al control por volatilidad")
        say("  implicita, asi que no es el artefacto mecanico de gamma ~ 1/sigma.")
    elif b1 < 0 and abs(t1) > 1.96 and not (b3 < 0 and abs(t3) > 1.96):
        say("  EFECTO CRUDO SI, MECANISMO NO. La relacion negativa aparece sin")
        say("  controles pero NO sobrevive al control por volatilidad implicita.")
        say("  Eso es exactamente la firma del confundido mecanico: gamma es")
        say("  proporcional a 1/sigma por construccion. No se puede atribuir al")
        say("  canal de cobertura con estos datos.")
    elif b1 < 0:
        say("  SIGNO CORRECTO, SIN SIGNIFICANCIA. Consistente con el paper, sin")
        say("  poder para rechazar la nula. Es el resultado que el diseno del")
        say("  piloto anticipaba con ~50 semanas.")
    else:
        say("  NO CONFIRMADO: el signo no es el del paper en esta muestra.")
    say("")
    say(f"  reporte: {REP}")
    say("=" * 96)
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
