"""Neutralizacion por beta y por sector del long-short.

PROBLEMA QUE ATACA ESTE MODULO
El long-short original (formar_carteras en engine.py) arma la cartera con
+1/6 a los 6 nombres de Gamma mas baja y -1/6 a los 6 de Gamma mas alta, SIN
mirar de que sector es cada uno ni que tan sensible es al mercado (beta). En
nuestro universo de 30 nombres, con fuerte peso en tech de gran capitalizacion,
esto termino siendo -sin que lo disenaramos asi- una apuesta en contra de tech
disfrazada de apuesta de Gamma: la pata corta (Gamma alto) se llena de
megacaps tech porque son las que tienen las opciones mas liquidas. El
resultado: un drawdown de -35% justo cuando tech tuvo un rally fuerte, que no
tiene nada que ver con si Gamma funciona o no.

LA CORRECCION (dos pasos, mismos datos que ya tenemos, sin comprar nada nuevo)

1) Sector-neutral por construccion: en vez de rankear Gamma en todo el
   universo, DENTRO de cada sector se calcula el z-score de Gamma (que tan
   por encima o por debajo esta cada nombre del promedio de SU sector). Por
   construccion, la suma de esos z-scores dentro de cada sector es cero, asi
   que el resultado ya no puede quedar cargado hacia un sector completo.

2) Beta-neutral por escalado inverso: el score de cada nombre se divide por
   su beta (sensibilidad al mercado, estimada con las mismas 281 velas
   diarias que ya tenemos, en ventana EXPANSIVA -solo con datos hasta la
   fecha de la señal, sin mirar al futuro-). Un nombre de beta alto pesa
   menos por cada unidad de Gamma-residual; uno de beta bajo pesa mas. Esto
   acerca la beta ponderada de la pata larga a la de la pata corta.

Con esto la cartera deja de ser "6 largos, 6 cortos": ahora CADA nombre del
universo recibe un peso proporcional a que tan lejos esta de su propio sector
en Gamma, en vez de un corte duro de los 6 extremos. Es el metodo estandar en
la literatura de factores (peso por z-score, no por cuantil duro) y aqui
ademas es lo que permite lograr neutralidad sectorial real con solo 30
nombres -con un corte duro de 6 muchos sectores se hubieran quedado sin
representacion de un lado.

No es un parametro que se ajusta despues de ver resultados: se aplica igual
sin importar que arroje. El diagnostico (beta_largo vs beta_corto, exposicion
neta por sector) se reporta para verificar si de verdad quedo mas neutral que
la version original -no se fuerza a que quede "bonito".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from .engine import costos, fechas_rebalanceo

# Sectores GICS-aproximados, asignados a mano (son solo 30 tickers conocidos,
# no hace falta comprar clasificacion de ningun proveedor).
SECTORES: dict[str, str] = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMD": "Tech", "CRM": "Tech",
    "MU": "Tech", "PLTR": "Tech", "SHOP": "Tech", "MSTR": "Tech",
    "GOOGL": "Comunicaciones", "META": "Comunicaciones", "NFLX": "Comunicaciones",
    "DIS": "Comunicaciones",
    "AMZN": "ConsumoDisc", "TSLA": "ConsumoDisc", "GME": "ConsumoDisc",
    "RIVN": "ConsumoDisc",
    "WMT": "ConsumoBasico", "KO": "ConsumoBasico", "PG": "ConsumoBasico",
    "JPM": "Financiero", "BAC": "Financiero", "GS": "Financiero",
    "COIN": "Financiero", "SOFI": "Financiero",
    "XOM": "Energia", "CVX": "Energia",
    "BA": "Industrial", "CAT": "Industrial", "UBER": "Industrial",
}


def calcular_beta(eq: pl.DataFrame, fechas: list, min_obs: int = 15,
                   clip: tuple[float, float] = (0.2, 3.5)) -> pl.DataFrame:
    """Beta diario contra SPY, ventana EXPANSIVA hasta cada fecha de señal.

    Point-in-time: para la fecha t solo se usan retornos con date <= t. Con
    min_obs insuficientes (arranque de la muestra) se usa beta=1.0 como
    supuesto neutral, no una estimacion ruidosa con 3-4 datos.
    """
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .select(["date", "symbol", "close"]).sort(["symbol", "date"]))
    wide = d.to_pandas().pivot(index="date", columns="symbol", values="close").sort_index()
    rets = wide.pct_change()
    simbolos = [s for s in wide.columns if s not in ("SPY", "QQQ")]

    filas = []
    for t in fechas:
        tt = pd.Timestamp(t)
        window = rets.loc[rets.index <= tt]
        spy_w = window["SPY"].dropna() if "SPY" in window.columns else pd.Series(dtype=float)
        var_spy = spy_w.var(ddof=1) if len(spy_w) > 1 else np.nan
        n_ok = len(spy_w)
        for sym in simbolos:
            beta, n_common = 1.0, n_ok
            if n_ok >= min_obs and var_spy and np.isfinite(var_spy) and var_spy > 0:
                s = window[sym].dropna()
                common = s.index.intersection(spy_w.index)
                n_common = len(common)
                if n_common >= min_obs:
                    cov = np.cov(s.loc[common], spy_w.loc[common], ddof=1)[0, 1]
                    b = cov / var_spy
                    if np.isfinite(b):
                        beta = float(np.clip(b, clip[0], clip[1]))
            filas.append({"date": t, "symbol": sym, "beta": beta, "n_obs": n_common})
    return pl.DataFrame(filas)


def formar_carteras_neutral(g: pl.DataFrame, beta_df: pl.DataFrame,
                             sectores: dict[str, str] | None = None) -> pl.DataFrame:
    """Pesos sector-neutral (z-score dentro del sector) y beta-escalados.

    g: gamma_exposure ya filtrada (sin SPY/QQQ). Requiere columnas
    date, underlying, gamma_exposure, adv_usd.
    """
    smap = sectores or SECTORES
    gg = (g.with_columns(pl.col("underlying").replace_strict(smap, default="Otro").alias("sector"))
            .join(beta_df, left_on=["date", "underlying"], right_on=["date", "symbol"], how="left")
            .with_columns(pl.col("beta").fill_null(1.0)))

    gg = gg.with_columns([
        pl.col("gamma_exposure").mean().over(["date", "sector"]).alias("sector_mean"),
        pl.col("gamma_exposure").std().over(["date", "sector"]).alias("sector_std"),
    ])
    gg = gg.with_columns(
        ((pl.col("gamma_exposure") - pl.col("sector_mean"))
         / (pl.col("sector_std") + 1e-12)).alias("gamma_z_sector"))
    gg = gg.with_columns((-pl.col("gamma_z_sector") / pl.col("beta")).alias("score"))

    pos_sum = (gg.filter(pl.col("score") > 0).group_by("date")
                 .agg(pl.col("score").sum().alias("pos_sum")))
    neg_sum = (gg.filter(pl.col("score") < 0).group_by("date")
                 .agg(pl.col("score").sum().alias("neg_sum")))
    gg = gg.join(pos_sum, on="date", how="left").join(neg_sum, on="date", how="left")
    gg = gg.with_columns(
        pl.when(pl.col("score") > 0).then(pl.col("score") / pl.col("pos_sum"))
          .when(pl.col("score") < 0).then(pl.col("score") / (-pl.col("neg_sum")))
          .otherwise(0.0).alias("w"))
    return gg


def exposicion_sectorial(pos_pesos: pl.DataFrame) -> pl.DataFrame:
    """Exposicion neta ($ largo + $ corto) por sector y fecha -- diagnostico."""
    return (pos_pesos.group_by(["date", "sector"])
            .agg(pl.col("w").sum().alias("expo_neta"),
                 pl.col("w").filter(pl.col("w") > 0).sum().alias("largo"),
                 pl.col("w").filter(pl.col("w") < 0).sum().alias("corto"))
            .sort(["date", "sector"]))


def simular_neutral(pos_pesos: pl.DataFrame, rets: pl.DataFrame, freq: str = "mensual",
                     borrow: dict | None = None) -> pl.DataFrame:
    """Igual disciplina que simular_periodica: rebalanceo periodico, deriva de
    pesos entre rebalanceos. Sin histeresis (banda) -- aqui no aplica porque
    no hay corte duro de rango, los pesos ya son continuos.
    """
    pos = costos(pos_pesos, borrow)
    pos = pos.join(rets, left_on=["date", "underlying"], right_on=["date", "symbol"],
                   how="inner").filter(pl.col("ret").is_finite())

    fechas = sorted(pos["date"].unique().to_list())
    rebal = set(fechas_rebalanceo(fechas, freq))
    w_act: dict = {}
    filas = []
    for t in fechas:
        s = pos.filter(pl.col("date") == t)
        obj = dict(zip(s["underlying"].to_list(), s["w"].to_list()))
        r = dict(zip(s["underlying"].to_list(), s["ret"].to_list()))
        cu = dict(zip(s["underlying"].to_list(), s["c_unit"].to_list()))
        bo = dict(zip(s["underlying"].to_list(), s["borrow_anual"].to_list()))
        beta_map = dict(zip(s["underlying"].to_list(), s["beta"].to_list()))

        if t in rebal or not w_act:
            nuevo = dict(obj)
            nombres = set(nuevo) | set(w_act)
            turn = sum(abs(nuevo.get(k, 0.0) - w_act.get(k, 0.0)) for k in nombres)
            c_tx = sum(abs(nuevo.get(k, 0.0) - w_act.get(k, 0.0)) * cu.get(k, 3e-4)
                       for k in nombres)
            w_act = nuevo
        else:
            turn = c_tx = 0.0

        bruto = sum(w_act.get(k, 0.0) * r.get(k, 0.0) for k in w_act)
        c_bo = sum(abs(min(w_act.get(k, 0.0), 0.0)) * bo.get(k, 0.005) / 52.0
                   for k in w_act)
        beta_l = sum(v * beta_map.get(k, 1.0) for k, v in w_act.items() if v > 0)
        beta_c = sum(v * beta_map.get(k, 1.0) for k, v in w_act.items() if v < 0)

        filas.append({"date": t, "rebalanceo": t in rebal, "ret_bruto": bruto,
                      "turnover": turn, "costo_tx": c_tx, "costo_borrow": c_bo,
                      "ret_neto": bruto - c_tx - c_bo,
                      "beta_largo": beta_l, "beta_corto": beta_c,
                      "beta_neta": beta_l + beta_c,
                      "n_pos": sum(1 for v in w_act.values() if abs(v) > 1e-12)})
        w_act = {k: v * (1.0 + r.get(k, 0.0)) for k, v in w_act.items()}
    return pl.DataFrame(filas)
