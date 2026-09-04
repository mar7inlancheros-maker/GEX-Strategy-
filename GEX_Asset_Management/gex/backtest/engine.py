"""Motor de backtest: simula el portafolio semana a semana.

Esto es lo que faltaba. Las puertas P1-P3b validan que la SENAL este bien
construida y que el MECANISMO exista. Este modulo responde otra pregunta:
cuanto habria rendido el portafolio.

DISCIPLINA POINT-IN-TIME
La senal de la fecha t se calcula con datos disponibles al cierre de t, y el
portafolio se mantiene desde el cierre de t hasta el cierre de t+1. Nunca se usa
informacion de t+1 para decidir en t. El open interest ademas ya viene con su
desfase natural de un dia (OPRA lo disemina antes de la apertura reflejando la
sesion anterior), asi que el sesgo va en contra nuestra, no a favor.

MODELO DE COSTOS
  - medio spread del subyacente: 2 bps por operacion en mega-caps liquidas,
    5 bps en los nombres de menor ADV$ (< $1.000M)
  - comision: 1 bp por operacion
  - costo de prestamo de la pata corta: se usa el CARRY IMPLICITO que extrajimos
    de la paridad put-call. Para GME, RIVN y SOFI ese carry ES el costo de
    prestamo del papel, medido en el propio mercado de opciones. Es la forma
    honesta de cobrarle a la pata corta lo que realmente cuesta.
"""
from __future__ import annotations

import numpy as np
import polars as pl

HALF_SPREAD_BPS_LIQ = 2.0
HALF_SPREAD_BPS_ILIQ = 5.0
COMMISSION_BPS = 1.0
ADV_LIQ_UMBRAL = 1_000e6
BORROW_MIN_ANUAL = 0.005          # 50 bps para papel facil de prestar


def semanal_returns(eq: pl.DataFrame, fechas: list) -> pl.DataFrame:
    """Retorno de cada accion entre fechas consecutivas de senal."""
    d = (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
           .select(["date", "symbol", "close"]))
    filas = []
    for i in range(len(fechas) - 1):
        t, nxt = fechas[i], fechas[i + 1]
        a = d.filter(pl.col("date") <= t).group_by("symbol").agg(
            pl.col("close").last().alias("p0"))
        b = d.filter(pl.col("date") <= nxt).group_by("symbol").agg(
            pl.col("close").last().alias("p1"))
        filas.append(a.join(b, on="symbol", how="inner")
                      .with_columns([(pl.col("p1") / pl.col("p0") - 1).alias("ret"),
                                     pl.lit(t).alias("date")]))
    return pl.concat(filas, how="vertical_relaxed").select(["date", "symbol", "ret"])


def formar_carteras(g: pl.DataFrame, n_pata: int = 6) -> pl.DataFrame:
    """Asigna pesos: +1/n a la Gamma mas BAJA, -1/n a la mas ALTA."""
    r = (g.with_columns(pl.col("gamma_exposure").rank("ordinal").over("date").alias("rk"))
          .with_columns(pl.col("gamma_exposure").count().over("date").alias("n")))
    return r.with_columns(
        pl.when(pl.col("rk") <= n_pata).then(1.0 / n_pata)
         .when(pl.col("rk") > pl.col("n") - n_pata).then(-1.0 / n_pata)
         .otherwise(0.0).alias("w"))


def costos(pos: pl.DataFrame, borrow: dict | None = None) -> pl.DataFrame:
    """Costo de transaccion por rotacion + costo de prestamo de los cortos."""
    hs = (pl.when(pl.col("adv_usd") >= ADV_LIQ_UMBRAL)
            .then(HALF_SPREAD_BPS_LIQ).otherwise(HALF_SPREAD_BPS_ILIQ))
    bmap = borrow or {}
    return pos.with_columns([
        ((hs + COMMISSION_BPS) / 1e4).alias("c_unit"),
        pl.col("underlying").replace_strict(bmap, default=BORROW_MIN_ANUAL)
          .clip(BORROW_MIN_ANUAL, 0.30).alias("borrow_anual"),
    ])


def simular(g: pl.DataFrame, rets: pl.DataFrame, n_pata: int = 6,
            borrow: dict | None = None, solo_largo: bool = False) -> pl.DataFrame:
    """Corre la simulacion y devuelve la serie de retornos por semana."""
    pos = formar_carteras(g, n_pata)
    if solo_largo:
        pos = pos.with_columns(pl.when(pl.col("w") > 0).then(pl.col("w") * 2.0)
                                 .otherwise(0.0).alias("w"))
    pos = costos(pos, borrow)
    pos = pos.join(rets, left_on=["date", "underlying"], right_on=["date", "symbol"],
                   how="inner").filter(pl.col("ret").is_finite())

    fechas = sorted(pos["date"].unique().to_list())
    prev = {}
    filas = []
    for t in fechas:
        s = pos.filter(pl.col("date") == t)
        w = dict(zip(s["underlying"].to_list(), s["w"].to_list()))
        r = dict(zip(s["underlying"].to_list(), s["ret"].to_list()))
        cu = dict(zip(s["underlying"].to_list(), s["c_unit"].to_list()))
        bo = dict(zip(s["underlying"].to_list(), s["borrow_anual"].to_list()))

        bruto = sum(w[k] * r[k] for k in w)
        # rotacion: |peso nuevo - peso anterior|, con el costo unitario de cada nombre
        nombres = set(w) | set(prev)
        turn = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in nombres)
        c_tx = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) * cu.get(k, 3e-4)
                   for k in nombres)
        c_bo = sum(abs(min(w[k], 0.0)) * bo[k] / 52.0 for k in w)
        filas.append({"date": t, "ret_bruto": bruto, "turnover": turn,
                      "costo_tx": c_tx, "costo_borrow": c_bo,
                      "ret_neto": bruto - c_tx - c_bo,
                      "n_largo": sum(1 for v in w.values() if v > 0),
                      "n_corto": sum(1 for v in w.values() if v < 0)})
        prev = w
    return pl.DataFrame(filas)


def metricas(s: pl.DataFrame, col: str = "ret_neto", per_year: float = 52.0) -> dict:
    x = s[col].to_numpy().astype(float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {}
    eq = np.cumprod(1.0 + x)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    mu, sd = x.mean(), x.std(ddof=1)
    t = mu / (sd / np.sqrt(len(x))) if sd > 0 else np.nan
    return {"n": len(x),
            "ret_total": float(eq[-1] - 1.0),
            "ret_anual": float((1 + mu) ** per_year - 1),
            "vol_anual": float(sd * np.sqrt(per_year)),
            "sharpe": float(mu / sd * np.sqrt(per_year)) if sd > 0 else np.nan,
            "max_dd": float(dd.min()),
            "hit_rate": float((x > 0).mean()),
            "t_stat": float(t),
            "media_sem": float(mu),
            "ic95_lo": float((mu - 1.96 * sd / np.sqrt(len(x))) * per_year),
            "ic95_hi": float((mu + 1.96 * sd / np.sqrt(len(x))) * per_year)}


# ---------------------------------------------------------------------------
# Rebalanceo periodico: la senal se recalcula cada semana, pero la CARTERA solo
# se re-arma en las fechas de rebalanceo. Entre medias los pesos DERIVAN con los
# retornos, que es lo que pasa de verdad en una cuenta.
#
# Por que importa: con rebalanceo semanal el turnover salio 150% por semana --
# en quintiles de 6 sobre 30 nombres casi todos entran y salen cada semana. El
# paper rebalancea MENSUAL. Ademas, medir los retornos semanalmente aunque se
# rebalancee mensual da 53 observaciones en vez de 13 para el mismo diseno: mas
# precision sobre el drawdown y la volatilidad sin cambiar la estrategia.
# ---------------------------------------------------------------------------

def fechas_rebalanceo(fechas: list, freq: str = "mensual") -> list:
    """Ultima fecha de senal de cada mes (o todas, si es semanal)."""
    if freq == "semanal":
        return list(fechas)
    por_mes = {}
    for f in fechas:
        por_mes[(f.year, f.month)] = f      # se queda la ultima de cada mes
    return [v for _, v in sorted(por_mes.items())]


def simular_periodica(g: pl.DataFrame, rets: pl.DataFrame, freq: str = "mensual",
                      n_pata: int = 6, borrow: dict | None = None,
                      solo_largo: bool = False, banda: int = 0) -> pl.DataFrame:
    """Rebalanceo periodico con deriva de pesos entre fechas de rebalanceo.

    banda: histeresis. Un nombre que ya esta en la cartera se mantiene mientras
    su rango siga dentro de (n_pata + banda). Reduce turnover sin cambiar la
    tesis: evita rotar por un cruce marginal del umbral.
    """
    pos = costos(formar_carteras(g, n_pata), borrow)
    if solo_largo:
        pos = pos.with_columns(pl.when(pl.col("w") > 0).then(pl.col("w") * 2.0)
                               .otherwise(0.0).alias("w"))
    pos = pos.join(rets, left_on=["date", "underlying"],
                   right_on=["date", "symbol"], how="inner")
    pos = pos.filter(pl.col("ret").is_finite())

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
        rk = dict(zip(s["underlying"].to_list(), s["rk"].to_list()))
        n_uni = int(s["n"][0]) if s.height else 30

        if t in rebal or not w_act:
            nuevo = dict(obj)
            if banda and w_act:
                for k, wv in w_act.items():          # histeresis
                    if k in rk and abs(wv) > 1e-12 and abs(nuevo.get(k, 0.0)) < 1e-12:
                        dentro = (wv > 0 and rk[k] <= n_pata + banda) or \
                                 (wv < 0 and rk[k] > n_uni - n_pata - banda)
                        if dentro:
                            nuevo[k] = np.sign(wv) / n_pata
            nombres = set(nuevo) | set(w_act)
            turn = sum(abs(nuevo.get(k, 0.0) - w_act.get(k, 0.0)) for k in nombres)
            c_tx = sum(abs(nuevo.get(k, 0.0) - w_act.get(k, 0.0)) * cu.get(k, 3e-4)
                       for k in nombres)
            w_act = nuevo
        else:
            turn = c_tx = 0.0

        bruto = sum(w_act.get(k, 0.0) * r.get(k, 0.0) for k in w_act)
        per_year = 52.0
        c_bo = sum(abs(min(w_act.get(k, 0.0), 0.0)) * bo.get(k, BORROW_MIN_ANUAL)
                   / per_year for k in w_act)
        filas.append({"date": t, "rebalanceo": t in rebal, "ret_bruto": bruto,
                      "turnover": turn, "costo_tx": c_tx, "costo_borrow": c_bo,
                      "ret_neto": bruto - c_tx - c_bo,
                      "n_pos": sum(1 for v in w_act.values() if abs(v) > 1e-12)})
        # deriva de los pesos con los retornos hasta el proximo rebalanceo
        w_act = {k: v * (1.0 + r.get(k, 0.0)) for k, v in w_act.items()}
    return pl.DataFrame(filas)
