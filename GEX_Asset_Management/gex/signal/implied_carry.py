"""Carry implicito (tasa + dividendos + costo de prestamo) de la paridad put-call.

POR QUE LA PRIMERA VERSION FALLO SU PROPIA PRUEBA
La v1 sacaba r y D de la regresion  C - P = a + b*K  usando pendiente e intercepto:
    b = -DF   ->  r = -ln(DF)/T
    a = DF*F  ->  D = S - a
Matematicamente correcto, estadisticamente terrible. El intercepto es el valor de
la recta en K = 0, y los strikes usados estan alrededor de S (275 en AAPL): es una
extrapolacion larguisima que multiplica el ruido. Resultado observado: AAPL, MSFT,
CAT, MU, CRM, META salieron con dividendo exactamente 0.00% (la mediana de estimados
ruidosos alrededor de cero, recortada en cero), y la r implicita salio dispersa
entre 0.72% y 6.92% SEGUN LA ACCION -- imposible, la tasa libre de riesgo es una
sola para todo el mercado. Esa dispersion era la prueba de que el estimador estaba
mezclando r con el carry de cada nombre.

LO QUE SI QUEDO CLARO EN LA V1
Los "dividendos" de GME (2.76%), RIVN (1.43%) y SOFI (1.14%) no son un error: son
el COSTO DE PRESTAMO del papel. Son nombres con short interest alto y dificiles de
pedir prestados, y la paridad put-call recoge el carry TOTAL, no solo el dividendo
declarado. Para valuar la opcion eso es lo correcto -- por eso este modulo se llama
carry y no dividendos.

ESTIMADOR V2, en dos etapas que respetan la estructura del problema

Etapa 1 -- la tasa es UNA SOLA por fecha.
    De cada regresion se usa SOLO la pendiente (que identifica DF sin extrapolar) y
    se agrupan todas las acciones de la fecha para estimar una r comun:
        r_fecha = mediana sobre todos los ajustes de  -ln(DF)/T
    Agrupar cientos de ajustes promedia el ruido y respeta que r no depende del emisor.

Etapa 2 -- el carry por accion, SIN extrapolar.
    Con DF(T) = exp(-r_fecha * T) ya fijo, para CADA strike se despeja directamente
        D_k = S - K*DF - (C - P)_k
    y se toma la MEDIANA sobre los strikes cerca del dinero. Cada D_k es una
    observacion directa, no una extrapolacion a K = 0: la varianza cae muchisimo.

CAVEAT QUE SIGUE ABIERTO
La paridad es exacta para europeas. En americanas el put ITM carga prima de
ejercicio anticipado, que baja C - P e infla D. Se mitiga usando solo strikes cerca
del dinero, y se reporta por separado el estimado restringido a K <= S (donde el put
esta OTM y esa prima es minima) como diagnostico del sesgo residual.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MNY_BAND = 0.12          # |ln(S/K)| <= 0.12
MIN_PAIRS = 3
T_MIN_FIT = 0.05         # plazos muy cortos amplifican el ruido de la pendiente
DF_MIN, DF_MAX = 0.55, 1.02
R_MIN, R_MAX = -0.01, 0.15
Q_MAX = 0.15             # carry anualizado maximo admitido


def _pairs(df: pl.DataFrame) -> pl.DataFrame:
    calls = (df.filter(pl.col("is_call"))
               .select(["date", "underlying", "expiration", "strike", "mid",
                        "close", "T"]).rename({"mid": "c"}))
    puts = (df.filter(~pl.col("is_call"))
              .select(["date", "underlying", "expiration", "strike", "mid"])
              .rename({"mid": "p"}))
    return (calls.join(puts, on=["date", "underlying", "expiration", "strike"],
                       how="inner")
                 .with_columns((pl.col("close") / pl.col("strike")).log()
                               .alias("lnm"))
                 .filter(pl.col("lnm").abs() <= MNY_BAND))


def implied_carry(df: pl.DataFrame, r_fallback: float = 0.042,
                  r_curve: pl.DataFrame | None = None) -> pl.DataFrame:
    """Carry por (fecha, accion, vencimiento) a partir de la paridad put-call.

    Con `r_curve` (curva del Tesoro, ver gex.curves) la tasa se LEE en vez de
    estimarse: la pendiente de la paridad no resuelve r en vencimientos cortos
    y arrastraba el estimado 3-5x por debajo del real. La paridad se sigue
    usando para D = dividendo + costo de prestamo, que es donde si vive.
    """
    pares = _pairs(df)
    grupos = []
    for key, g in pares.group_by(["date", "underlying", "expiration"],
                                 maintain_order=True):
        if g.height < MIN_PAIRS:
            continue
        K = g["strike"].to_numpy().astype(float)
        y = (g["c"] - g["p"]).to_numpy().astype(float)
        S = float(g["close"][0]); T = float(g["T"][0])
        if T <= 0 or K.std() < 1e-9:
            continue
        b = np.polyfit(K, y, 1)[0]
        grupos.append((key, K, y, S, T, -b, g["lnm"].to_numpy().astype(float)))

    # ---- Etapa 1: la tasa
    if r_curve is not None:
        # observada: r(fecha, T) de la curva del Tesoro, interpolada por plazo
        from gex.curves import rate_lookup
        r_grupo = rate_lookup(r_curve, [k[0] for k, *_ in grupos],
                              [g[4] for g in grupos])
        r_fecha = None
    else:
        # estimada de las pendientes (sesgada en vencimientos cortos, ver docstring)
        r_grupo = None
        por_fecha = {}
        for key, K, y, S, T, DF, lnm in grupos:
            if T >= T_MIN_FIT and DF_MIN < DF < DF_MAX:
                por_fecha.setdefault(key[0], []).append(-np.log(DF) / T)
        r_fecha = {d: float(np.median(v)) for d, v in por_fecha.items() if len(v) >= 5}

    # ---- Etapa 2: carry por (fecha, accion, vencimiento), sin extrapolar
    out = []
    for i, (key, K, y, S, T, DF_raw, lnm) in enumerate(grupos):
        r = float(r_grupo[i]) if r_grupo is not None else r_fecha.get(key[0], r_fallback)
        if not (R_MIN < r < R_MAX):
            r = r_fallback
        DF = float(np.exp(-r * T))
        Dk = S - K * DF - y                      # una observacion por strike
        D = float(np.median(Dk))
        sel = lnm >= 0.0                         # K <= S: put OTM, sesgo americano minimo
        D_otm = float(np.median(Dk[sel])) if sel.sum() >= 2 else np.nan
        if not (-0.01 * S <= D <= Q_MAX * S * max(T, 0.02)):
            continue
        out.append({"date": key[0], "underlying": key[1], "expiration": key[2],
                    "r_impl": r, "div_pv_impl": max(D, 0.0),
                    "div_pv_otm": D_otm, "spot": S, "T": T,
                    "n_pairs": len(K), "disp": float(np.std(Dk))})
    if not out:
        return pl.DataFrame()
    return pl.DataFrame(out)


def attach_carry(df: pl.DataFrame, carry: pl.DataFrame,
                 r_fallback: float = 0.042,
                 r_curve: pl.DataFrame | None = None) -> pl.DataFrame:
    """Cobertura completa: r por contrato y carry q por accion, D(T) = S*q*T.

    Con `r_curve` la tasa se interpola al plazo T de CADA contrato (r(T), como
    pedia el plan); sin ella se usa la mediana por fecha del r estimado.
    """
    if r_curve is not None:
        from gex.curves import rate_lookup
        r_contrato = rate_lookup(r_curve, df["date"].to_list(),
                                 df["T"].to_numpy())
        df = df.with_columns(pl.Series("r_curva", r_contrato))
    if carry.is_empty():
        r0 = pl.col("r_curva") if r_curve is not None else pl.lit(r_fallback)
        return df.with_columns([r0.alias("r"), pl.lit(0.0).alias("div_pv"),
                                pl.lit(r_curve is not None).alias("carry_ok")])
    r_fecha = carry.group_by("date").agg(pl.col("r_impl").median().alias("r_date"))
    q_accion = (carry.filter(pl.col("T") > T_MIN_FIT)
                     .with_columns((pl.col("div_pv_impl") / pl.col("spot")
                                    / pl.col("T")).alias("q"))
                     .group_by(["date", "underlying"])
                     .agg(pl.col("q").median().alias("q_div")))
    r_fecha = r_fecha.with_columns(pl.col("date").cast(df.schema["date"]))
    q_accion = q_accion.with_columns(pl.col("date").cast(df.schema["date"]))
    j = (df.join(r_fecha, on="date", how="left")
           .join(q_accion, on=["date", "underlying"], how="left")
           .with_columns([
               (pl.col("r_curva") if r_curve is not None
                else pl.col("r_date").fill_null(r_fallback)).alias("r"),
               pl.col("q_div").fill_null(0.0).clip(0.0, Q_MAX).alias("q_div")])
           .with_columns([
               (pl.col("close") * pl.col("q_div") * pl.col("T")).alias("div_pv"),
               (pl.lit(True) if r_curve is not None
                else pl.col("r_date").is_not_null()).alias("carry_ok")])
           .drop("r_date"))
    return j.drop("r_curva") if r_curve is not None else j


def resumen_por_accion(carry: pl.DataFrame) -> pl.DataFrame:
    return (carry.filter(pl.col("T") > T_MIN_FIT)
                 .with_columns([
                     (pl.col("div_pv_impl") / pl.col("spot") / pl.col("T")).alias("q"),
                     (pl.col("div_pv_otm") / pl.col("spot") / pl.col("T")).alias("q_otm")])
                 .group_by("underlying")
                 .agg([pl.col("q").median().alias("div_yield_med"),
                       pl.col("q_otm").median().alias("q_otm_med"),
                       pl.col("r_impl").median().alias("r_med"),
                       pl.col("disp").median().alias("rmse_med"),
                       pl.len().alias("n")])
                 .sort("div_yield_med", descending=True))
