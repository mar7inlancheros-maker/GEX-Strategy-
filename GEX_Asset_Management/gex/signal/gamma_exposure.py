"""Ecuacion 1 de Soebhag (2023): net gamma exposure por accion y por dia.

    Gamma_i,t = 0.01 * S_t^2 * sum_j( sign_j * gamma_j * OI_j * m_j ) / ADV$_i,t-1

con sign = +1 para calls y -1 para puts. Con m = 100 se reduce a
    Gamma = S^2 * (sum_calls gamma*OI - sum_puts gamma*OI) / ADV$

El S aparece DOS veces y es intencional: el primero convierte acciones a dolares,
el segundo convierte "movimiento de $1" en "movimiento de 1%". Con un solo S el
ranking queda sesgado por nivel de precio.

PENDIENTES DECLARADOS (afectan el nivel de gamma, no el orden de magnitud):
  - tasa libre de riesgo: constante de config, no curva por vencimiento
  - dividendos: cero. Sesga la gamma de los puts en pagadores de dividendo
    (KO, PG, XOM, CVX, JPM). Cuantificado en la Fase 0.D: usar BSM en vez de CRR
    daba 5-10% de error en sum(gamma); el efecto de ignorar dividendos es del
    mismo orden. Se corrige con la API de corporate actions.
  - desfase del OI: ts_ref viene vacio, asi que el desfase de un dia sigue siendo
    inferencia. Se valida en oi_lag_test().
"""
from __future__ import annotations

import numpy as np
import polars as pl

from gex.pricing.crr import crr_implied_vol_vec, crr_vec

N_STEPS = 400
IV_MIN, IV_MAX = 0.01, 5.0
MAX_REL_SPREAD = 0.50
ATM_BAND = 0.10          # |ln(S/K)| < 0.1  => near-the-money (Bali-Hovakimian)


def add_adv(eq: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    """ADV$ = promedio movil de 21 dias habiles de (cierre x volumen), rezagado 1 dia."""
    return (eq.with_columns(pl.col("ts_event").dt.date().alias("date"))
              .sort(["symbol", "date"])
              .with_columns((pl.col("close") * pl.col("volume")).alias("dollar_vol"))
              .with_columns(
                  pl.col("dollar_vol").rolling_mean(window, min_samples=window)
                    .shift(1).over("symbol").alias("adv_usd"))
              .select(["date", "symbol", "close", "volume", "dollar_vol", "adv_usd"]))


def prepare(chain: pl.DataFrame, eq_adv: pl.DataFrame, r: float = 0.042) -> pl.DataFrame:
    """Une spot y ADV$, calcula T, aplica filtros de calidad de cotizacion."""
    df = chain.join(eq_adv, left_on=["date", "underlying"],
                    right_on=["date", "symbol"], how="inner")
    df = df.with_columns([
        ((pl.col("expiration").dt.date() - pl.col("date")).dt.total_days()
         .cast(pl.Float64) / 365.0).alias("T"),
        pl.lit(r).alias("r"),
        pl.lit(0.0).alias("div_pv"),          # PENDIENTE: dividendos discretos
    ])
    intrinsic = (pl.when(pl.col("is_call"))
                   .then(pl.col("close") - pl.col("strike"))
                   .otherwise(pl.col("strike") - pl.col("close"))
                   .clip(lower_bound=0.0))
    return df.filter(
        (pl.col("T") >= 1.0 / 365.0)
        & (pl.col("mid") > 0)
        & (pl.col("open_interest") > 0)
        & (pl.col("adv_usd") > 0)
        & (pl.col("rel_spread") <= MAX_REL_SPREAD)
        & (pl.col("mid") >= intrinsic - 1e-9)          # sin precios bajo el intrinseco
    )


def solve_greeks(df: pl.DataFrame) -> pl.DataFrame:
    """IV invertida del mid con arbol CRR americano + gamma del mismo arbol."""
    n = df.height
    S = df["close"].to_numpy().astype(np.float64)
    K = df["strike"].to_numpy().astype(np.float64)
    T = df["T"].to_numpy().astype(np.float64)
    r = df["r"].to_numpy().astype(np.float64)
    dv = df["div_pv"].to_numpy().astype(np.float64)
    cp = df["is_call"].to_numpy().astype(np.bool_)
    px = df["mid"].to_numpy().astype(np.float64)
    am = np.ones(n, dtype=np.bool_)

    iv = crr_implied_vol_vec(px, S, K, T, r, dv, cp, am, N_STEPS)
    ok = np.isfinite(iv) & (iv > IV_MIN) & (iv < IV_MAX)
    iv_safe = np.where(ok, iv, 0.3)
    _, delta, gamma = crr_vec(S, K, T, r, iv_safe, dv, cp, am, N_STEPS)

    return df.with_columns([
        pl.Series("iv", iv), pl.Series("iv_ok", ok),
        pl.Series("delta", delta), pl.Series("gamma", gamma),
        pl.Series("ln_moneyness", np.log(S / K)),
    ]).filter(pl.col("iv_ok") & pl.col("gamma").is_finite() & (pl.col("gamma") >= 0))


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Ecuacion 1 + descomposiciones por moneyness y por vencimiento."""
    g = pl.col("gamma") * pl.col("open_interest") * pl.col("multiplier")
    signed = pl.when(pl.col("is_call")).then(g).otherwise(-g)
    is_atm = pl.col("ln_moneyness").abs() < ATM_BAND
    is_otm = ((pl.col("is_call") & (pl.col("ln_moneyness") < -ATM_BAND))
              | (~pl.col("is_call") & (pl.col("ln_moneyness") > ATM_BAND)))
    is_fast = pl.col("T") <= 31.0 / 365.0

    agg = df.group_by(["date", "underlying"]).agg([
        signed.sum().alias("net_num"),
        g.abs().sum().alias("gross_num"),
        pl.when(is_atm).then(signed).otherwise(0.0).sum().alias("num_atm"),
        pl.when(is_otm).then(signed).otherwise(0.0).sum().alias("num_otm"),
        pl.when(~is_atm & ~is_otm).then(signed).otherwise(0.0).sum().alias("num_itm"),
        pl.when(is_fast).then(signed).otherwise(0.0).sum().alias("num_fast"),
        pl.when(~is_fast).then(signed).otherwise(0.0).sum().alias("num_slow"),
        pl.col("close").first().alias("spot"),
        pl.col("adv_usd").first().alias("adv_usd"),
        pl.len().alias("n_contracts"),
        pl.col("open_interest").sum().alias("total_oi"),
        pl.col("iv").median().alias("iv_median"),
    ])
    scale = 0.01 * pl.col("spot") ** 2 / pl.col("adv_usd")
    return agg.with_columns([
        (pl.col("net_num") * scale).alias("gamma_exposure"),
        (pl.col("gross_num") * scale).alias("gamma_gross"),
        (pl.col("num_atm") * scale).alias("gex_atm"),
        (pl.col("num_otm") * scale).alias("gex_otm"),
        (pl.col("num_itm") * scale).alias("gex_itm"),
        (pl.col("num_fast") * scale).alias("gex_fast"),
        (pl.col("num_slow") * scale).alias("gex_slow"),
    ]).with_columns(
        (pl.col("gamma_exposure").abs() / pl.col("gamma_gross")).alias("net_gross_ratio")
    ).sort(["date", "gamma_exposure"])


def winsorize_zscore(g: pl.DataFrame, col: str = "gamma_exposure") -> pl.DataFrame:
    """Recorte 1%/99% por fecha (como el paper) + z-score cross-seccional."""
    return (g.with_columns([
                pl.col(col).quantile(0.01).over("date").alias("_lo"),
                pl.col(col).quantile(0.99).over("date").alias("_hi")])
             .with_columns(pl.col(col).clip(pl.col("_lo"), pl.col("_hi")).alias("gex_w"))
             .with_columns(((pl.col("gex_w") - pl.col("gex_w").mean().over("date"))
                            / pl.col("gex_w").std().over("date")).alias("gex_z"))
             .drop(["_lo", "_hi"]))
