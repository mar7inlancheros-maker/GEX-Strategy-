"""Cadenas de opciones sinteticas con respuesta CONOCIDA.

PARA QUE SIRVEN Y PARA QUE NO
------------------------------
SIRVEN para contrastar el motor: si construyo una cadena de un solo strike con
OI 1.000 y una gamma que puedo calcular a mano, se exactamente que GEX debe
salir. Si el motor no lo reproduce, el motor esta mal, y no hay lugar a debate.

NO SIRVEN para decir absolutamente nada sobre el mercado. Una cadena sintetica
tiene el posicionamiento que yo le he puesto, asi que "descubrir" en ella que el
GEX predice retornos seria descubrir mi propio generador.

Por eso todo lo que sale de aqui lleva `is_synthetic=True` en el DataFrame y el
registro de experimentos lo separa del recuento de intentos reales
(`registry.n_trials`). El sello del PROJECT_PLAN -- SINTETICO, NO ES EVIDENCIA --
se aplica aqui en el origen y no al final, cuando ya nadie recuerda de donde
salio el numero.

DETERMINISMO
------------
Todo generador acepta `seed`. Dos ejecuciones con la misma semilla dan la misma
cadena, byte a byte. Un test que falla de vez en cuando es peor que no tenerlo.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ...options.greeks import bs_gamma

CANONICAL_COLUMNS = [
    "timestamp", "symbol", "underlying_price", "expiration", "tau",
    "strike", "option_type", "open_interest", "implied_volatility",
    "gamma", "bid", "ask", "mid", "volume", "multiplier",
    "risk_free_rate", "dividend_yield", "source", "is_synthetic",
]


def single_strike_chain(
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    tau: float = 0.25,
    iv: float = 0.20,
    call_oi: int = 1_000,
    put_oi: int = 0,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    multiplier: int = 100,
    symbol: str = "TEST",
) -> pd.DataFrame:
    """El caso mas simple posible: un strike, un vencimiento.

    Su GEX se calcula a mano en dos lineas, lo que lo convierte en la piedra de
    toque del motor:

        gamma = bs_gamma(spot, strike, tau, r, iv, q)      # identica call y put
        GEX_spot_scaled = gamma * (call_oi - put_oi) * mult * spot^2 * 0.01

    (la resta viene de la convencion `conventional`: calls +1, puts -1)
    """
    rows = []
    for opt_type, oi in (("C", call_oi), ("P", put_oi)):
        rows.append({
            "strike": float(strike),
            "option_type": opt_type,
            "open_interest": int(oi),
            "implied_volatility": float(iv),
            "tau": float(tau),
        })
    df = pd.DataFrame(rows)
    return _finalize(
        df, spot=spot, symbol=symbol, risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield, multiplier=multiplier,
    )


def make_chain(
    *,
    spot: float = 100.0,
    strike_range: tuple[float, float] = (0.85, 1.15),
    strike_step: float = 1.0,
    expiries_days: tuple[int, ...] = (1, 7, 30, 60),
    base_iv: float = 0.18,
    skew: float = 0.30,
    term_slope: float = 0.02,
    oi_scale: float = 5_000.0,
    oi_concentration: float = 0.04,
    put_call_oi_ratio: float = 1.3,
    call_oi_center: float = 1.03,
    put_oi_center: float = 0.97,
    zero_dte_hours: float = 6.0,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    multiplier: int = 100,
    symbol: str = "TEST",
    as_of: date | None = None,
    seed: int = 20260831,
    oi_noise: float = 0.0,
) -> pd.DataFrame:
    """Cadena realista: varios vencimientos, sonrisa, estructura temporal y OI.

    Los parametros generan una cadena con las regularidades que se observan de
    verdad, sin pretender ser un mercado:

      skew            : la IV sube en strikes bajos (los puts OTM cotizan mas
                        caros: es la sonrisa de indices, no un capricho).
      term_slope      : la IV sube con el plazo (estructura normal).
      oi_concentration: anchura de la campana de OI alrededor de su centro.
      put_call_oi_ratio: hay mas OI en puts que en calls, que es lo habitual en
                        indices porque se usan para cubrir.

    LOS CENTROS DE OI SEPARADOS SON LO QUE HACE QUE EXISTA UN GAMMA FLIP
    -------------------------------------------------------------------
    `call_oi_center` y `put_oi_center` colocan la campana de OI de calls POR
    ENCIMA del spot y la de puts POR DEBAJO, que es como se distribuye de verdad
    en un indice: se compran puts para cubrir (strikes bajos) y calls para
    especular al alza (strikes altos).

    Esto no es cosmetica. Con un `put_call_oi_ratio` constante y los dos centros
    juntos, el GEX neto en cada strike es gamma*(OI_call - OI_put), cuyo signo lo
    fija la razon de OI y NO CAMBIA NUNCA al mover el spot: no existe flip que
    encontrar, por mucho que se busque. Separando los centros, un spot bajo tiene
    cerca los strikes cargados de puts (GEX negativo) y un spot alto los cargados
    de calls (GEX positivo), asi que hay cruce por cero. El flip es exactamente
    ese cambio de vecindario.

    `zero_dte_hours` da a los vencimientos de dte=0 unas horas de vida en vez de
    tau=0. Con tau=0 la opcion esta VENCIDA y su gamma es cero por definicion
    (ver `greeks`), de modo que un "0DTE" con tau=0 no aportaria nada al GEX y el
    caso mas interesante del proyecto quedaria vacio. Seis horas es una sesion a
    media mañana.

    `oi_noise > 0` añade dispersion lognormal reproducible al OI, util para
    comprobar que la deteccion de muros no depende de un perfil perfectamente
    liso.
    """
    rng = np.random.default_rng(seed)
    as_of = as_of or date(2026, 8, 31)

    lo, hi = spot * strike_range[0], spot * strike_range[1]
    strikes = np.round(np.arange(lo, hi + strike_step, strike_step) / strike_step) * strike_step
    strikes = np.unique(strikes)

    rows = []
    for dte in expiries_days:
        # dte=0 no es tau=0: la opcion vence hoy pero aun le quedan horas de vida.
        tau = (zero_dte_hours / 24.0 / 365.0) if dte <= 0 else dte / 365.0
        expiration = as_of + timedelta(days=max(int(dte), 0))
        for K in strikes:
            moneyness = float(K) / spot
            # Sonrisa: pendiente negativa en moneyness, mas plana a plazo largo.
            iv = base_iv + skew * (1.0 - moneyness) + term_slope * np.sqrt(max(tau, 1e-6))
            iv = float(max(iv, 0.01))

            for opt_type in ("C", "P"):
                # Cada tipo tiene SU centro de open interest: calls por encima del
                # spot, puts por debajo. Es lo que crea el gamma flip.
                centre = call_oi_center if opt_type == "C" else put_oi_center
                bell = float(
                    np.exp(-((moneyness - centre) ** 2) / (2.0 * oi_concentration ** 2))
                )
                oi = oi_scale * bell * (put_call_oi_ratio if opt_type == "P" else 1.0)
                if oi_noise > 0:
                    oi *= float(rng.lognormal(mean=0.0, sigma=oi_noise))
                rows.append({
                    "strike": float(K),
                    "option_type": opt_type,
                    "open_interest": int(max(round(oi), 0)),
                    "implied_volatility": iv,
                    "tau": float(tau),
                    "expiration": expiration,
                })

    df = pd.DataFrame(rows)
    return _finalize(
        df, spot=spot, symbol=symbol, risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield, multiplier=multiplier, as_of=as_of,
    )


def _finalize(
    df: pd.DataFrame,
    *,
    spot: float,
    symbol: str,
    risk_free_rate: float,
    dividend_yield: float,
    multiplier: int,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Completa el esquema canonico y calcula la gamma teorica de cada contrato."""
    as_of = as_of or date(2026, 8, 31)
    out = df.copy()

    out["underlying_price"] = float(spot)
    out["symbol"] = symbol
    out["timestamp"] = pd.Timestamp(as_of, tz="UTC")
    out["risk_free_rate"] = float(risk_free_rate)
    out["dividend_yield"] = float(dividend_yield)
    out["multiplier"] = int(multiplier)
    out["source"] = "synthetic"
    out["is_synthetic"] = True

    if "expiration" not in out.columns:
        out["expiration"] = [
            as_of + timedelta(days=int(round(t * 365.0))) for t in out["tau"]
        ]

    out["gamma"] = bs_gamma(
        float(spot),
        out["strike"].to_numpy(dtype=float),
        out["tau"].to_numpy(dtype=float),
        float(risk_free_rate),
        out["implied_volatility"].to_numpy(dtype=float),
        float(dividend_yield),
    )

    # Precios coherentes con un spread proporcional. No pretenden ser realistas;
    # existen para que los controles de calidad tengan algo que validar.
    from ...options.pricing import bs_price

    theo = bs_price(
        float(spot),
        out["strike"].to_numpy(dtype=float),
        out["tau"].to_numpy(dtype=float),
        float(risk_free_rate),
        out["implied_volatility"].to_numpy(dtype=float),
        float(dividend_yield),
        out["option_type"].to_numpy(),
    )
    half_spread = np.maximum(0.01, 0.005 * np.abs(theo))
    out["mid"] = theo
    out["bid"] = np.maximum(theo - half_spread, 0.0)
    out["ask"] = theo + half_spread
    out["volume"] = 0

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[CANONICAL_COLUMNS]


def expected_gex_single_strike(
    *,
    spot: float,
    strike: float,
    tau: float,
    iv: float,
    call_oi: int,
    put_oi: int,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    multiplier: int = 100,
) -> float:
    """GEX esperado de `single_strike_chain` bajo convention/definition por defecto.

    Se calcula APARTE del motor, con la formula escrita a mano. Que las dos
    coincidan es la prueba; si compartieran codigo, no probaria nada.
    """
    gamma = float(bs_gamma(spot, strike, tau, risk_free_rate, iv, dividend_yield))
    net_oi = call_oi - put_oi          # conventional: calls +1, puts -1
    return gamma * net_oi * multiplier * spot ** 2 * 0.01
