"""Griegas de Black-Scholes-Merton. La gamma es la pieza critica del proyecto.

UNIDADES — declaradas, porque casi todos los errores de GEX son de unidades
----------------------------------------------------------------------------
    delta   sin unidades. Variacion del precio de la opcion por cada $1 del
            subyacente, POR ACCION (no por contrato).
    gamma   1/$. Variacion de delta por cada $1 del subyacente, POR ACCION.
            Es la que multiplica el GEX; el multiplicador de contrato (100) lo
            aplica `gex.py`, NO este modulo. Aplicarlo dos veces da un GEX 100
            veces mayor y sigue teniendo un aspecto perfectamente razonable.
    vega    $ por 1,00 de volatilidad (es decir, por 100 puntos de IV).
            `vega_per_pct` lo da por punto porcentual, que es como se lee.
    theta   $ por AÑO. `theta_per_day` divide entre 365.

PARIDAD PUT-CALL: LA MEJOR PRUEBA QUE TIENE ESTE MODULO
-------------------------------------------------------
Bajo BSM la gamma de una call y la de una put con el mismo strike y vencimiento
son IDENTICAS. Se deduce de la paridad: C - P = S e^{-qT} - K e^{-rT}, cuyo lado
derecho es lineal en S, luego su segunda derivada es cero, luego
d2C/dS2 = d2P/dS2.

Esto tiene una consecuencia que conviene tener presente en todo el proyecto:

    LA DIFERENCIA ENTRE EL GEX DE CALLS Y EL DE PUTS NO VIENE DE LA GAMMA.
    Viene ENTERA de la convencion de signo y del open interest.

Es decir: el "call gamma" y el "put gamma" que publica todo el mundo no miden dos
curvaturas distintas. Miden la misma curvatura repartida segun un supuesto sobre
quien esta al otro lado (A1). Por eso el placebo de invertir el signo importa
tanto: si el signo no aporta, el GEX neto es poco mas que OI ponderado por gamma.

FORMULAS (con rendimiento por dividendo continuo q)
---------------------------------------------------
    gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))          <- igual para call y put
    delta_call = e^{-qT} N(d1)
    delta_put  = e^{-qT} (N(d1) - 1)
    vega  = S e^{-qT} phi(d1) sqrt(T)                    <- igual para call y put
    theta_call = -S phi(d1) sigma e^{-qT}/(2 sqrt(T))
                 + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
    theta_put  = -S phi(d1) sigma e^{-qT}/(2 sqrt(T))
                 - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)

SPY ES AMERICANA (SUPUESTO A4)
------------------------------
Estas formulas son europeas. Para SPY son una APROXIMACION cuyo error se
concentra en puts muy ITM, que es justo donde el open interest puede ser alto.
`pricing.american_binomial_gamma` existe para medir ese error; `compute_greeks`
acepta `exercise_style` y devuelve el aviso en `GreeksResult.warnings` para que
nadie lo aplique a SPY sin enterarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pricing import (
    SIGMA_FLOOR_DEFAULT,
    T_FLOOR_DEFAULT,
    OptionType,
    _is_call_mask,
    _norm_cdf,
    _norm_pdf,
    _prepare,
    d1_d2,
)


# --------------------------------------------------------------------------- #
# Griegas individuales
# --------------------------------------------------------------------------- #

def bs_gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> NDArray[np.float64]:
    """Gamma BSM, por accion, en 1/$.

        gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))

    Identica para call y put: no lleva `option_type` a proposito. Si alguna vez
    alguien le pasa uno, que falle.

    Limites (ver `pricing` para la politica completa):
      - T == 0        -> 0,0. La opcion ha vencido: no hay curvatura que cubrir.
      - 0 < T < suelo -> se aplica el suelo; el recuento sale en `compute_greeks`.
      - muy ITM/OTM   -> phi(d1) hace underflow y gamma -> 0. Es CORRECTO, no un
                         fallo: una opcion lejana no tiene curvatura.
      - sigma == 0    -> suelo de sigma, por la misma razon que T.
    """
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    d1, _ = d1_d2(Sa, Ka, Ta, ra, sa, qa, t_floor=t_floor, sigma_floor=sigma_floor)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.exp(-qa * Ta) * _norm_pdf(d1) / (Sa * sa * np.sqrt(Ta))

    gamma = np.where(expired, 0.0, gamma)
    gamma = np.where(invalid, np.nan, gamma)
    # phi(d1) puede desbordar a 0 y dejar un 0/0 -> NaN en casos extremos. El
    # limite economico ahi es cero, no indefinido.
    gamma = np.where(~np.isfinite(gamma) & ~invalid, 0.0, gamma)
    return gamma


def bs_delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: ArrayLike | OptionType = "C",
    *,
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> NDArray[np.float64]:
    """Delta BSM, por accion. Call en (0,1), put en (-1,0).

    En T == 0 el limite es la funcion escalon: 1 si la call acaba ITM, 0 si no.
    Justo en S == K no esta definida; se devuelve 0,5 (call) por continuidad por
    la izquierda del promedio, y se marca. Es un caso de medida nula en datos
    reales pero aparece en los tests.
    """
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    is_call = _is_call_mask(option_type, Sa.shape)
    d1, _ = d1_d2(Sa, Ka, Ta, ra, sa, qa, t_floor=t_floor, sigma_floor=sigma_floor)

    disc_q = np.exp(-qa * Ta)
    nd1 = _norm_cdf(d1)
    delta = np.where(is_call, disc_q * nd1, disc_q * (nd1 - 1.0))

    itm_call = np.where(Sa > Ka, 1.0, np.where(Sa < Ka, 0.0, 0.5))
    itm_put = np.where(Sa < Ka, -1.0, np.where(Sa > Ka, 0.0, -0.5))
    delta = np.where(expired, np.where(is_call, itm_call, itm_put), delta)
    return np.where(invalid, np.nan, delta)


def bs_vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> NDArray[np.float64]:
    """Vega BSM, en $ por 1,00 de volatilidad. Identica para call y put."""
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    d1, _ = d1_d2(Sa, Ka, Ta, ra, sa, qa, t_floor=t_floor, sigma_floor=sigma_floor)
    vega = Sa * np.exp(-qa * Ta) * _norm_pdf(d1) * np.sqrt(Ta)
    vega = np.where(expired, 0.0, vega)
    vega = np.where(invalid, np.nan, vega)
    return np.where(~np.isfinite(vega) & ~invalid, 0.0, vega)


def bs_theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: ArrayLike | OptionType = "C",
    *,
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> NDArray[np.float64]:
    """Theta BSM, en $ por AÑO (negativa casi siempre). Dividir entre 365 por dia."""
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    is_call = _is_call_mask(option_type, Sa.shape)
    d1, d2 = d1_d2(Sa, Ka, Ta, ra, sa, qa, t_floor=t_floor, sigma_floor=sigma_floor)

    disc_r = np.exp(-ra * Ta)
    disc_q = np.exp(-qa * Ta)
    decay = -(Sa * _norm_pdf(d1) * sa * disc_q) / (2.0 * np.sqrt(Ta))

    theta_call = decay + qa * Sa * disc_q * _norm_cdf(d1) - ra * Ka * disc_r * _norm_cdf(d2)
    theta_put = decay - qa * Sa * disc_q * _norm_cdf(-d1) + ra * Ka * disc_r * _norm_cdf(-d2)
    theta = np.where(is_call, theta_call, theta_put)

    theta = np.where(expired, 0.0, theta)
    return np.where(invalid, np.nan, theta)


# --------------------------------------------------------------------------- #
# Calculo conjunto con diagnostico
# --------------------------------------------------------------------------- #

@dataclass
class GreeksResult:
    """Las cuatro griegas mas el diagnostico de que paso al calcularlas.

    El diagnostico no es adorno. `n_floored` y `gamma_share_floored` responden a
    la unica pregunta que importa el dia de vencimiento: ¿cuanta de la gamma
    total viene de contratos cuya gamma la fijo un suelo arbitrario en lugar del
    mercado? Si esa fraccion es alta, el GEX de ese dia mide el suelo.
    """

    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]

    n_total: int = 0
    n_invalid: int = 0
    n_expired: int = 0
    n_floored: int = 0
    gamma_share_floored: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def vega_per_pct(self) -> NDArray[np.float64]:
        """Vega por punto porcentual de IV, que es como se lee en pantalla."""
        return self.vega / 100.0

    @property
    def theta_per_day(self) -> NDArray[np.float64]:
        return self.theta / 365.0

    def report(self) -> str:
        lines = [
            f"contratos            : {self.n_total}",
            f"invalidos (NaN)      : {self.n_invalid}",
            f"vencidos (T=0)       : {self.n_expired}",
            f"con suelo aplicado   : {self.n_floored}",
            f"gamma que viene del suelo: {self.gamma_share_floored:.1%}",
        ]
        lines.extend(f"AVISO: {w}" for w in self.warnings)
        return "\n".join(lines)


def compute_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: ArrayLike | OptionType = "C",
    *,
    exercise_style: str = "european",
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> GreeksResult:
    """Las cuatro griegas de una vez, con el diagnostico del calculo.

    `exercise_style="american"` no cambia la formula -- seguimos usando BSM --
    pero deja constancia del supuesto A4 en `warnings`. Preferimos un aviso
    visible a una gamma americana a medias: la de verdad esta en
    `pricing.american_binomial_gamma` y se usa para MEDIR el error, no para
    sustituir a esta en la cadena entera.
    """
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, floored = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )

    kwargs = dict(t_floor=t_floor, sigma_floor=sigma_floor)
    gamma = bs_gamma(Sa, Ka, Ta, ra, sa, qa, **kwargs)
    delta = bs_delta(Sa, Ka, Ta, ra, sa, qa, option_type, **kwargs)
    vega = bs_vega(Sa, Ka, Ta, ra, sa, qa, **kwargs)
    theta = bs_theta(Sa, Ka, Ta, ra, sa, qa, option_type, **kwargs)

    total_gamma = float(np.nansum(np.abs(gamma)))
    floored_gamma = float(np.nansum(np.abs(gamma[floored]))) if np.any(floored) else 0.0
    share = floored_gamma / total_gamma if total_gamma > 0 else 0.0

    warnings: list[str] = []
    if exercise_style.lower() == "american":
        warnings.append(
            "estilo americano: la gamma BSM es una APROXIMACION (supuesto A4). "
            "El error se concentra en puts muy ITM. Medirlo con "
            "pricing.american_binomial_gamma antes de fiarse del agregado."
        )
    if share > 0.10:
        warnings.append(
            f"el {share:.1%} de la gamma total viene de contratos que tocaron el "
            f"suelo t_floor={t_floor:.2e}. El agregado depende del suelo, no del "
            f"mercado: barrer la sensibilidad antes de reportarlo."
        )
    if int(invalid.sum()) > 0:
        warnings.append(
            f"{int(invalid.sum())} contratos con entradas invalidas -> NaN. No se "
            f"han convertido en ceros: un cero se suma sin que nadie lo note."
        )

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        n_total=int(Sa.size),
        n_invalid=int(invalid.sum()),
        n_expired=int(expired.sum()),
        n_floored=int(floored.sum()),
        gamma_share_floored=share,
        warnings=warnings,
    )
