"""Black-Scholes-Merton, volatilidad implicita y arbol binomial americano.

QUE HAY AQUI Y POR QUE
----------------------
1. `bs_price` / `d1_d2`  : la forma cerrada europea, vectorizada.
2. `implied_volatility`  : biseccion vectorizada (ver mas abajo por que no Newton).
3. `american_binomial_*` : arbol CRR. No esta para valorar: esta para MEDIR el
                           supuesto A4, es decir cuanto se equivoca la gamma
                           europea cuando se aplica a SPY, que es americana.

EL LIMITE T -> 0 ES EL PROBLEMA CENTRAL
---------------------------------------
La gamma de una opcion ATM diverge al acercarse el vencimiento:

    gamma_ATM ~ 1 / (S * sigma * sqrt(T))    ->  infinito cuando T -> 0

No es una curiosidad matematica. El 0DTE es una categoria de investigacion
explicita de este proyecto, y esta divergencia significa que el DIA DE
VENCIMIENTO el GEX total esta dominado por un puñado de contratos ATM. El numero
que salga depende brutalmente de como se trate ese limite.

Politica adoptada, explicita y configurable:

    T <  0            -> NaN (dato corrupto, no se inventa nada)
    T == 0            -> la opcion ha vencido: gamma 0, precio = valor intrinseco
    0 < T < T_FLOOR   -> se aplica el suelo Y SE CUENTA cuantos contratos lo tocan

Lo importante es la segunda mitad: el suelo se aplica pero NO en silencio. Quien
agregue GEX recibe el recuento de contratos afectados, porque un total en el que
el 30% de la gamma viene de contratos que tocaron el suelo no es un total, es un
artefacto del suelo. La sensibilidad a `T_FLOOR` se reporta como cualquier otro
parametro (PROJECT_PLAN seccion 25).

El suelo por defecto es una hora. Con datos intradia el `T` correcto se calcula
con horas reales hasta el vencimiento y casi nunca se toca el suelo; con datos
diarios, el dia de vencimiento cae de lleno en el.

POR QUE BISECCION Y NO NEWTON PARA LA IV
----------------------------------------
Newton-Raphson usa vega en el denominador. Vega tiende a cero en dos sitios: muy
OTM y cerca del vencimiento. Son exactamente las dos zonas donde MAS contratos
hay en una cadena de SPX. Newton ahi no converge lento: diverge y devuelve
basura plausible. La biseccion converge siempre, es vectorizable sin bucles de
Python y con 100 iteraciones sobre [1e-6, 5.0] deja un error < 1e-7. En una
cadena de 28.000 filas cuesta milisegundos.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Una hora, en años. Ver la discusion del limite T -> 0 arriba.
T_FLOOR_DEFAULT: float = 1.0 / (365.0 * 24.0)

# Volatilidad minima. Una IV de cero hace estallar d1 igual que T=0, y aparece en
# datos reales como valor centinela cuando el proveedor no sabe calcularla.
SIGMA_FLOOR_DEFAULT: float = 1e-4

OptionType = Literal["C", "P"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Normal estandar
# --------------------------------------------------------------------------- #

def _norm_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Densidad normal estandar. Explicita: es mas rapida que scipy y trivial."""
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


try:  # scipy esta declarado como dependencia, pero el modulo no debe morir sin el
    from scipy.special import ndtr as _ndtr

    def _norm_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _ndtr(x)

except ImportError:  # pragma: no cover - camino de respaldo
    _erf_vec = np.vectorize(math.erf, otypes=[np.float64])

    def _norm_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * (1.0 + _erf_vec(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Saneado de entradas
# --------------------------------------------------------------------------- #

def _prepare(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    *,
    t_floor: float,
    sigma_floor: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_],
]:
    """Difunde a arrays, aplica suelos y devuelve las mascaras de diagnostico.

    Devuelve (S, K, T, r, sigma, q, invalido, vencido, suelo_tocado).

    Ningun dato malo se convierte en un numero plausible: `invalido` se propaga
    como NaN. Un cero silencioso en una gamma es peor que un NaN, porque un NaN
    se ve y un cero se suma.
    """
    S, K, T, r, sigma, q = (
        np.asarray(x, dtype=np.float64) for x in (S, K, T, r, sigma, q)
    )
    S, K, T, r, sigma, q = np.broadcast_arrays(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = (np.array(x, dtype=np.float64, copy=True)
                            for x in (S, K, T, r, sigma, q))

    invalid = (
        ~np.isfinite(S) | ~np.isfinite(K) | ~np.isfinite(T)
        | ~np.isfinite(r) | ~np.isfinite(sigma) | ~np.isfinite(q)
        | (S <= 0.0) | (K <= 0.0) | (T < 0.0) | (sigma < 0.0)
    )
    expired = (T == 0.0) & ~invalid

    live = ~invalid & ~expired
    floored_t = live & (T < t_floor)
    floored_sigma = live & (sigma < sigma_floor)
    floored = floored_t | floored_sigma

    T = np.where(floored_t, t_floor, T)
    sigma = np.where(floored_sigma, sigma_floor, sigma)

    return S, K, T, r, sigma, q, invalid, expired, floored


# --------------------------------------------------------------------------- #
# d1 / d2
# --------------------------------------------------------------------------- #

def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    t_floor: float = T_FLOOR_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d1 y d2 de Black-Scholes-Merton con rendimiento por dividendo continuo.

        d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)

    Devuelve NaN donde la entrada es invalida o la opcion ha vencido.
    """
    S, K, T, r, sigma, q, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    sqrt_T = np.sqrt(T)
    denom = sigma * sqrt_T

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * np.square(sigma)) * T) / denom
    d2 = d1 - denom

    bad = invalid | expired
    d1 = np.where(bad, np.nan, d1)
    d2 = np.where(bad, np.nan, d2)
    return d1, d2


# --------------------------------------------------------------------------- #
# Precio europeo
# --------------------------------------------------------------------------- #

def bs_price(
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
    """Precio Black-Scholes-Merton europeo.

        call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    En T == 0 devuelve el valor intrinseco, que es el limite correcto y no una
    convencion: max(S-K, 0) para la call.
    """
    Sa, Ka, Ta, ra, sa, qa, invalid, expired, _ = _prepare(
        S, K, T, r, sigma, q, t_floor=t_floor, sigma_floor=sigma_floor
    )
    is_call = _is_call_mask(option_type, Sa.shape)

    d1, d2 = d1_d2(Sa, Ka, Ta, ra, sa, qa, t_floor=t_floor, sigma_floor=sigma_floor)
    disc_r = np.exp(-ra * Ta)
    disc_q = np.exp(-qa * Ta)

    call = Sa * disc_q * _norm_cdf(d1) - Ka * disc_r * _norm_cdf(d2)
    put = Ka * disc_r * _norm_cdf(-d2) - Sa * disc_q * _norm_cdf(-d1)
    price = np.where(is_call, call, put)

    intrinsic = np.where(is_call, np.maximum(Sa - Ka, 0.0), np.maximum(Ka - Sa, 0.0))
    price = np.where(expired, intrinsic, price)
    price = np.where(invalid, np.nan, price)
    return price


def _is_call_mask(option_type: ArrayLike | OptionType, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """Normaliza el tipo de opcion a una mascara booleana difundida.

    Acepta 'C'/'P', 'call'/'put', mayusculas o minusculas, escalar o array.
    Cualquier otra cosa es un error ruidoso: adivinar el tipo de una opcion es
    exactamente la clase de silencio que invierte el signo de un GEX entero.
    """
    arr = np.asarray(option_type)
    if arr.dtype.kind in ("U", "S", "O"):
        flat = np.char.upper(arr.astype(str))
        is_call = np.isin(flat, ("C", "CALL"))
        is_put = np.isin(flat, ("P", "PUT"))
        unknown = ~(is_call | is_put)
        if np.any(unknown):
            bad = np.unique(flat[unknown])[:5]
            raise ValueError(f"tipo de opcion no reconocido: {list(bad)}; se espera C/P")
    elif arr.dtype.kind == "b":
        is_call = arr
    else:
        raise TypeError(f"option_type debe ser 'C'/'P' o booleano, no {arr.dtype}")
    return np.broadcast_to(is_call, shape)


# --------------------------------------------------------------------------- #
# Volatilidad implicita
# --------------------------------------------------------------------------- #

def implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: ArrayLike | OptionType = "C",
    *,
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    iterations: int = 100,
    t_floor: float = T_FLOOR_DEFAULT,
) -> NDArray[np.float64]:
    """IV por biseccion vectorizada. Devuelve NaN donde no hay solucion.

    Se usa para AUDITAR la IV del proveedor (supuesto A7): se re-resuelve desde
    el mid y se compara. Discrepancias grandes senalan mid basura, no IV mala.

    Sin solucion significa: precio fuera de las cotas de no arbitraje. Con
    `price` por debajo del intrinseco o por encima del maximo teorico no existe
    sigma que lo reproduzca, y devolver el extremo del intervalo seria inventar
    un numero. Devuelve NaN.

    100 iteraciones sobre [1e-6, 5.0] dejan un intervalo de 5e-6/2^100; el limite
    real es la precision de doble, no el numero de pasos.
    """
    price = np.asarray(price, dtype=np.float64)
    Sa, Ka, Ta, ra, _, qa, invalid, expired, _ = _prepare(
        S, K, T, r, 0.2, q, t_floor=t_floor, sigma_floor=SIGMA_FLOOR_DEFAULT
    )
    price, Sa, Ka, Ta, ra, qa = np.broadcast_arrays(price, Sa, Ka, Ta, ra, qa)
    is_call = _is_call_mask(option_type, Sa.shape)

    disc_r = np.exp(-ra * Ta)
    disc_q = np.exp(-qa * Ta)
    intrinsic = np.where(
        is_call,
        np.maximum(Sa * disc_q - Ka * disc_r, 0.0),
        np.maximum(Ka * disc_r - Sa * disc_q, 0.0),
    )
    upper_bound = np.where(is_call, Sa * disc_q, Ka * disc_r)

    unsolvable = (
        invalid | expired
        | ~np.isfinite(price)
        | (price < intrinsic - 1e-10)
        | (price > upper_bound + 1e-10)
    )

    lo = np.full(Sa.shape, sigma_low, dtype=np.float64)
    hi = np.full(Sa.shape, sigma_high, dtype=np.float64)

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        val = bs_price(Sa, Ka, Ta, ra, mid, qa, is_call, t_floor=t_floor)
        too_low = val < price          # el precio BSM se queda corto -> subir sigma
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)

    iv = 0.5 * (lo + hi)
    return np.where(unsolvable, np.nan, iv)


# --------------------------------------------------------------------------- #
# Arbol binomial americano — para MEDIR el supuesto A4
# --------------------------------------------------------------------------- #

def _crr_backward(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    is_call: bool,
    steps: int,
    stop_step: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Induccion hacia atras en un arbol CRR, parando en `stop_step`.

    Devuelve (precios_del_nodo, valores_del_nodo) en ese paso, o None si los
    parametros no forman un arbol valido.

    CRR: u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q)dt} - d)/(u - d).
    Como u*d = 1, en los pasos pares el nodo central cae exactamente en S, que es
    lo que hace utilizable el estimador de gamma del paso 2.
    """
    if not (S > 0 and K > 0 and sigma > 0 and steps >= 1 and T > 0):
        return None
    if stop_step < 0 or stop_step > steps:
        return None

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        # dt demasiado grande para esta sigma: el arbol deja de ser una medida de
        # probabilidad. Mejor None que un precio con probabilidad negativa.
        return None

    j = np.arange(steps + 1, dtype=np.float64)
    prices = S * np.power(u, 2.0 * j - steps)
    values = np.maximum(prices - K, 0.0) if is_call else np.maximum(K - prices, 0.0)

    for step in range(steps - 1, stop_step - 1, -1):
        j = np.arange(step + 1, dtype=np.float64)
        prices = S * np.power(u, 2.0 * j - step)
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        exercise = (prices - K) if is_call else (K - prices)
        values = np.maximum(values, exercise)   # <- el ejercicio anticipado

    return prices, values


def american_binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "C",
    *,
    steps: int = 512,
) -> float:
    """Precio americano por arbol Cox-Ross-Rubinstein. Escalar, no vectorizado.

    No esta para producir precios en masa: esta para cuantificar cuanto se desvia
    la gamma europea en SPY (supuesto A4). Se llama sobre una rejilla pequeña de
    casos representativos, no sobre la cadena entera.
    """
    is_call = option_type.upper().startswith("C")
    if T <= 0:
        if not (S > 0 and K > 0):
            return float("nan")
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    out = _crr_backward(S, K, T, r, sigma, q, is_call, steps, stop_step=0)
    return float("nan") if out is None else float(out[1][0])


def american_binomial_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "C",
    *,
    steps: int = 512,
) -> float:
    """Gamma americana con el estimador de tres nodos DEL PROPIO ARBOL.

    POR QUE NO SE BOMBEA EL SPOT
    ----------------------------
    Lo natural seria gamma ~ [V(S+h) - 2V(S) + V(S-h)] / h^2 llamando tres veces
    al arbol. NO FUNCIONA, y falla de una forma que engaña: al mover S se
    reconstruye la rejilla ENTERA, de modo que los nodos terminales caen en
    posiciones distintas respecto a K. El precio del arbol como funcion de S no
    es suave, sino escalonado con saltos del tamaño del espaciado de la rejilla.
    La segunda diferencia con h pequeño amplifica ese escalon por 1/h^2 y devuelve
    un numero enorme, finito y de aspecto razonable. En pruebas daba 0,265 frente
    a los 0,0188 correctos: un factor 14 sin ningun sintoma visible.

    LO QUE SE HACE EN SU LUGAR
    --------------------------
    El arbol ya contiene la informacion. En el paso 2 hay tres nodos, y como
    u*d = 1 el central es S exactamente:

        S_uu = S u^2      V_uu
        S_ud = S          V_ud
        S_dd = S d^2      V_dd

        delta_sup = (V_uu - V_ud) / (S_uu - S_ud)
        delta_inf = (V_ud - V_dd) / (S_ud - S_dd)
        gamma     = (delta_sup - delta_inf) / ((S_uu - S_dd) / 2)

    Es el estimador CRR estandar: usa la curvatura que el arbol ya calculo, sin
    perturbar la rejilla. El desfase temporal es de 2*dt (con 800 pasos y T=1,
    0,0025 años) y es despreciable frente al sesgo que evita.
    """
    is_call = option_type.upper().startswith("C")
    if T <= 0:
        return 0.0 if (S > 0 and K > 0) else float("nan")
    if steps < 2:
        return float("nan")

    out = _crr_backward(S, K, T, r, sigma, q, is_call, steps, stop_step=2)
    if out is None:
        return float("nan")
    prices, values = out
    if prices.size != 3:
        return float("nan")

    s_dd, s_ud, s_uu = prices[0], prices[1], prices[2]
    v_dd, v_ud, v_uu = values[0], values[1], values[2]

    delta_up = (v_uu - v_ud) / (s_uu - s_ud)
    delta_dn = (v_ud - v_dd) / (s_ud - s_dd)
    return float((delta_up - delta_dn) / (0.5 * (s_uu - s_dd)))
