"""Arbol binomial Cox-Ross-Rubinstein para opciones americanas sobre acciones.

Por que CRR y no Black-Scholes: OptionMetrics -- la fuente del paper de Soebhag
(2023) -- calcula greeks de opciones americanas con arbol binomial que incorpora
dividendos discretos y ejercicio anticipado. BSM europeo introduce error
sistematico en gamma, concentrado en ITM y en subyacentes con dividendo alto.

Dividendos: modelo escrowed. S_adj = S - PV(dividendos con ex-date <= T). Como
S_adj = S - constante, d(S_adj)/dS = 1 y por tanto gamma_S == gamma_S_adj.

Atajo EXACTO (teorema, no aproximacion): una call americana sobre subyacente sin
dividendos antes del vencimiento nunca se ejerce anticipadamente => su precio es
identico al europeo. Ver use_bsm_shortcut().

NOTA sobre gamma por diferencias finitas: NO usar bump-and-reprice sobre el arbol.
Los nodos se re-cuantizan al mover S, y la segunda diferencia amplifica ese ruido
por 1/h^2. La gamma correcta se lee de los nodos del paso 2 del propio arbol (una
sola construccion) y se valida contra BSM analitico y contra auto-convergencia en N.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

__all__ = ["crr", "crr_vec", "crr_implied_vol", "crr_implied_vol_vec",
           "use_bsm_shortcut", "SIGMA_LO", "SIGMA_HI"]

# Piso de vol: por debajo de ~r*sqrt(dt) la probabilidad riesgo-neutral del arbol
# se sale de [0,1] y el modelo deja de estar definido. Coincide con el filtro de
# calidad de cotizacion del pipeline (IV valida en [1%, 500%]).
SIGMA_LO = 0.01
SIGMA_HI = 5.0


@njit(cache=True, inline="always")
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


@njit(cache=True, inline="always")
def _norm_pdf(x):
    return 0.3989422804014327 * math.exp(-0.5 * x * x)


@njit(cache=True)
def bsm_price_vega(S, K, T, r, sigma, is_call):
    """Black-Scholes europeo (sin dividend yield; los dividendos entran via S_adj)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return np.nan, np.nan
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vt
    d2 = d1 - vt
    dfr = math.exp(-r * T)
    if is_call:
        px = S * _norm_cdf(d1) - K * dfr * _norm_cdf(d2)
    else:
        px = K * dfr * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return px, S * _norm_pdf(d1) * math.sqrt(T)


@njit(cache=True)
def bsm_implied_vol(target, S, K, T, r, is_call):
    """IV europea por Newton con salvaguarda de biseccion. Semilla del refinamiento CRR."""
    if not (target > 0.0) or T <= 0.0 or S <= 0.0:
        return np.nan
    intrinsic = (S - K * math.exp(-r * T)) if is_call else (K * math.exp(-r * T) - S)
    if target < max(intrinsic, 0.0) - 1e-9:
        return np.nan
    sig = max(math.sqrt(2.0 * math.pi / T) * target / S, 0.05)   # semilla Brenner-Subrahmanyam
    sig = min(max(sig, SIGMA_LO), SIGMA_HI)
    for _ in range(30):
        px, vega = bsm_price_vega(S, K, T, r, sig, is_call)
        if np.isnan(px):
            return np.nan
        diff = px - target
        if abs(diff) < 1e-10:
            return sig
        if vega < 1e-12:
            break
        step = diff / vega
        if step > 0.5:
            step = 0.5
        elif step < -0.5:
            step = -0.5
        new = sig - step
        if new <= SIGMA_LO or new >= SIGMA_HI or np.isnan(new):
            break
        if abs(new - sig) < 1e-12:
            return new
        sig = new
    lo, hi = SIGMA_LO, SIGMA_HI
    f_lo, _ = bsm_price_vega(S, K, T, r, lo, is_call)
    f_lo -= target
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid, _ = bsm_price_vega(S, K, T, r, mid, is_call)
        f_mid -= target
        if abs(f_mid) < 1e-12 or (hi - lo) < 1e-12:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=False)
def crr(S, K, T, r, sigma, div_pv, is_call, american, n_steps):
    """Precio, delta y gamma via arbol CRR. Devuelve (precio, delta, gamma)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or n_steps < 3:
        return np.nan, np.nan, np.nan
    s_adj = S - div_pv
    if s_adj <= 0.0:
        return np.nan, np.nan, np.nan

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    u2 = u * u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    if p < 0.0 or p > 1.0:
        return np.nan, np.nan, np.nan
    q = 1.0 - p

    v = np.empty(n_steps + 1)
    s_low = s_adj * d ** n_steps          # un solo pow, no uno por nodo
    s_node = s_low
    for j in range(n_steps + 1):
        v[j] = (s_node - K) if is_call else (K - s_node)
        if v[j] < 0.0:
            v[j] = 0.0
        s_node *= u2

    s_uu = s_ud = s_dd = 0.0
    v_uu = v_ud = v_dd = 0.0
    s_up = s_dn = 0.0
    v_up = v_dn = 0.0

    for i in range(n_steps - 1, -1, -1):
        s_low *= u                         # nodo mas bajo del nivel i
        s_node = s_low
        if american:
            for j in range(i + 1):
                cont = disc * (p * v[j + 1] + q * v[j])
                ex = (s_node - K) if is_call else (K - s_node)
                v[j] = cont if cont > ex else ex
                s_node *= u2
        else:
            for j in range(i + 1):
                v[j] = disc * (p * v[j + 1] + q * v[j])
        if i == 2:
            s_dd = s_low; s_ud = s_low * u2; s_uu = s_low * u2 * u2
            v_dd, v_ud, v_uu = v[0], v[1], v[2]
        elif i == 1:
            s_dn = s_low; s_up = s_low * u2
            v_dn, v_up = v[0], v[1]

    delta = (v_up - v_dn) / (s_up - s_dn)
    delta_up = (v_uu - v_ud) / (s_uu - s_ud)
    delta_dn = (v_ud - v_dd) / (s_ud - s_dd)
    gamma = (delta_up - delta_dn) / (0.5 * (s_uu - s_dd))
    return v[0], delta, gamma


@njit(cache=True, inline="always")
def _crr_px(S, K, T, r, sigma, div_pv, is_call, american, n_steps):
    px, _, _ = crr(S, K, T, r, sigma, div_pv, is_call, american, n_steps)
    return px


@njit(cache=True)
def crr_implied_vol(target, S, K, T, r, div_pv, is_call, american, n_steps):
    """IV sobre el arbol CRR.

    Estrategia hibrida: semilla analitica BSM sobre S_adj (exacta cuando no hay
    ejercicio anticipado) + refinamiento por secante sobre el arbol. Tipicamente
    2-4 construcciones de arbol en vez de las ~60 de una biseccion pura.
    """
    if not (target > 0.0) or T <= 0.0:
        return np.nan
    s_adj = S - div_pv
    if s_adj <= 0.0:
        return np.nan

    sig = bsm_implied_vol(target, s_adj, K, T, r, is_call)
    if np.isnan(sig):
        sig = 0.4
    if not american:
        return sig

    f0 = _crr_px(S, K, T, r, sig, div_pv, is_call, True, n_steps) - target
    if np.isnan(f0):
        return np.nan
    if abs(f0) < 1e-9:
        return sig
    sig1 = sig * (0.97 if f0 > 0.0 else 1.03)
    sig1 = min(max(sig1, SIGMA_LO), SIGMA_HI)
    f1 = _crr_px(S, K, T, r, sig1, div_pv, is_call, True, n_steps) - target

    for _ in range(12):
        if np.isnan(f1):
            return np.nan
        if abs(f1) < 1e-9 or abs(sig1 - sig) < 1e-9:
            return sig1
        den = f1 - f0
        if abs(den) < 1e-14:
            break
        new = sig1 - f1 * (sig1 - sig) / den
        if np.isnan(new) or new <= SIGMA_LO or new >= SIGMA_HI:
            break
        sig, f0 = sig1, f1
        sig1 = new
        f1 = _crr_px(S, K, T, r, sig1, div_pv, is_call, True, n_steps) - target
    else:
        return sig1

    lo, hi = SIGMA_LO, SIGMA_HI                       # salvaguarda: biseccion
    f_lo = _crr_px(S, K, T, r, lo, div_pv, is_call, True, n_steps) - target
    f_hi = _crr_px(S, K, T, r, hi, div_pv, is_call, True, n_steps) - target
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0.0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = _crr_px(S, K, T, r, mid, div_pv, is_call, True, n_steps) - target
        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < 1e-9 or (hi - lo) < 1e-10:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


@njit(parallel=True, cache=True)
def crr_vec(S, K, T, r, sigma, div_pv, is_call, american, n_steps):
    n = S.shape[0]
    px = np.empty(n); dl = np.empty(n); gm = np.empty(n)
    for i in prange(n):
        a, b, c = crr(S[i], K[i], T[i], r[i], sigma[i], div_pv[i],
                      is_call[i], american[i], n_steps)
        px[i] = a; dl[i] = b; gm[i] = c
    return px, dl, gm


@njit(parallel=True, cache=True)
def crr_implied_vol_vec(target, S, K, T, r, div_pv, is_call, american, n_steps):
    n = S.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = crr_implied_vol(target[i], S[i], K[i], T[i], r[i], div_pv[i],
                                 is_call[i], american[i], n_steps)
    return out


def use_bsm_shortcut(is_call, div_pv):
    """True donde American == European de forma EXACTA (call sin dividendos)."""
    return np.asarray(is_call) & (np.asarray(div_pv) <= 0.0)


@njit(cache=True)
def crr_vega_1pt(S, K, T, r, sigma, div_pv, is_call, american, n_steps):
    """Sensibilidad del precio a +1 punto de volatilidad.

    Regla de produccion: si esta sensibilidad es ~0, la IV NO es identificable
    (el precio es insensible a la vol porque el ejercicio anticipado es optimo
    de inmediato y el contrato vale su valor intrinseco). Esos contratos deben
    descartarse -- y no afecta a la Ecuacion 1, porque su gamma es exactamente 0.
    """
    p0 = _crr_px(S, K, T, r, sigma, div_pv, is_call, american, n_steps)
    p1 = _crr_px(S, K, T, r, sigma + 0.01, div_pv, is_call, american, n_steps)
    return p1 - p0


@njit(parallel=True, cache=True)
def crr_vega_1pt_vec(S, K, T, r, sigma, div_pv, is_call, american, n_steps):
    n = S.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = crr_vega_1pt(S[i], K[i], T[i], r[i], sigma[i], div_pv[i],
                              is_call[i], american[i], n_steps)
    return out
