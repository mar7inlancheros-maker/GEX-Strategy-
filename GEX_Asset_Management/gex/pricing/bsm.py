"""Black-Scholes-Merton con dividend yield continuo.

Se usa para: (a) opciones donde el ejercicio anticipado es demostrablemente
irrelevante, (b) semilla de la inversion de IV, (c) control de convergencia
del arbol CRR.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
    return d1, d1 - vol_t


def price(S, K, T, r, sigma, q=0.0, is_call=True):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    call = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    if np.all(is_call):
        return call
    put = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    return np.where(is_call, call, put)


def gamma(S, K, T, r, sigma, q=0.0):
    """Gamma por accion. Identica para call y put (mismo valor analitico)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def gamma_atm_approx(S, T, sigma):
    """Aproximacion ATM: gamma ~= 1 / (S * sigma * sqrt(2*pi*T)).

    Solo para la puerta de validacion P1 (chequeo de orden de magnitud).
    """
    return 1.0 / (S * sigma * np.sqrt(T) * SQRT_2PI)


def implied_vol(target, S, K, T, r, q=0.0, is_call=True,
                lo=1e-3, hi=5.0, tol=1e-10, max_iter=100):
    """IV europea por biseccion + Newton. Devuelve nan si no hay solucion."""
    from scipy.optimize import brentq

    def f(sig):
        return float(price(S, K, T, r, sig, q, is_call)) - float(target)

    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
    except Exception:
        return np.nan
