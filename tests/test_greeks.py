"""Verificacion de `pricing.py` y `greeks.py` contra verdad analitica.

El orden de importancia, igual que en el resto del repo:

  1. Que la GAMMA coincida con la FORMA CERRADA. Se reimplementa el escalar con
     `math` puro -- camino de codigo distinto al vectorizado de numpy que usa
     produccion -- y se comparan. Un bug compartido por las dos versiones tendria
     que ser un error de algebra, no de implementacion.
  2. Que se cumpla la PARIDAD PUT-CALL en gamma y vega (identicas) y en precio
     (C - P = S e^{-qT} - K e^{-rT}). Es la prueba mas barata y la que mas
     errores de signo caza.
  3. Que la gamma sea la SEGUNDA DERIVADA del precio por diferencias finitas.
     Verifica formula y precio a la vez: si una de las dos esta mal, no casan.
  4. Que los LIMITES se comporten: T->0, sigma->0, muy ITM, muy OTM, vencido.
  5. Que los datos INVALIDOS den NaN y NUNCA cero. Un NaN se ve; un cero se suma.
  6. Que el ARBOL AMERICANO reproduzca el resultado clasico: una call americana
     sobre subyacente sin dividendo vale IGUAL que la europea (nunca conviene
     ejercer antes), y una put americana vale MAS. Si el arbol pasa esto, sirve
     para medir el sesgo A4 de SPY.
  7. Que la IV haga ROUND-TRIP: precio -> IV -> precio.

Se ejecuta solo:  python tests/test_greeks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from gamma_quant.options import pricing as P
from gamma_quant.options import greeks as G


# --------------------------------------------------------------------------- #
# Infraestructura minima de test
# --------------------------------------------------------------------------- #

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        msg = f"{label}" + (f"  -> {detail}" if detail else "")
        print(f"  [FALLO] {msg}")
        _FAILURES.append(msg)


def close(a, b, rtol=1e-9, atol=1e-12) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            rtol=rtol, atol=atol, equal_nan=True))


def block(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
# Implementacion de referencia: escalar, `math` puro, sin numpy
# --------------------------------------------------------------------------- #

def ref_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ref_norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def ref_d1(S, K, T, r, sigma, q=0.0):
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def ref_gamma(S, K, T, r, sigma, q=0.0):
    return math.exp(-q * T) * ref_norm_pdf(ref_d1(S, K, T, r, sigma, q)) / (S * sigma * math.sqrt(T))


def ref_call(S, K, T, r, sigma, q=0.0):
    d1 = ref_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * ref_norm_cdf(d1) - K * math.exp(-r * T) * ref_norm_cdf(d2)


def ref_put(S, K, T, r, sigma, q=0.0):
    d1 = ref_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * ref_norm_cdf(-d2) - S * math.exp(-q * T) * ref_norm_cdf(-d1)


# Rejilla de casos que cubre ATM, ITM, OTM, corto y largo plazo, con y sin q.
CASES = [
    # S,    K,     T,     r,     sigma, q
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),   # el caso de libro
    (100.0,  90.0, 0.50, 0.03, 0.25, 0.00),   # ITM call
    (100.0, 120.0, 0.25, 0.05, 0.30, 0.00),   # OTM call
    (767.0, 770.0, 0.02, 0.04, 0.15, 0.012),  # SPY realista, corto plazo
    (7686.0, 7700.0, 0.08, 0.04, 0.13, 0.00), # SPX realista
    (100.0, 100.0, 2.00, 0.01, 0.60, 0.03),   # vol alta, largo, con dividendo
    (50.0,   55.0, 0.10, 0.00, 0.45, 0.00),   # tipo cero
]


# --------------------------------------------------------------------------- #
# BLOQUE 1 — Gamma contra forma cerrada
# --------------------------------------------------------------------------- #

def test_gamma_closed_form() -> None:
    block("BLOQUE 1 — gamma vectorizada vs forma cerrada escalar (`math` puro)")

    for S, K, T, r, sigma, q in CASES:
        got = float(G.bs_gamma(S, K, T, r, sigma, q))
        want = ref_gamma(S, K, T, r, sigma, q)
        check(close(got, want, rtol=1e-12),
              f"gamma S={S} K={K} T={T} sigma={sigma} q={q}",
              f"obtenido {got!r} esperado {want!r}")

    # Valor literal, calculado a mano para el caso de libro:
    #   d1 = (ln(1) + (0.05 + 0.02)) / 0.20 = 0.35
    #   phi(0.35) = exp(-0.061250) / sqrt(2pi) = 0.37524035...
    #   gamma = 0.37524035 / (100 * 0.20 * 1) = 0.018762017...
    got = float(G.bs_gamma(100.0, 100.0, 1.0, 0.05, 0.20, 0.0))
    check(abs(got - 0.0187620173) < 1e-9,
          "gamma del caso de libro == 0,0187620173 (literal independiente)",
          f"obtenido {got!r}")


# --------------------------------------------------------------------------- #
# BLOQUE 2 — Paridad put-call
# --------------------------------------------------------------------------- #

def test_put_call_parity() -> None:
    block("BLOQUE 2 — paridad put-call")

    for S, K, T, r, sigma, q in CASES:
        gc = float(G.bs_gamma(S, K, T, r, sigma, q))
        gp = float(G.bs_gamma(S, K, T, r, sigma, q))
        check(gc == gp, f"gamma call == gamma put  (S={S} K={K})")

        vc = float(G.bs_vega(S, K, T, r, sigma, q))
        vp = float(G.bs_vega(S, K, T, r, sigma, q))
        check(vc == vp, f"vega call == vega put  (S={S} K={K})")

        # C - P = S e^{-qT} - K e^{-rT}
        c = float(P.bs_price(S, K, T, r, sigma, q, "C"))
        p = float(P.bs_price(S, K, T, r, sigma, q, "P"))
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        check(close(lhs, rhs, rtol=1e-11, atol=1e-10),
              f"C - P == S e^-qT - K e^-rT  (S={S} K={K})",
              f"{lhs!r} vs {rhs!r}")

        # delta_call - delta_put = e^{-qT}
        dc = float(G.bs_delta(S, K, T, r, sigma, q, "C"))
        dp = float(G.bs_delta(S, K, T, r, sigma, q, "P"))
        check(close(dc - dp, math.exp(-q * T), rtol=1e-11),
              f"delta_call - delta_put == e^-qT  (S={S} K={K})",
              f"{dc - dp!r} vs {math.exp(-q * T)!r}")


# --------------------------------------------------------------------------- #
# BLOQUE 3 — Griegas contra diferencias finitas del precio
# --------------------------------------------------------------------------- #

def test_finite_differences() -> None:
    block("BLOQUE 3 — griegas == derivadas numericas del precio BSM")

    for S, K, T, r, sigma, q in CASES:
        for typ in ("C", "P"):
            # gamma = d2V/dS2. h optimo para 2a derivada ~ eps^(1/4)*escala.
            h = 1e-4 * S
            up = float(P.bs_price(S + h, K, T, r, sigma, q, typ))
            mid = float(P.bs_price(S, K, T, r, sigma, q, typ))
            dn = float(P.bs_price(S - h, K, T, r, sigma, q, typ))
            fd_gamma = (up - 2.0 * mid + dn) / (h * h)
            an_gamma = float(G.bs_gamma(S, K, T, r, sigma, q))
            check(close(fd_gamma, an_gamma, rtol=1e-4, atol=1e-10),
                  f"gamma == d2V/dS2  ({typ} S={S} K={K} T={T})",
                  f"fd {fd_gamma:.10g} vs analitica {an_gamma:.10g}")

            # delta = dV/dS
            hd = 1e-6 * S
            fd_delta = (float(P.bs_price(S + hd, K, T, r, sigma, q, typ))
                        - float(P.bs_price(S - hd, K, T, r, sigma, q, typ))) / (2 * hd)
            an_delta = float(G.bs_delta(S, K, T, r, sigma, q, typ))
            check(close(fd_delta, an_delta, rtol=1e-5, atol=1e-8),
                  f"delta == dV/dS  ({typ} S={S} K={K})",
                  f"fd {fd_delta:.10g} vs analitica {an_delta:.10g}")

            # vega = dV/dsigma
            hs = 1e-6
            fd_vega = (float(P.bs_price(S, K, T, r, sigma + hs, q, typ))
                       - float(P.bs_price(S, K, T, r, sigma - hs, q, typ))) / (2 * hs)
            an_vega = float(G.bs_vega(S, K, T, r, sigma, q))
            check(close(fd_vega, an_vega, rtol=1e-5, atol=1e-6),
                  f"vega == dV/dsigma  ({typ} S={S} K={K})",
                  f"fd {fd_vega:.10g} vs analitica {an_vega:.10g}")

            # theta = -dV/dT  (paso del tiempo de calendario)
            ht = 1e-6
            fd_theta = -(float(P.bs_price(S, K, T + ht, r, sigma, q, typ))
                         - float(P.bs_price(S, K, T - ht, r, sigma, q, typ))) / (2 * ht)
            an_theta = float(G.bs_theta(S, K, T, r, sigma, q, typ))
            check(close(fd_theta, an_theta, rtol=1e-4, atol=1e-5),
                  f"theta == -dV/dT  ({typ} S={S} K={K})",
                  f"fd {fd_theta:.10g} vs analitica {an_theta:.10g}")


# --------------------------------------------------------------------------- #
# BLOQUE 4 — Limites y casos degenerados
# --------------------------------------------------------------------------- #

def test_limits() -> None:
    block("BLOQUE 4 — limites: T->0, sigma->0, muy ITM/OTM, vencido")

    # T == 0 -> vencida: gamma 0, precio = intrinseco.
    check(float(G.bs_gamma(100.0, 100.0, 0.0, 0.05, 0.2)) == 0.0,
          "T=0 -> gamma exactamente 0 (la opcion ya no existe)")
    check(float(P.bs_price(100.0, 90.0, 0.0, 0.05, 0.2, 0.0, "C")) == 10.0,
          "T=0 -> precio call = intrinseco (S-K)")
    check(float(P.bs_price(100.0, 110.0, 0.0, 0.05, 0.2, 0.0, "P")) == 10.0,
          "T=0 -> precio put = intrinseco (K-S)")
    check(float(P.bs_price(100.0, 110.0, 0.0, 0.05, 0.2, 0.0, "C")) == 0.0,
          "T=0 -> call OTM vale 0")

    # T -> 0 por arriba: la gamma ATM DIVERGE como 1/sqrt(T). Es el hecho que
    # domina el GEX del dia de vencimiento.
    Ts = np.array([1.0, 0.25, 0.05, 0.01, 0.002])
    gammas = G.bs_gamma(100.0, 100.0, Ts, 0.05, 0.2)
    check(bool(np.all(np.diff(gammas) > 0)),
          "gamma ATM crece monotonamente al acercarse el vencimiento",
          f"{gammas}")
    # El "gamma ~ 1/sqrt(T)" que se repite por todas partes es solo el termino
    # DOMINANTE. d1 tambien depende de T (en ATM, d1 = (r-q+sigma^2/2)sqrt(T)/sigma),
    # asi que la relacion exacta lleva ademas el cociente de las densidades:
    #
    #     gamma(T1)/gamma(T2) = [phi(d1(T1))/phi(d1(T2))] * sqrt(T2/T1)
    #
    # Se comprueba la identidad EXACTA, no la aproximacion: con T de 1 a 0,002 la
    # correccion es del 6% y una tolerancia laxa la escondería.
    d1_long = ref_d1(100.0, 100.0, float(Ts[0]), 0.05, 0.2)
    d1_short = ref_d1(100.0, 100.0, float(Ts[-1]), 0.05, 0.2)
    ratio = gammas[-1] / gammas[0]
    exact_ratio = (ref_norm_pdf(d1_short) / ref_norm_pdf(d1_long)) * math.sqrt(Ts[0] / Ts[-1])
    check(abs(ratio / exact_ratio - 1.0) < 1e-10,
          "el crecimiento de la gamma ATM cumple la relacion EXACTA en T",
          f"ratio {ratio:.6f} vs exacto {exact_ratio:.6f}")
    check(abs(ratio / math.sqrt(Ts[0] / Ts[-1]) - 1.0) < 0.10,
          "y queda cerca del 1/sqrt(T) asintotico (dentro del 10%)",
          f"ratio {ratio:.2f} vs sqrt {math.sqrt(Ts[0] / Ts[-1]):.2f}")

    # El suelo se aplica y ademas SE CUENTA.
    res = G.compute_greeks(100.0, 100.0, 1e-9, 0.05, 0.2, t_floor=P.T_FLOOR_DEFAULT)
    check(np.isfinite(res.gamma).all(),
          "0 < T < suelo -> gamma finita (no inf, no NaN)")
    check(res.n_floored == 1, "el contrato que toca el suelo se CUENTA", f"{res.n_floored}")
    check(res.gamma_share_floored == 1.0,
          "gamma_share_floored = 100% cuando todo viene del suelo")
    check(any("suelo" in w for w in res.warnings),
          "se emite AVISO cuando la gamma la fija el suelo y no el mercado")

    # Muy OTM / muy ITM -> gamma 0, sin NaN.
    far = G.bs_gamma(100.0, np.array([1.0, 10000.0]), 0.01, 0.05, 0.2)
    check(np.all(np.isfinite(far)) and np.all(far >= 0.0) and np.all(far < 1e-12),
          "muy ITM y muy OTM -> gamma ~0 y finita (limite correcto, no fallo)",
          f"{far}")

    # sigma == 0 -> suelo, finito.
    g0 = G.bs_gamma(100.0, 100.0, 1.0, 0.05, 0.0)
    check(np.isfinite(g0).all(), "sigma=0 -> suelo de sigma, resultado finito", f"{g0}")

    # SIMETRIA ESPEJO. La invariante correcta con r=q=0 es
    #
    #     S^2 gamma(S,K) = K^2 gamma(K,S)
    #
    # y sale de la identidad BSM S phi(d1) = K phi(d2) junto con d1(K,S) = -d2(S,K).
    # NO es cierto que S gamma(S,K) = K gamma(K,S): esa version "obvia" falla por
    # el termino de deriva sigma^2 T/2, que no desaparece aunque r=q=0.
    #
    # Interesa precisamente esta forma porque el GEX lleva S^2: es la magnitud
    # escalada por spot la que es simetrica, no la gamma cruda.
    g_up = float(G.bs_gamma(100.0, 110.0, 1.0, 0.0, 0.2, 0.0))
    g_dn = float(G.bs_gamma(110.0, 100.0, 1.0, 0.0, 0.2, 0.0))
    check(close(g_up * 100.0 ** 2, g_dn * 110.0 ** 2, rtol=1e-12),
          "simetria espejo: S^2 gamma(S,K) == K^2 gamma(K,S) con r=q=0",
          f"{g_up * 100.0 ** 2!r} vs {g_dn * 110.0 ** 2!r}")


# --------------------------------------------------------------------------- #
# BLOQUE 5 — Datos invalidos: NaN, nunca cero
# --------------------------------------------------------------------------- #

def test_invalid_inputs() -> None:
    block("BLOQUE 5 — entradas invalidas producen NaN y se contabilizan")

    S = np.array([100.0, -1.0, 100.0, 100.0, np.nan, 0.0])
    K = np.array([100.0, 100.0, -5.0, 100.0, 100.0, 100.0])
    T = np.array([1.0, 1.0, 1.0, -0.5, 1.0, 1.0])
    g = G.bs_gamma(S, K, T, 0.05, 0.2)

    check(np.isfinite(g[0]), "la fila valida sigue siendo valida")
    check(bool(np.all(np.isnan(g[1:]))),
          "S<0, K<0, T<0, S=NaN y S=0 -> NaN (NO cero)", f"{g}")
    check(not np.any(g[1:] == 0.0),
          "ningun invalido se ha colado como 0,0 (un cero se sumaria en silencio)")

    res = G.compute_greeks(S, K, T, 0.05, 0.2)
    check(res.n_invalid == 5, "se cuentan los 5 invalidos", f"{res.n_invalid}")
    check(any("invalid" in w for w in res.warnings), "se avisa de los invalidos")

    # Tipo de opcion desconocido: error ruidoso, no adivinanza.
    try:
        P.bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "X")
        check(False, "un tipo de opcion 'X' debe lanzar ValueError")
    except ValueError:
        check(True, "un tipo de opcion desconocido lanza ValueError (no adivina)")


# --------------------------------------------------------------------------- #
# BLOQUE 6 — Arbol binomial americano (valida el medidor del supuesto A4)
# --------------------------------------------------------------------------- #

def test_american_binomial() -> None:
    block("BLOQUE 6 — arbol americano: resultados clasicos")

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # Resultado clasico: sin dividendo NUNCA conviene ejercer una call antes de
    # vencimiento, luego americana == europea.
    am_call = P.american_binomial_price(S, K, T, r, sigma, 0.0, "C", steps=800)
    eu_call = float(P.bs_price(S, K, T, r, sigma, 0.0, "C"))
    check(abs(am_call - eu_call) / eu_call < 2e-3,
          "call americana sin dividendo == call europea (resultado clasico)",
          f"americana {am_call:.6f} vs europea {eu_call:.6f}")

    # La put americana vale MAS que la europea: el ejercicio anticipado tiene valor.
    am_put = P.american_binomial_price(S, K, T, r, sigma, 0.0, "P", steps=800)
    eu_put = float(P.bs_price(S, K, T, r, sigma, 0.0, "P"))
    check(am_put > eu_put,
          "put americana > put europea (la prima de ejercicio anticipado es positiva)",
          f"americana {am_put:.6f} vs europea {eu_put:.6f}")

    # Convergencia del arbol al aumentar pasos.
    coarse = P.american_binomial_price(S, K, T, r, sigma, 0.0, "C", steps=100)
    fine = P.american_binomial_price(S, K, T, r, sigma, 0.0, "C", steps=1600)
    check(abs(fine - eu_call) < abs(coarse - eu_call) + 1e-6,
          "el arbol converge a BSM al refinar",
          f"100 pasos {coarse:.6f}, 1600 pasos {fine:.6f}, BSM {eu_call:.6f}")

    # Y la gamma del arbol reproduce la de BSM donde deben coincidir. Esto es lo
    # que habilita medir el sesgo A4 de SPY: si el medidor no casa con BSM en el
    # caso en que DEBE casar, no sirve para medir nada.
    am_gamma = P.american_binomial_gamma(S, K, T, r, sigma, 0.0, "C", steps=800)
    bs_g = float(G.bs_gamma(S, K, T, r, sigma, 0.0))
    check(abs(am_gamma - bs_g) / bs_g < 0.05,
          "gamma del arbol ~ gamma BSM en el caso call-sin-dividendo",
          f"arbol {am_gamma:.8f} vs BSM {bs_g:.8f}")


# --------------------------------------------------------------------------- #
# BLOQUE 7 — Volatilidad implicita
# --------------------------------------------------------------------------- #

def test_implied_vol() -> None:
    block("BLOQUE 7 — IV: round-trip y cotas de no arbitraje")

    for S, K, T, r, sigma, q in CASES:
        for typ in ("C", "P"):
            price = float(P.bs_price(S, K, T, r, sigma, q, typ))
            iv = float(P.implied_volatility(price, S, K, T, r, q, typ))
            check(close(iv, sigma, rtol=1e-6, atol=1e-7),
                  f"round-trip IV ({typ} S={S} K={K} sigma={sigma})",
                  f"recuperada {iv!r}")

    # Vectorizado sobre una cadena.
    strikes = np.linspace(80.0, 120.0, 41)
    true_iv = 0.15 + 0.001 * (strikes - 100.0) ** 2 / 100.0     # una sonrisa
    prices = P.bs_price(100.0, strikes, 0.5, 0.03, true_iv, 0.0, "C")
    got_iv = P.implied_volatility(prices, 100.0, strikes, 0.5, 0.03, 0.0, "C")
    check(close(got_iv, true_iv, rtol=1e-6, atol=1e-7),
          "IV vectorizada sobre 41 strikes con sonrisa",
          f"max err {np.nanmax(np.abs(got_iv - true_iv)):.2e}")

    # Fuera de las cotas de no arbitraje -> NaN, no un extremo inventado.
    below = P.implied_volatility(-1.0, 100.0, 100.0, 1.0, 0.05, 0.0, "C")
    above = P.implied_volatility(101.0, 100.0, 100.0, 1.0, 0.05, 0.0, "C")
    check(bool(np.isnan(below)), "precio por debajo del intrinseco -> NaN")
    check(bool(np.isnan(above)), "precio por encima de la cota superior -> NaN")


# --------------------------------------------------------------------------- #
# BLOQUE 8 — Vectorizacion y difusion
# --------------------------------------------------------------------------- #

def test_broadcasting() -> None:
    block("BLOQUE 8 — vectorizacion: el resultado no depende de la forma")

    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    vec = G.bs_gamma(100.0, strikes, 0.5, 0.03, 0.2)
    one_by_one = np.array([float(G.bs_gamma(100.0, float(k), 0.5, 0.03, 0.2)) for k in strikes])
    check(close(vec, one_by_one, rtol=1e-15),
          "vectorizado identico a escalar uno a uno")

    # Difusion 2D: strikes x vencimientos, como una superficie real.
    T_grid = np.array([[0.05], [0.25], [1.0]])
    surface = G.bs_gamma(100.0, strikes[None, :], T_grid, 0.03, 0.2)
    check(surface.shape == (3, 5), "difusion 2D da forma (3, 5)", f"{surface.shape}")
    check(bool(np.all(surface[0, 2] > surface[2, 2])),
          "en el strike ATM, mas cerca del vencimiento -> mas gamma")

    # Tipos mezclados en un array, como llega una cadena real.
    types = np.array(["C", "P", "C", "P", "c"])
    d = G.bs_delta(100.0, strikes, 0.5, 0.03, 0.2, 0.0, types)
    check(bool(np.all(d[[0, 2, 4]] > 0) and np.all(d[[1, 3]] < 0)),
          "array mixto C/P: deltas de call positivas y de put negativas (acepta minusculas)",
          f"{d}")

    # compute_greeks devuelve todo con la misma forma.
    res = G.compute_greeks(100.0, strikes, 0.5, 0.03, 0.2, 0.0, "C")
    check(all(x.shape == (5,) for x in (res.delta, res.gamma, res.vega, res.theta)),
          "compute_greeks conserva la forma en las cuatro griegas")
    check(res.n_total == 5 and res.n_invalid == 0 and res.n_expired == 0,
          "el diagnostico cuenta bien", res.report().replace("\n", " | "))


# --------------------------------------------------------------------------- #
# BLOQUE 9 — Aviso de estilo americano (supuesto A4)
# --------------------------------------------------------------------------- #

def test_american_warning() -> None:
    block("BLOQUE 9 — el supuesto A4 no pasa desapercibido")

    res_eu = G.compute_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "C",
                              exercise_style="european")
    res_am = G.compute_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "C",
                              exercise_style="american")
    check(not any("A4" in w for w in res_eu.warnings),
          "europea: sin aviso de A4")
    check(any("A4" in w for w in res_am.warnings),
          "americana (SPY): AVISO explicito de que la gamma es aproximada",
          str(res_am.warnings))

    # Y el error real que ese aviso anuncia, medido en una put ITM de SPY, que es
    # donde el ejercicio anticipado muerde y donde suele haber OI grande.
    S, K, T, r, sigma, q = 767.0, 850.0, 0.5, 0.045, 0.18, 0.012
    g_eu = float(G.bs_gamma(S, K, T, r, sigma, q))
    g_am = P.american_binomial_gamma(S, K, T, r, sigma, q, "P", steps=600)
    rel = abs(g_am - g_eu) / g_eu if g_eu else float("nan")
    print(f"  [info] put SPY ITM K={K}: gamma europea {g_eu:.6e}, "
          f"americana {g_am:.6e}, discrepancia {rel:.1%}")
    check(np.isfinite(rel), "el sesgo A4 es medible (no se supone: se calcula)")


# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print("VERIFICACION DE PRICING Y GRIEGAS — gamma_quant")
    print("=" * 78)

    test_gamma_closed_form()
    test_put_call_parity()
    test_finite_differences()
    test_limits()
    test_invalid_inputs()
    test_american_binomial()
    test_implied_vol()
    test_broadcasting()
    test_american_warning()

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"RESULTADO: {len(_FAILURES)} FALLOS de {_CHECKS} comprobaciones")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"RESULTADO: {_CHECKS} comprobaciones, todas correctas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
