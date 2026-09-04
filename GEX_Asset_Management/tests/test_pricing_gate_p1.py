"""Puerta de validacion P1 -- motor de valuacion (IV + greeks).

    python3 tests/test_pricing_gate_p1.py

DISENO DE LAS METRICAS (importante, leer antes de tocar los umbrales):

1. Lo que entra en la Ecuacion 1 no es la gamma de un contrato, es la suma
   sum(sign * gamma * OI) sobre toda la cadena. Los errores de oscilacion del
   arbol CRR tienen signo aleatorio entre strikes y se CANCELAN en la suma. Por
   eso la metrica de puerta es el error AGREGADO, no el maximo por contrato.

2. Gamma neta es una diferencia de dos numeros grandes (calls menos puts). La
   razon net/gross medida sobre cadenas sinteticas realistas va de 0.3% a 18%.
   Consecuencia: el error debe normalizarse por gamma BRUTA, no por la neta, o
   la metrica explota justo en la region que importa (Gamma ~ 0 = decil L).

3. Hay contratos donde la IV NO es identificable: put americano profundamente
   ITM cuyo ejercicio inmediato es optimo => precio = intrinseco, insensible a
   la vol. No es un bug del solver. Su gamma es exactamente 0, asi que no
   afectan la Ecuacion 1: la regla de produccion es descartarlos.
"""
from __future__ import annotations

import sys, time, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gex.pricing import bsm
from gex.pricing.crr import (crr, crr_implied_vol, crr_vec, crr_implied_vol_vec,
                             crr_vega_1pt, crr_vega_1pt_vec, use_bsm_shortcut)

N_STEPS = 400
R = 0.042
VEGA_MIN = 0.005              # 1/2 centavo por punto de vol: umbral de identificabilidad
PILOT_CONTRACT_DAYS = 19_000_000   # 30 tickers x ~2.500 contratos x ~250 dias
RESULTS = []


def check(name, value, limit, mode="lt", unit="", info=False):
    ok = bool(value < limit) if mode == "lt" else bool(value > limit)
    RESULTS.append((name, value, limit, mode, ok, unit, info))
    return ok


S_GRID = np.array([25.0, 100.0, 480.0])
MNY = np.array([0.75, 0.90, 0.975, 1.0, 1.025, 1.10, 1.35])
T_GRID = np.array([7 / 365, 30 / 365, 91 / 365, 365 / 365])
VOL_GRID = np.array([0.15, 0.35, 0.80])


def grid(puts=True):
    for S in S_GRID:
        for m in MNY:
            for T in T_GRID:
                for v in VOL_GRID:
                    yield S, S * m, T, v, True
                    if puts:
                        yield S, S * m, T, v, False


# ---- P1.1  CRR europeo vs Black-Scholes analitico
d_px = s_px = d_gm = s_gm = 0.0
rel_gm, max_px_bp = [], 0.0
for S, K, T, v, cp in grid():
    px, _, gm = crr(S, K, T, R, v, 0.0, cp, False, N_STEPS)
    px_bs = float(bsm.price(S, K, T, R, v, 0.0, cp))
    gm_bs = float(bsm.gamma(S, K, T, R, v, 0.0))
    d_px += abs(px - px_bs); s_px += px_bs
    d_gm += abs(gm - gm_bs);  s_gm += gm_bs
    max_px_bp = max(max_px_bp, abs(px - px_bs) / S * 1e4)
    if gm_bs * S > 1e-3:
        rel_gm.append(abs(gm - gm_bs) / gm_bs)
check("P1.1a CRR vs BSM: error agregado de sum(gamma)", d_gm / s_gm, 2e-3)
check("P1.1b CRR vs BSM: error agregado de precio", d_px / s_px, 5e-4)
check("P1.1c CRR vs BSM: error de precio por contrato (max, bp del spot)", max_px_bp, 0, "lt", "bp", True)
check("P1.1d CRR vs BSM: error rel de gamma por contrato, p95", float(np.percentile(rel_gm, 95)), 2e-2)

# ---- P1.2  Teorema: call americana sin dividendos == europea
e = max(abs(crr(S, K, T, R, v, 0.0, True, True, N_STEPS)[0]
            - crr(S, K, T, R, v, 0.0, True, False, N_STEPS)[0])
        for S, K, T, v, _ in grid(puts=False))
check("P1.2  Call americana == europea sin dividendos (error abs max)", e, 1e-12)

# ---- P1.3  Prima de ejercicio anticipado del put
worst, n_pos = 0.0, 0
for S, K, T, v, _ in grid(puts=False):
    pa = crr(S, K, T, R, v, 0.0, False, True, N_STEPS)[0]
    pe = crr(S, K, T, R, v, 0.0, False, False, N_STEPS)[0]
    worst = min(worst, pa - pe); n_pos += (pa - pe) > 1e-6
check("P1.3a Prima de ejercicio del put nunca negativa", abs(worst), 1e-12)
check("P1.3b Casos con prima de ejercicio > 0 (deben existir)", n_pos, 0, "gt", " casos")

# ---- P1.4  Round-trip precio -> IV -> precio, con clasificacion correcta
#
# HALLAZGO: filtrar por vega baja NO equivale a "no importa para la Ecuacion 1".
# Vega ~ sqrt(T) y gamma ~ 1/sqrt(T): las opciones muy cortas ATM tienen vega
# baja Y gamma alta. Descartarlas por vega borraria justo los contratos que mas
# aportan a Gamma. Hay que separar dos causas distintas de vega baja:
#   (a) frontera de ejercicio inmediato (put americano ITM): precio = intrinseco,
#       gamma EXACTAMENTE 0  -> se descarta, es inocuo.
#   (b) vencimiento muy corto: IV mal condicionada pero gamma grande
#       -> se conserva, y su riesgo de precision se mide (P1.12), no se esconde.
VEGA_WELL = 0.05          # bien condicionada: 5 centavos por punto de vol
e_iv_well, e_iv_all, e_rt = [], [], []
n_fail = n_tot = 0
n_exer, exer_gamma_max = 0, 0.0
n_short_ill, short_ill_gamma_max = 0, 0.0
for S, K, T, v, cp in grid():
    for div in (0.0, 0.02 * S):
        px, _, gm = crr(S, K, T, R, v, div, cp, True, N_STEPS)
        if px < 0.005:
            continue
        intrinsic = max(S - div - K, 0.0) if cp else max(K - (S - div), 0.0)
        vg = crr_vega_1pt(S, K, T, R, v, div, cp, True, N_STEPS)
        if px <= intrinsic + 1e-8:                      # caso (a)
            n_exer += 1
            exer_gamma_max = max(exer_gamma_max, abs(gm) * S)
            continue
        if abs(vg) < VEGA_MIN:                          # caso (b)
            n_short_ill += 1
            short_ill_gamma_max = max(short_ill_gamma_max, abs(gm) * S)
        n_tot += 1
        iv = crr_implied_vol(px, S, K, T, R, div, cp, True, N_STEPS)
        if np.isnan(iv):
            n_fail += 1; continue
        e_rt.append(abs(crr(S, K, T, R, iv, div, cp, True, N_STEPS)[0] - px))
        e_iv_all.append(abs(iv - v))
        if abs(vg) >= VEGA_WELL:
            e_iv_well.append(abs(iv - v))
check("P1.4a Round-trip: IV en contratos bien condicionados (max)", max(e_iv_well), 1e-4)
check("P1.4b Round-trip: precio reconstruido (error abs max)", max(e_rt), 1e-6)
check("P1.4c Round-trip: tasa de fallo del solver", n_fail / n_tot, 1e-9)
check("P1.4d Gamma*S maxima en la frontera de ejercicio (pequena, no exacta 0)", exer_gamma_max, 0, "lt", "", True)
check("P1.4e Contratos en frontera de ejercicio (se descartan)", n_exer, 0, "gt", " casos", True)
check("P1.4f Contratos cortos con IV mal condicionada (se conservan)", n_short_ill, 0, "gt", " casos", True)
check("P1.4g Gamma*S maxima entre los mal condicionados (por eso no se descartan)", short_ill_gamma_max, 0, "lt", "", True)
check("P1.4h IV: error max incluyendo mal condicionados", max(e_iv_all), 0, "lt", "", True)

# ---- P1.5  Paridad put-call europea
e = max(abs(crr(S, K, T, R, v, 0.0, True, False, N_STEPS)[0]
            - crr(S, K, T, R, v, 0.0, False, False, N_STEPS)[0]
            - (S - K * np.exp(-R * T)))
        for S, K, T, v, _ in grid(puts=False))
check("P1.5  Paridad put-call europea (error abs max)", e, 1e-8)

# ---- P1.6  Auto-convergencia de gamma en N (americano con dividendos)
rel = []
for S, K, T, v, cp in grid():
    g4 = crr(S, K, T, R, v, 0.02 * S, cp, True, 400)[2]
    g16 = crr(S, K, T, R, v, 0.02 * S, cp, True, 1600)[2]
    if g16 * S > 1e-3:
        rel.append(abs(g4 - g16) / g16)
check("P1.6a Gamma N=400 vs N=1600: mediana", float(np.median(rel)), 5e-3)
check("P1.6b Gamma N=400 vs N=1600: p95", float(np.percentile(rel, 95)), 2e-2)

# ---- P1.7  Gamma ATM vs aproximacion analitica
ratios = [crr(S, S, T, R, v, 0.0, True, False, N_STEPS)[2] / float(bsm.gamma_atm_approx(S, T, v))
          for S in S_GRID for T in T_GRID for v in VOL_GRID]
check("P1.7a Gamma ATM / aproximacion: min", min(ratios), 0.80, "gt")
check("P1.7b Gamma ATM / aproximacion: max", max(ratios), 1.20)

# ---- P1.8  Convergencia monotona
g_ref = crr(100.0, 105.0, 0.25, R, 0.30, 2.0, False, True, 1600)[2]
errs = [abs(crr(100.0, 105.0, 0.25, R, 0.30, 2.0, False, True, n)[2] - g_ref) / g_ref
        for n in (50, 100, 200, 400, 800)]
check("P1.8a Error de gamma decrece de N=50 a N=800", errs[0] - errs[-1], 0.0, "gt")
check("P1.8b Error de gamma N=400 vs referencia N=1600", errs[3], 5e-3)

# ---- P1.9  Rendimiento (umbrales derivados del tamano del piloto, no inventados)
n = 200_000
rng = np.random.default_rng(0)
S = rng.uniform(20, 500, n)
K = S * rng.uniform(0.7, 1.4, n)
T = rng.uniform(5 / 365, 1.5, n)
r = np.full(n, R)
sig = rng.uniform(0.15, 0.9, n)
div = np.where(rng.random(n) < 0.4, 0.0, S * rng.uniform(0.002, 0.02, n))  # ~40% sin dividendo
cp = rng.random(n) < 0.5
am = np.ones(n, dtype=np.bool_)
crr_vec(S[:99], K[:99], T[:99], r[:99], sig[:99], div[:99], cp[:99], am[:99], N_STEPS)
t0 = time.time(); px, dl, gm = crr_vec(S, K, T, r, sig, div, cp, am, N_STEPS)
rate_g = n / (time.time() - t0)
crr_implied_vol_vec(px[:99], S[:99], K[:99], T[:99], r[:99], div[:99], cp[:99], am[:99], N_STEPS)
t0 = time.time(); iv = crr_implied_vol_vec(px, S, K, T, r, div, cp, am, N_STEPS)
rate_iv = n / (time.time() - t0)
vg = crr_vega_1pt_vec(S, K, T, r, sig, div, cp, am, N_STEPS)
ident = np.abs(vg) >= VEGA_MIN
ok_iv = (~np.isnan(iv)) & ident
# umbral: el piloto completo debe procesarse en < 2 horas
min_rate = PILOT_CONTRACT_DAYS / 7200.0
check("P1.9a Rendimiento greeks (contratos/s)", rate_g, min_rate, "gt", "/s")
check("P1.9b Rendimiento inversion de IV (contratos/s)", rate_iv, min_rate, "gt", "/s")
check("P1.9c Tasa de exito de IV en contratos identificables", float(((~np.isnan(iv))[ident]).mean()), 0.995, "gt")
well = (~np.isnan(iv)) & (np.abs(vg) >= 0.05)
px_rt, _, _ = crr_vec(S, K, T, r, np.where(np.isnan(iv), sig, iv), div, cp, am, N_STEPS)
ok_rt = ~np.isnan(iv) & ~np.isnan(px_rt)
check("P1.9d Round-trip de PRECIO en lote grande (max abs)", float(np.max(np.abs(px_rt[ok_rt] - px[ok_rt]))), 1e-6)
check("P1.9h IV en lote grande, bien condicionados (max)", float(np.max(np.abs(iv[well] - sig[well]))), 1e-2)
check("P1.9g IV en lote grande, incluyendo mal condicionados (max)", float(np.max(np.abs(iv[ok_iv] - sig[ok_iv]))), 0, "lt", "", True)
check("P1.9e Horas para procesar el piloto completo", PILOT_CONTRACT_DAYS / rate_iv / 3600, 2.0, "lt", " h")
check("P1.9f Fraccion del lote con atajo EXACTO (call sin dividendos)", float(use_bsm_shortcut(cp, div).mean()), 0.05, "gt")

# ---- P1.10  Error si se usara BSM en vez de CRR (justifica el arbol)
lnmy = np.log(S / K)
gm_bsm = np.asarray(bsm.gamma(S, K, T, r, sig, 0.0))
pd_ = (~cp) & (div > 0)
for lab, msk in (("put ITM", pd_ & (lnmy < -0.1)), ("put ATM", pd_ & (np.abs(lnmy) <= 0.1)),
                 ("put OTM", pd_ & (lnmy > 0.1))):
    if msk.sum() > 100:
        check(f"P1.10 Error de sum(gamma) usando BSM en vez de CRR: {lab}",
              abs(gm[msk].sum() - gm_bsm[msk].sum()) / gm_bsm[msk].sum(), 0, "lt", "", True)

# ---- P1.11  Error de Gamma agregada a nivel de CADENA COMPLETA
def chain(S, dy, iv_atm, seed):
    rg = np.random.default_rng(seed); rows = []
    for T in np.array([7, 14, 21, 30, 45, 60, 91, 182]) / 365.0:
        w = max(0.12, 2.5 * iv_atm * np.sqrt(T))
        st = 1.0 if S < 50 else 2.5
        for K in np.unique(np.round(S * np.exp(np.linspace(-w, w, 40)) / st) * st):
            if K <= 0: continue
            m = np.log(S / K); iv = min(max(iv_atm * (1 + .6 * m ** 2 - .25 * m), .05), 2.5)
            base = np.exp(-(m / .18) ** 2) * np.exp(-T * 1.2)
            for c in (True, False):
                sk = 1.25 if (c and m < 0) or ((not c) and m > 0) else .8
                oi = base * sk * rg.lognormal(6.5, 1.0)
                if oi < 1: continue
                rows.append((S, float(K), float(T), float(iv), c, float(round(oi)),
                             S * dy * T if dy > 0 else 0.0))
    return rows

def agg(rows, n):
    net = gross = 0.0
    for S, K, T, iv, c, oi, dv in rows:
        _, _, g = crr(S, K, T, R, iv, dv, c, True, n)
        if np.isnan(g): continue
        net += g * oi if c else -g * oi; gross += abs(g * oi)
    return net, gross

worst_gross, ratios_ng = 0.0, []
for nm, S, dy, iv, sd in [("AAPL", 235., .005, .26, 1), ("NVDA", 180., 0., .45, 2),
                          ("KO", 68., .030, .18, 3), ("GME", 24., 0., .90, 4),
                          ("MSTR", 320., 0., 1.10, 5)]:
    rows = chain(S, dy, iv, sd)
    n4, g4 = agg(rows, 400); n16, g16 = agg(rows, 1600)
    worst_gross = max(worst_gross, abs(n4 - n16) / g16)
    ratios_ng.append(abs(n16) / g16)
def agg_drop_unident(rows, n=N_STEPS):
    """Gamma neta y bruta descartando contratos en frontera de ejercicio (IV no identificable)."""
    net = gross = net_all = gross_all = 0.0
    for S, K, T, iv, c, oi, dv in rows:
        px, _, g = crr(S, K, T, R, iv, dv, c, True, n)
        if np.isnan(g): continue
        net_all += g * oi if c else -g * oi
        gross_all += abs(g * oi)
        intr = max(S - dv - K, 0.0) if c else max(K - (S - dv), 0.0)
        if px <= intr + 1e-8:
            continue
        net += g * oi if c else -g * oi
        gross += abs(g * oi)
    return net, gross, net_all, gross_all

lost = []
for nm, S, dy, iv0, sd in [("AAPL", 235., .005, .26, 1), ("KO", 68., .030, .18, 3),
                           ("GME", 24., 0., .90, 4)]:
    rows = chain(S, dy, iv0, sd)
    net, gross, net_all, gross_all = agg_drop_unident(rows)
    lost.append(abs(net - net_all) / gross_all)
check("P1.11d Gamma perdida al descartar IV no identificable (norm. por bruta)", float(np.max(lost)), 5e-3)
check("P1.11a Error de Gamma de cadena, normalizado por gamma BRUTA (max)", worst_gross, 1e-3)
check("P1.11b Razon de cancelacion net/gross: minima observada", min(ratios_ng), 0, "lt", "", True)
check("P1.11c Razon de cancelacion net/gross: maxima observada", max(ratios_ng), 0, "lt", "", True)

# ---- P1.12  Sensibilidad de Gamma al ruido en los datos de entrada
#  Con net/gross tan bajo, un error pequeno en la IV o el OI se amplifica en Gamma.
#  Esto dice cuanta precision de DATOS hace falta, que importa mas que el metodo numerico.
def gamma_net(rows, iv_shift=None, oi_mult=None, n=N_STEPS):
    net = 0.0
    for i, (S, K, T, iv, c, oi, dv) in enumerate(rows):
        if iv_shift is not None:
            iv = max(0.02, iv + iv_shift[i])
        if oi_mult is not None:
            oi = oi * oi_mult[i]
        _, _, g = crr(S, K, T, R, iv, dv, c, True, n)
        if np.isnan(g): continue
        net += g * oi if c else -g * oi
    return net

rg = np.random.default_rng(11)
amp_iv, amp_oi = [], []
for nm, S, dy, iv0, sd in [("AAPL", 235., .005, .26, 1), ("NVDA", 180., 0., .45, 2),
                           ("KO", 68., .030, .18, 3), ("GME", 24., 0., .90, 4)]:
    rows = chain(S, dy, iv0, sd)
    base = gamma_net(rows)
    d_iv = [abs(gamma_net(rows, iv_shift=rg.normal(0, 0.005, len(rows))) - base) / abs(base)
            for _ in range(3)]
    d_oi = [abs(gamma_net(rows, oi_mult=rg.normal(1.0, 0.01, len(rows))) - base) / abs(base)
            for _ in range(3)]
    amp_iv.append(float(np.mean(d_iv))); amp_oi.append(float(np.mean(d_oi)))
check("P1.12a Cambio en Gamma ante ruido de +/-0.5 pt de vol en la IV (medio)", float(np.mean(amp_iv)), 0, "lt", "", True)
check("P1.12b Cambio en Gamma ante ruido de +/-1% en el OI (medio)", float(np.mean(amp_oi)), 0, "lt", "", True)
check("P1.12c Cambio en Gamma ante ruido en la IV (peor caso)", float(np.max(amp_iv)), 0, "lt", "", True)

# ------------------------------------------------------------------ reporte
print("\n" + "=" * 96)
print("PUERTA DE VALIDACION P1 -- MOTOR DE VALUACION".center(96))
print("=" * 96)
n_ok = n_gate = 0
for name, val, lim, mode, ok, unit, info in RESULTS:
    if not info:
        n_gate += 1; n_ok += ok
    tag = "info " if info else ("PASS " if ok else "FALLA")
    fv = f"{val:,.0f}{unit}" if abs(val) >= 1000 else f"{val:.3e}{unit}"
    line = f"[{tag}] {name:<68} {fv:>13}"
    if not info:
        line += f"  {'<' if mode == 'lt' else '>'} " + (f"{lim:,.0f}{unit}" if abs(lim) >= 1000 else f"{lim:.1e}{unit}")
    print(line)
print("-" * 96)
print(f"{n_ok}/{n_gate} chequeos de puerta superados   (N_STEPS={N_STEPS})")
print("=" * 96)
sys.exit(0 if n_ok == n_gate else 1)
