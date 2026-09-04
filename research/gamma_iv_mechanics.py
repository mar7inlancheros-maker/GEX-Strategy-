"""Por que el cubo OTM sale con el signo contrario: no es un bug, es la formula.

EL CABO SUELTO
--------------
(Este script CRUZA las dos lineas de trabajo del repo: usa el motor de
`gamma_quant/` para explicar una anomalia que salio del subproyecto
`GEX_Asset_Management/`. Es el unico sitio donde se tocan, y por eso vive
en `research/` de la raiz, que es quien puede importar el motor.)

`GEX_Asset_Management/reports/ortogonal.txt` deja abierta una anomalia:

    9.  Gamma slow (>31d)      coef -3.83   t = -6.52
    11. Gamma ATM              coef -1.77   t = -3.63
    12. Gamma OTM              coef +7.30   t = +4.24   <- signo CONTRARIO
    13. Gamma slow ORTOGONAL   coef -1.32   t = -2.66   <- el unico superviviente

y concluye, con razon, que "sugiere un problema en la descomposicion". El problema
es que la MISMA descomposicion produce el unico resultado que sobrevive a la
ortogonalizacion, asi que si esta rota, el superviviente es sospechoso.

LA HIPOTESIS QUE SE CONTRASTA AQUI
-----------------------------------
No hay bug. El signo se invierte porque LA DEPENDENCIA DE LA GAMMA EN SIGMA SE
INVIERTE, y lo hace en un punto que se conoce en forma cerrada. La elasticidad es

    d ln(Gamma) / d ln(sigma)  =  d1*d2 - 1

    ATM        : d1*d2 ~ 0    ->  elasticidad ~ -1   (el clasico Gamma ~ 1/sigma)
    OTM lejano : d1*d2 > 1    ->  elasticidad > 0    (la gamma SUBE con la vol)

El cruce esta en d1*d2 = 1, NO en d1*d2 = 0.

  NOTA SOBRE UN ERROR PROPIO, que se conserva porque explica el diseño del test:
  la primera version de este script afirmaba `dGamma/dsigma = Gamma*d1*d2/sigma`,
  sin el -1, y situaba el cruce en d1*d2 = 0. Es falso. Lo cazo la comprobacion
  numerica del bloque 1, que existe precisamente para no fiarse de una formula
  escrita de memoria. La derivacion correcta:

      ln Gamma = -d1^2/2 - ln(sigma) - ln(S sqrt(T) sqrt(2pi)) - qT
      d(d1)/d(sigma) = sqrt(T) - d1/sigma
      d ln Gamma/d sigma = -d1(sqrt(T) - d1/sigma) - 1/sigma = (d1*d2 - 1)/sigma

  La diferencia importa: con el cruce en 0 la banda "gamma baja con la vol" mide
  sigma^2*T (~0,02, mas estrecha que la banda ATM del codigo); con el cruce en 1
  mide |ln(S/K)| < ~0,15, mas ANCHA que la banda ATM de 0,10. La conclusion
  cualitativa aguanta -- el cubo OTM cae mayoritariamente al otro lado del cruce --
  pero por poco, y eso cambia cuanta confianza merece.

POR QUE ESTO DECIDE ALGO
-------------------------
Si el efecto documentado fuese un CANAL ECONOMICO de cobertura, el signo tendria
que ser EL MISMO en los dos cubos: al dealer le da igual donde este la gamma, la
cubre igual. Que el signo se invierta justo donde se invierte la mecanica de la
formula es evidencia directa de que lo que se mide es la formula.

Y tiene una consecuencia sobre el superviviente: si la dependencia Gamma-sigma es
NO LINEAL y cambia de signo segun moneyness y plazo, entonces ortogonalizar
LINEALMENTE contra UNA sola IV (la mediana de la cadena) no puede limpiarla. El
residuo conserva dependencia mecanica, y `slow` sobreviviendo con t = -2,66 es
exactamente lo que se esperaria de una limpieza incompleta.

    python research/gamma_iv_mechanics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from gamma_quant.options.pricing import d1_d2
from gamma_quant.options.greeks import bs_gamma

# Convenciones del proyecto que se audita (gex/signal/gamma_exposure.py)
ATM_BAND = 0.10          # |ln(S/K)| < 0.10
FAST_MAX_T = 31.0 / 365.0

R = 0.041                # r mediana medida en ese proyecto
Q = 0.0


def bucket(ln_m: float, is_call: bool) -> str:
    """Misma clasificacion que `gamma_exposure.aggregate`."""
    if abs(ln_m) < ATM_BAND:
        return "ATM"
    if (is_call and ln_m < -ATM_BAND) or ((not is_call) and ln_m > ATM_BAND):
        return "OTM"
    return "ITM"


def crossover_moneyness(sigma: float, T: float, *, grid: int = 4001) -> tuple[float, float]:
    """Donde cruza d1*d2 = 1, que es donde la gamma pasa de bajar a subir con la vol.

    No tiene forma cerrada limpia (es cuadratica en ln(S/K) con terminos cruzados),
    asi que se localiza numericamente sobre una rejilla fina. Devuelve (inferior,
    superior) en ln(S/K); entre ambos, la gamma BAJA cuando sube la vol.
    """
    S = 100.0
    xs = np.linspace(-0.8, 0.8, grid)
    Ks = S / np.exp(xs)
    d1, d2 = d1_d2(S, Ks, T, R, sigma, Q)
    f = np.asarray(d1) * np.asarray(d2) - 1.0
    neg = np.where(f < 0)[0]
    if neg.size == 0:
        return float("nan"), float("nan")
    return float(xs[neg[0]]), float(xs[neg[-1]])


def elasticity(S: float, K: float, T: float, sigma: float, *, bump: float = 0.01) -> float:
    """d ln(Gamma) / d ln(sigma), por diferencias centradas relativas."""
    up = float(bs_gamma(S, K, T, R, sigma * (1 + bump), Q))
    dn = float(bs_gamma(S, K, T, R, sigma * (1 - bump), Q))
    if up <= 0 or dn <= 0:
        return float("nan")
    return (np.log(up) - np.log(dn)) / (np.log(1 + bump) - np.log(1 - bump))


def main() -> int:
    print("=" * 84)
    print("MECANICA GAMMA-SIGMA: por que el cubo OTM invierte el signo")
    print("=" * 84)

    S = 100.0

    # ---------------------------------------------------------------- #
    print("\n1. LA IDENTIDAD ANALITICA   d ln(Gamma)/d ln(sigma) = d1*d2 - 1")
    print("-" * 84)
    print(f"{'ln(S/K)':>9} {'d1':>8} {'d2':>8} {'d1*d2-1':>9} {'medida':>9} "
          f"{'error':>10} {'gamma vs vol':>14}")
    max_err = 0.0
    T, sigma = 0.25, 0.30
    for ln_m in (-0.40, -0.25, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25, 0.40):
        K = S / np.exp(ln_m)
        d1, d2 = d1_d2(S, K, T, R, sigma, Q)
        pred = float(d1) * float(d2) - 1.0
        el = elasticity(S, K, T, sigma)
        err = abs(pred - el)
        max_err = max(max_err, err)
        lect = "SUBE" if pred > 0 else "baja"
        print(f"{ln_m:>9.2f} {float(d1):>8.3f} {float(d2):>8.3f} {pred:>9.3f} "
              f"{el:>9.3f} {err:>10.2e} {lect:>14}")
    print(f"\n  Error maximo entre forma cerrada y medida numerica: {max_err:.2e}")
    print("  (la identidad se verifica; la version SIN el -1 fallaba en 5 de 11 puntos)")

    lo, hi = crossover_moneyness(sigma, T)
    print(f"\n  Zona donde la gamma BAJA con la vol (sigma={sigma}, T={T}):")
    print(f"     {lo:+.3f} < ln(S/K) < {hi:+.3f}")
    print(f"  Banda ATM del codigo auditado: |ln(S/K)| < {ATM_BAND}")
    print(f"  => el cubo ATM cae ENTERO en la zona 'gamma ~ 1/sigma'.")
    print(f"  => el cubo OTM empieza en {ATM_BAND} y el cruce esta en {hi:.3f}:")
    print(f"     una franja estrecha comparte regimen con ATM, y de {hi:.3f} en")
    print(f"     adelante la gamma SUBE con la vol. Ahi vive la inversion de signo.")

    # ---------------------------------------------------------------- #
    print("\n\n2. ELASTICIDAD MEDIA POR CUBO, ponderada por |gamma x OI|")
    print("-" * 84)
    print("""   Se pondera por MAGNITUD |gamma*OI| y no por la suma con signo. Razon: la
   suma neta (calls menos puts) casi se cancela, asi que su elasticidad se
   dispara y cambia de signo por la cancelacion, no por el mecanismo. La media
   ponderada por magnitud responde a la pregunta correcta: en este cubo, como
   reacciona tipicamente la gamma a la volatilidad.""")

    strikes = S * np.exp(np.linspace(-0.50, 0.50, 401))
    tenors = {"fast (<=31d)": 14 / 365.0, "slow (>31d)": 120 / 365.0}

    print(f"\n{'cubo':>8} {'plazo':>14} {'peso %':>9} {'elast. media':>13} {'lectura':>24}")
    results: dict[tuple[str, str], float] = {}
    for tname, Tb in tenors.items():
        rows: dict[str, list[tuple[float, float]]] = {"ATM": [], "OTM": [], "ITM": []}
        for K in strikes:
            ln_m = float(np.log(S / K))
            for is_call in (True, False):
                b = bucket(ln_m, is_call)
                centre = 1.03 if is_call else 0.97
                oi = float(np.exp(-((K / S - centre) ** 2) / (2 * 0.06 ** 2)))
                oi *= 1.0 if is_call else 1.4
                gam = float(bs_gamma(S, K, Tb, R, sigma, Q))
                w = oi * gam                       # magnitud, sin signo
                if w <= 0:
                    continue
                d1, d2 = d1_d2(S, K, Tb, R, sigma, Q)
                rows[b].append((w, float(d1) * float(d2) - 1.0))
        grand = sum(w for v in rows.values() for w, _ in v)
        for b in ("ATM", "OTM", "ITM"):
            if not rows[b]:
                continue
            ws = np.array([w for w, _ in rows[b]])
            es = np.array([e for _, e in rows[b]])
            el = float(np.average(es, weights=ws))
            results[(b, tname)] = el
            lect = "gamma SUBE con la vol" if el > 0 else "gamma BAJA con la vol"
            print(f"{b:>8} {tname:>14} {100*ws.sum()/grand:>8.1f}% {el:>13.3f} {lect:>24}")

    # ---------------------------------------------------------------- #
    print("\n\n3. CONTRASTE CON LO OBSERVADO EN GEX_Asset_Management/reports/ortogonal.txt")
    print("-" * 84)
    print("""   La regresion es volatilidad futura ~ gamma. Si en un cubo la gamma SUBE
   con la vol, y la vol es persistente, el coeficiente sale POSITIVO. Si baja,
   NEGATIVO. Basta comparar signos.""")
    obs = {"ATM": -3.63, "OTM": +4.24}
    print(f"\n{'cubo':>8} {'t observado':>13} {'elast. teorica':>16} {'signos':>14}")
    agree_all = True
    for b in ("ATM", "OTM"):
        el = results.get((b, "slow (>31d)"), float("nan"))
        ok_b = np.sign(el) == np.sign(obs[b])
        agree_all &= bool(ok_b)
        print(f"{b:>8} {obs[b]:>13.2f} {el:>16.3f} "
              f"{'COINCIDEN' if ok_b else 'no coinciden':>14}")
    print(f"\n  Los dos signos se predicen desde la formula: {agree_all}")

    print("\n" + "=" * 84)
    print("LECTURA")
    print("=" * 84)
    print("""
  LO QUE QUEDA ESTABLECIDO (con certeza matematica)
  -------------------------------------------------
  La elasticidad de la gamma respecto a la volatilidad es exactamente

      d ln(Gamma) / d ln(sigma) = d1*d2 - 1

  y en una cadena real recorre TODO el rango de -1 (en el dinero) a +6 (en las
  alas), CAMBIANDO DE SIGNO. El punto de cruce depende de sigma*sqrt(T), asi que
  se mueve con el plazo: en vencimientos cortos cae cerca del dinero y casi todo
  el cubo OTM queda en regimen "la gamma sube con la vol"; en vencimientos largos
  se aleja y el mismo cubo OTM sigue en regimen "la gamma baja con la vol".

  LA CONSECUENCIA METODOLOGICA, QUE ES LO IMPORTANTE
  ---------------------------------------------------
  Si la dependencia Gamma-sigma es NO LINEAL y CAMBIA DE SIGNO segun moneyness y
  plazo, entonces ortogonalizar LINEALMENTE contra UNA sola IV --la mediana de la
  cadena, que es lo que hace `run_ortogonal.py`-- NO PUEDE eliminarla. Queda
  residuo mecanico por construccion.

  Eso no invalida el veredicto principal de ese informe: la Gamma cruda pasa de
  t = -2,48 a t = +0,04 al limpiarla, y esa caida es real. Lo que si pone en duda
  es EL UNICO SUPERVIVIENTE: `slow` ortogonal con t = -2,66. Sobrevivir a una
  limpieza incompleta no es evidencia de un canal economico; es lo que cabe
  esperar cuando el confundidor tiene una forma que el control no captura.

  LO QUE **NO** QUEDA DEMOSTRADO -- y conviene no exagerar
  --------------------------------------------------------
  La hipotesis de partida era que el signo invertido del cubo OTM (+4,24) se
  explicaba enteramente por este mecanismo. La reproduccion sale A MEDIAS:

      ATM slow : elasticidad -0,92   vs  t observado -3,63   -> coincide
      OTM fast : elasticidad +3,04                           -> signo compatible
      OTM slow : elasticidad -0,38   vs  t observado +4,24   -> NO coincide

  Con pesos de OI sinteticos y una sola sigma no se reproduce el signo del cubo
  OTM agregado. Puede ser la mezcla de plazos dentro del cubo, el perfil real de
  OI, o que haya ademas un problema en la descomposicion. NO se puede afirmar que
  la anomalia este explicada.

  EL CONTRASTE QUE LO ZANJARIA (necesita el panel OPRA real)
  -----------------------------------------------------------
  1. Ortogonalizar contra la IV EMPAREJADA AL PLAZO de cada cubo, y en forma no
     lineal: incluir 1/sigma y 1/(sigma*sqrt(T)) como controles, no sigma a secas.
  2. Repetir la descomposicion por moneyness usando d1*d2 = 1 como frontera en
     lugar de |ln(S/K)| = 0,10. La frontera economicamente relevante es donde
     cambia el signo de la elasticidad, no un numero redondo.
  3. Si `slow` muere ahi, el libro se cierra limpio y la conclusion pasa de
     "un superviviente sin explicar" a "nada sobrevive".

  Un dato previo apunta a que morira: en el test de prima de varianza --el mas
  limpio, porque restar la IV elimina el artefacto por construccion-- `slow` YA
  MUERE (t = -1,06).
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
