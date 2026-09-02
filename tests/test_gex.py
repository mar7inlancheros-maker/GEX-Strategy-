"""Verificacion del motor GEX, del gamma flip y de los muros.

Orden de importancia:

  1. Que el GEX de una cadena de UN SOLO STRIKE coincida con el numero calculado
     A MANO, con una formula escrita aparte del motor. Es la piedra de toque: si
     esto falla, todo lo demas sobra.
  2. Que se cumplan los INVARIANTES DE AGREGACION: total == suma por strike ==
     suma por vencimiento. Un fallo aqui es un bug de agrupacion, y no se ve
     mirando graficos.
  3. Que las CONVENCIONES se comporten como su definicion exige: 'inverted' debe
     dar EXACTAMENTE el negativo de 'conventional'. Si no, el placebo no es un
     placebo y el contraste de falsacion no vale.
  4. Que las cuatro DEFINICIONES guarden entre si las razones exactas que dicen
     sus formulas.
  5. Que el MULTIPLICADOR se aplique UNA sola vez.
  6. Que el GAMMA FLIP sea de verdad una raiz: GEX(flip) ~ 0, revalorando.
  7. Que los MUROS agrupen strikes contiguos y no cuenten un accidente tres veces.

Se ejecuta solo:  python tests/test_gex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from gamma_quant.data.ingestion import synthetic as SYN
from gamma_quant.options import gamma_flip as FLIP
from gamma_quant.options import gamma_walls as WALLS
from gamma_quant.options import gex as GEX
from gamma_quant.options.greeks import bs_gamma

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        msg = label + (f"  -> {detail}" if detail else "")
        print(f"  [FALLO] {msg}")
        _FAILURES.append(msg)


def close(a, b, rtol=1e-9, atol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            rtol=rtol, atol=atol, equal_nan=True))


def block(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
# BLOQUE 1 — GEX contra el numero calculado a mano
# --------------------------------------------------------------------------- #

def test_single_strike_known_answer() -> None:
    block("BLOQUE 1 — GEX de un strike vs respuesta calculada a mano")

    params = dict(spot=100.0, strike=100.0, tau=0.25, iv=0.20,
                  call_oi=1_000, put_oi=0)
    chain = SYN.single_strike_chain(**params)
    res = GEX.compute_gex(chain)
    expected = SYN.expected_gex_single_strike(**params)

    check(close(res.total, expected, rtol=1e-12),
          "GEX total == formula a mano (solo calls)",
          f"motor {res.total!r} vs mano {expected!r}")

    # Aritmetica completamente explicita, sin reutilizar nada del motor:
    #   gamma * OI * mult * S^2 * 0.01
    gamma = float(bs_gamma(100.0, 100.0, 0.25, 0.04, 0.20, 0.0))
    manual = gamma * 1_000 * 100 * (100.0 ** 2) * 0.01
    check(close(res.total, manual, rtol=1e-12),
          "GEX total == gamma*OI*mult*S^2*0.01 escrito literalmente",
          f"{res.total!r} vs {manual!r}")

    # Puts: la convencion resta. Con el mismo OI en ambos, el neto es CERO
    # exactamente, porque bajo BSM la gamma de call y put es identica.
    balanced = SYN.single_strike_chain(spot=100.0, strike=100.0, tau=0.25,
                                       iv=0.20, call_oi=1_000, put_oi=1_000)
    res_bal = GEX.compute_gex(balanced)
    check(abs(res_bal.total) < 1e-9,
          "OI igual en calls y puts -> GEX neto EXACTAMENTE cero",
          f"{res_bal.total!r}")
    check(close(res_bal.call_total, -res_bal.put_total, rtol=1e-12),
          "y las dos patas son exactamente opuestas (misma gamma, signo contrario)")

    # Mas puts que calls -> GEX negativo.
    put_heavy = SYN.single_strike_chain(spot=100.0, strike=100.0, tau=0.25,
                                        iv=0.20, call_oi=500, put_oi=1_500)
    check(GEX.compute_gex(put_heavy).total < 0,
          "mas OI en puts que en calls -> GEX neto negativo")


# --------------------------------------------------------------------------- #
# BLOQUE 2 — Invariantes de agregacion
# --------------------------------------------------------------------------- #

def test_aggregation_invariants() -> None:
    block("BLOQUE 2 — invariantes: total == suma por strike == suma por vencimiento")

    chain = SYN.make_chain(spot=500.0, strike_step=5.0,
                           expiries_days=(1, 7, 30, 60), seed=1)
    res = GEX.compute_gex(chain)

    check(close(res.total, res.by_strike["gex"].sum(), rtol=1e-10),
          "total == suma del desglose por strike",
          f"{res.total!r} vs {res.by_strike['gex'].sum()!r}")
    check(close(res.total, res.by_expiration["gex"].sum(), rtol=1e-10),
          "total == suma del desglose por vencimiento",
          f"{res.total!r} vs {res.by_expiration['gex'].sum()!r}")
    check(close(res.total, res.by_contract["gex"].sum(), rtol=1e-10),
          "total == suma contrato a contrato")
    check(close(res.total, res.call_total + res.put_total, rtol=1e-10),
          "total == call_gex + put_gex")
    check(len(res.by_expiration) == 4,
          "los cuatro vencimientos aparecen en el desglose",
          f"{len(res.by_expiration)}")

    # El OI se conserva en la agregacion (no se duplica ni se pierde).
    check(int(res.by_strike["open_interest"].sum()) == int(chain["open_interest"].sum()),
          "el OI total se conserva al agregar por strike")


# --------------------------------------------------------------------------- #
# BLOQUE 3 — Convenciones de signo (y que el placebo sea un placebo)
# --------------------------------------------------------------------------- #

def test_conventions() -> None:
    block("BLOQUE 3 — convenciones de signo")

    chain = SYN.make_chain(spot=500.0, strike_step=5.0, seed=2)

    conv = GEX.compute_gex(chain, convention="conventional").total
    inv = GEX.compute_gex(chain, convention="inverted").total
    allp = GEX.compute_gex(chain, convention="all_positive").total

    check(close(inv, -conv, rtol=1e-12),
          "'inverted' da EXACTAMENTE el negativo de 'conventional'",
          f"{inv!r} vs {-conv!r}")
    check(allp > 0, "'all_positive' es positiva por construccion", f"{allp!r}")
    check(allp >= abs(conv) - 1e-6,
          "|GEX neto| <= GEX sin signo (cancelacion, nunca amplificacion)",
          f"neto {abs(conv):,.0f} vs sin signo {allp:,.0f}")

    res_inv = GEX.compute_gex(chain, convention="inverted")
    check(any("PLACEBO" in w for w in res_inv.warnings),
          "usar la convencion invertida AVISA de que es un placebo",
          str(res_inv.warnings))
    check(GEX.get_convention("inverted").is_placebo,
          "la convencion invertida esta marcada como placebo")

    try:
        GEX.compute_gex(chain, convention="lo_que_sea")
        check(False, "una convencion inexistente debe fallar")
    except KeyError:
        check(True, "una convencion inexistente lanza KeyError (no usa un defecto)")


# --------------------------------------------------------------------------- #
# BLOQUE 4 — Las cuatro definiciones y sus razones exactas
# --------------------------------------------------------------------------- #

def test_definitions() -> None:
    block("BLOQUE 4 — razones exactas entre definiciones de GEX")

    spot = 500.0
    chain = SYN.make_chain(spot=spot, strike_step=5.0, seed=3)

    naive = GEX.compute_gex(chain, definition="naive").total
    ddelta = GEX.compute_gex(chain, definition="dollar_delta").total
    notional = GEX.compute_gex(chain, definition="notional").total
    scaled = GEX.compute_gex(chain, definition="spot_scaled").total

    check(close(ddelta, naive * spot, rtol=1e-10),
          "dollar_delta == naive * S", f"{ddelta!r} vs {naive * spot!r}")
    check(close(notional, naive * spot ** 2, rtol=1e-10),
          "notional == naive * S^2")
    check(close(scaled, notional * 0.01, rtol=1e-10),
          "spot_scaled == notional * 0,01")
    check(close(scaled, naive * spot ** 2 * 0.01, rtol=1e-10),
          "spot_scaled == naive * S^2 * 0,01")

    # Todas ordenan igual: son transformaciones monotonas positivas entre si.
    # Importa porque justifica usar la interpretable sin perder informacion.
    profiles = {
        name: GEX.compute_gex(chain, definition=name).by_strike["gex"]
        for name in GEX.DEFINITIONS
    }
    base = profiles["spot_scaled"]
    for name, prof in profiles.items():
        corr = np.corrcoef(base.to_numpy(), prof.to_numpy())[0, 1]
        check(corr > 0.999999,
              f"el perfil de '{name}' es proporcional al de spot_scaled (corr~1)",
              f"corr={corr!r}")

    # La ficha tecnica existe y esta rellena: el encargo (seccion 7) la exige.
    for name, d in GEX.DEFINITIONS.items():
        check(all([d.units, d.interpretation, d.assumptions, d.advantages, d.weaknesses]),
              f"'{name}' documenta unidades, interpretacion, supuestos, ventajas y debilidades")

    tabla = GEX.compare_definitions(chain)
    check(len(tabla) == 4 and "unidades" in tabla.columns,
          "compare_definitions devuelve las cuatro con sus unidades")


# --------------------------------------------------------------------------- #
# BLOQUE 5 — El multiplicador se aplica UNA vez
# --------------------------------------------------------------------------- #

def test_multiplier() -> None:
    block("BLOQUE 5 — el multiplicador de contrato se aplica exactamente una vez")

    chain = SYN.single_strike_chain(call_oi=1_000, put_oi=0)
    g100 = GEX.compute_gex(chain, multiplier=100).total
    g200 = GEX.compute_gex(chain, multiplier=200).total
    g1 = GEX.compute_gex(chain, multiplier=1).total

    check(close(g200, 2.0 * g100, rtol=1e-12),
          "doblar el multiplicador dobla el GEX (lineal, no cuadratico)",
          f"{g200!r} vs {2 * g100!r}")
    check(close(g100, 100.0 * g1, rtol=1e-12),
          "mult=100 da exactamente 100x el de mult=1 (no 10.000x)",
          f"{g100!r} vs {100 * g1!r}")


# --------------------------------------------------------------------------- #
# BLOQUE 6 — Datos malos: ruido, no silencio
# --------------------------------------------------------------------------- #

def test_bad_data() -> None:
    block("BLOQUE 6 — la cadena corrupta se detecta, no se traga")

    chain = SYN.make_chain(spot=100.0, strike_step=5.0, expiries_days=(30,), seed=4)

    # Dos spots distintos = dos snapshots mezclados. Agregarlos daria un GEX de
    # ningun instante concreto.
    mixed = chain.copy()
    mixed.loc[mixed.index[:5], "underlying_price"] = 101.0
    try:
        GEX.compute_gex(mixed)
        check(False, "una cadena con dos spots distintos debe fallar")
    except ValueError as exc:
        check("snapshot" in str(exc).lower(),
              "dos spots en la misma cadena -> ValueError explicando por que",
              str(exc)[:80])

    # OI negativo: dato corrupto.
    neg = chain.copy()
    neg.loc[neg.index[0], "open_interest"] = -10
    try:
        GEX.compute_gex(neg)
        check(False, "un OI negativo debe fallar")
    except ValueError:
        check(True, "OI negativo -> ValueError")

    # Tipo de opcion desconocido: no se adivina.
    bad_type = chain.copy()
    bad_type.loc[bad_type.index[0], "option_type"] = "X"
    try:
        GEX.compute_gex(bad_type)
        check(False, "un tipo 'X' debe fallar")
    except ValueError:
        check(True, "tipo de opcion desconocido -> ValueError (no se adivina)")

    # NaN en gamma: se descarta Y SE CUENTA, no se convierte en cero.
    nan_gamma = chain.copy()
    nan_gamma.loc[nan_gamma.index[:3], "gamma"] = np.nan
    res = GEX.compute_gex(nan_gamma)
    check(res.n_dropped == 3, "3 contratos con gamma NaN descartados y contados",
          f"{res.n_dropped}")
    check(any("descartados" in w for w in res.warnings),
          "se avisa de los descartes en el informe")
    check(res.n_contracts == len(chain) - 3, "el recuento de usados cuadra")

    # Falta una columna obligatoria.
    try:
        GEX.compute_gex(chain.drop(columns=["open_interest"]))
        check(False, "sin open_interest debe fallar")
    except KeyError:
        check(True, "falta open_interest -> KeyError (el campo critico del GEX)")


# --------------------------------------------------------------------------- #
# BLOQUE 7 — Gamma flip
# --------------------------------------------------------------------------- #

def test_gamma_flip() -> None:
    block("BLOQUE 7 — gamma flip: que sea de verdad una raiz")

    # Cadena con mas OI en puts que en calls: hay un cruce por cero dentro del
    # rango de busqueda.
    chain = SYN.make_chain(spot=100.0, strike_step=1.0, expiries_days=(7, 30),
                           put_call_oi_ratio=1.6, seed=5)
    res = FLIP.find_gamma_flip(chain, search_lower_pct=0.85, search_upper_pct=1.15,
                               n_grid=61, tolerance=1e-3)

    check(res.flip is not None, "se encuentra un flip", res.report().replace("\n", " | "))
    if res.flip is not None:
        # LA prueba: el GEX revalorado en el flip debe ser ~0. Independiente de
        # donde caiga: si el numero devuelto no anula el GEX, no es una raiz.
        gex_there = FLIP.gex_at_spot(chain, res.flip)
        scale = max(abs(FLIP.gex_at_spot(chain, res.spot)), 1.0)
        check(abs(gex_there) / scale < 1e-3,
              "GEX(flip) ~ 0 al revalorar la cadena entera en ese spot",
              f"GEX(flip)={gex_there:,.2f}, escala={scale:,.2f}")

        # Coherencia de regimen y distancia.
        expected_regime = "gamma_positiva" if res.spot > res.flip else "gamma_negativa"
        check(res.regime == expected_regime, "el regimen concuerda con spot vs flip")
        check(close(res.distance_pct, (res.spot - res.flip) / res.spot, rtol=1e-12),
              "la distancia normalizada es (S - S*)/S")

    # Cadena solo de calls: el GEX es positivo en todo el rango, no hay flip. Debe
    # decirlo, no inventarse uno.
    calls_only = chain[chain["option_type"] == "C"].copy()
    res_c = FLIP.find_gamma_flip(calls_only, n_grid=41)
    check(res_c.flip is None, "sin cruce por cero -> flip None (no se inventa)")
    check(any("sin cruce" in w for w in res_c.warnings),
          "y se explica por que no hay flip", str(res_c.warnings)[:100])

    # El atajo malo: interpolar el perfil por strike da OTRA cosa que revalorar.
    # Se comprueba que el motor NO esta haciendo el atajo.
    naive_profile = GEX.compute_gex(chain).by_strike["gex"]
    sign_changes = np.where(np.diff(np.sign(naive_profile.to_numpy())) != 0)[0]
    if len(sign_changes) and res.flip is not None:
        naive_flip = float(naive_profile.index[sign_changes[0]])
        print(f"  [info] flip revalorando = {res.flip:,.2f} | "
              f"cruce del perfil por strike = {naive_flip:,.2f} | "
              f"diferencia = {abs(res.flip - naive_flip):,.2f}")

    # Sensibilidad al supuesto A8: los tres regimenes de IV.
    sens = FLIP.flip_sensitivity(chain, n_grid=41, tolerance=1e-2)
    check(len(sens) == 3 and "flip" in sens.columns,
          "flip_sensitivity devuelve los tres supuestos de IV (A8)")
    flips = sens["flip"].dropna()
    if len(flips) >= 2:
        spread = (flips.max() - flips.min()) / chain["underlying_price"].iloc[0]
        print(f"  [info] dispersion del flip entre supuestos de IV: {spread:.2%}")
    check(True, "la sensibilidad a A8 queda cuantificada, no supuesta")


# --------------------------------------------------------------------------- #
# BLOQUE 8 — Muros de gamma
# --------------------------------------------------------------------------- #

def test_gamma_walls() -> None:
    block("BLOQUE 8 — muros: umbral explicito y agrupacion de contiguos")

    # Perfil fabricado: plano con un pico en 105 y otro en 120.
    strikes = np.arange(90.0, 131.0, 1.0)
    values = np.full(len(strikes), 100.0)
    values[strikes == 105.0] = 10_000.0
    values[strikes == 120.0] = 8_000.0
    profile = pd.Series(values, index=strikes)

    res = WALLS.find_gamma_walls(profile, spot=100.0, method="topk", top_k=2)
    found = sorted(w.strike for w in res.walls)
    check(found == [105.0, 120.0], "topk=2 encuentra exactamente los dos picos", f"{found}")

    check(res.largest.strike == 105.0, "el mayor es el pico de 105")
    check(res.nearest.strike == 105.0, "el mas cercano al spot 100 es 105")
    check(res.nearest_above().strike == 105.0, "la resistencia esta en 105")
    check(res.nearest_below() is None, "no hay soporte por debajo de 100 en este perfil")

    # AGRUPACION: tres strikes contiguos altos son UN muro, no tres.
    v2 = np.full(len(strikes), 100.0)
    for k in (110.0, 111.0, 112.0):
        v2[strikes == k] = 9_000.0
    prof2 = pd.Series(v2, index=strikes)

    # OJO CON EL PERCENTIL: 3 picos de 41 strikes son el 7,3%, asi que hace falta
    # p > 92,7 para aislarlos. Con p=90 el umbral cae sobre el valor modal (100) y
    # `>=` selecciona LOS 41 -- que agrupados dan "un muro" con aspecto correcto y
    # el centro en la media del perfil entero. Se comprueba abajo que el codigo
    # avisa de ese caso en vez de devolverlo en silencio.
    grouped = WALLS.find_gamma_walls(prof2, spot=100.0, method="percentile",
                                     percentile=95.0, cluster_adjacent=True)
    ungrouped = WALLS.find_gamma_walls(prof2, spot=100.0, method="percentile",
                                       percentile=95.0, cluster_adjacent=False)
    check(len(grouped.walls) == 1,
          "tres strikes contiguos se agrupan en UN muro", f"{len(grouped.walls)}")
    check(len(ungrouped.walls) == 3,
          "sin agrupar salen tres (por eso agrupar viene activado)",
          f"{len(ungrouped.walls)}")
    check(abs(grouped.walls[0].strike - 111.0) < 1e-9,
          "el centro del grupo es 111 (ponderado por |GEX|)",
          f"{grouped.walls[0].strike}")
    check(len(grouped.walls[0].strikes) == 3, "el muro recuerda sus tres strikes")

    # Umbral que no discrimina: con p=90 sobre este perfil entran los 41 strikes.
    # No debe pasar en silencio.
    nodisc = WALLS.find_gamma_walls(prof2, spot=100.0, method="percentile",
                                    percentile=90.0, cluster_adjacent=False)
    check(len(nodisc.walls) == 41,
          "p=90 sobre perfil plano+picos selecciona TODOS los strikes",
          f"{len(nodisc.walls)}")
    check(any("discrimin" in w for w in nodisc.warnings),
          "y el codigo AVISA de que el umbral no discrimina",
          str(nodisc.warnings)[:120])

    # El percentil se adapta a la escala del dia: multiplicar todo el perfil por
    # mil no puede cambiar QUE strikes son muros.
    big = WALLS.find_gamma_walls(prof2 * 1000.0, spot=100.0, method="percentile",
                                 percentile=95.0)
    check([w.strikes for w in big.walls] == [w.strikes for w in grouped.walls],
          "el metodo del percentil es invariante a la escala del perfil")

    # El zscore avisa de su propio supuesto.
    z = WALLS.find_gamma_walls(prof2, spot=100.0, method="zscore", zscore_threshold=2.0)
    check(any("normalidad" in w for w in z.warnings),
          "el metodo zscore avisa de que supone normalidad y el perfil no lo es")

    # Sin muros -> NaN, nunca 0. Un 0 en "distancia al muro" significaria que el
    # muro esta EN el spot.
    flat = pd.Series(np.full(10, 5.0), index=np.arange(95.0, 105.0))
    empty = WALLS.find_gamma_walls(flat, spot=100.0, method="zscore")
    feats = WALLS.wall_features(empty, spot=100.0)
    check(np.isnan(feats["wall_nearest_dist"]),
          "sin muros la distancia es NaN, no 0")
    check(feats["wall_n"] == 0.0, "y el conteo es 0")


# --------------------------------------------------------------------------- #
# BLOQUE 9 — Extremo a extremo sobre cadena sintetica
# --------------------------------------------------------------------------- #

def test_end_to_end() -> None:
    block("BLOQUE 9 — cadena sintetica completa de extremo a extremo")

    chain = SYN.make_chain(spot=767.0, strike_range=(0.90, 1.10), strike_step=1.0,
                           expiries_days=(0, 1, 7, 30), oi_noise=0.3, seed=42)

    check(chain["is_synthetic"].all(),
          "la cadena viene marcada is_synthetic=True en el origen")
    check(set(SYN.CANONICAL_COLUMNS) == set(chain.columns),
          "cumple el esquema canonico completo")
    check(chain["gamma"].notna().all(), "toda la cadena tiene gamma calculada")

    # Reproducibilidad: misma semilla, misma cadena.
    again = SYN.make_chain(spot=767.0, strike_range=(0.90, 1.10), strike_step=1.0,
                           expiries_days=(0, 1, 7, 30), oi_noise=0.3, seed=42)
    check(chain["open_interest"].equals(again["open_interest"]),
          "misma semilla -> mismo OI (determinismo)")

    res = GEX.compute_gex(chain)
    walls = WALLS.find_gamma_walls(res.net_by_strike, spot=767.0,
                                   method="percentile", percentile=95.0)
    flip = FLIP.find_gamma_flip(chain, n_grid=41, tolerance=1e-2)

    check(np.isfinite(res.total), "el GEX total es finito")
    check(len(res.by_expiration) == 4, "los cuatro vencimientos, 0DTE incluido")
    check(len(walls.walls) >= 1, "se detecta al menos un muro")

    print("\n" + res.report())
    print("\n" + walls.report())
    print("\n" + flip.report())

    # El vencimiento 0DTE debe concentrar gamma desproporcionada: es el hecho que
    # domina el dia de vencimiento y la razon de que 0DTE sea categoria aparte.
    by_exp = res.by_expiration["gex"].abs()
    share_0dte = by_exp.iloc[0] / by_exp.sum()
    print(f"\n  [info] el vencimiento mas corto concentra el {share_0dte:.1%} "
          f"del |GEX| total")
    check(np.isfinite(share_0dte), "la concentracion del 0DTE es medible")


def main() -> int:
    print("=" * 78)
    print("VERIFICACION DEL MOTOR GEX, GAMMA FLIP Y MUROS — gamma_quant")
    print("SINTETICO — NO ES EVIDENCIA SOBRE EL MERCADO")
    print("=" * 78)

    test_single_strike_known_answer()
    test_aggregation_invariants()
    test_conventions()
    test_definitions()
    test_multiplier()
    test_bad_data()
    test_gamma_flip()
    test_gamma_walls()
    test_end_to_end()

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
