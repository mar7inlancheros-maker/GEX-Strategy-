#!/usr/bin/env python3
"""Borra las cadenas bajadas con la ventana de cotizacion equivocada.

EL BUG: la ventana estaba fija en 19:55-20:00 UTC. Eso equivale a 15:55-16:00 ET
solo durante el horario de verano (EDT, UTC-4). En horario estandar (EST, UTC-5)
el cierre son las 21:00 UTC, asi que la ventana caia a las 14:55-15:00 ET --
UNA HORA ANTES DEL CIERRE.

Consecuencias: la muestra mezclaba snapshots de las 15:55 y de las 14:55 segun la
epoca del ano, y el precio del subyacente (cierre diario, 16:00 ET) quedaba
descalzado una hora respecto a la cotizacion de la opcion. Ese descalce entra en
la IV y de ahi en la gamma, amplificado por la razon de cancelacion del hallazgo H1.

Ya corregido: close_window_utc() calcula la ventana en ET y la convierte a UTC,
asi que sigue al horario de verano automaticamente, y reintenta a las 13:00 ET
en las medias sesiones.

Este script borra solo los dias afectados (EST) para que la re-descarga los rehaga.
"""
from __future__ import annotations

import pathlib
import shutil
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent
CHAINS = ROOT / "data" / "raw" / "opra_chain"

# Transiciones de horario de verano en EE.UU.
# EDT: 2025-03-09 -> 2025-11-02  y  2026-03-08 -> 2026-11-01
# EST (afectado): 2025-11-02 -> 2026-03-08
EST_INI, EST_FIN = date(2025, 11, 2), date(2026, 3, 8)


def es_est(d: date) -> bool:
    return EST_INI <= d < EST_FIN


def main() -> int:
    if not CHAINS.exists():
        print("No hay cadenas en disco.")
        return 0
    dirs = sorted(CHAINS.glob("date=*"))
    buenos, malos = [], []
    for p in dirs:
        try:
            d = date.fromisoformat(p.name.split("=", 1)[1])
        except ValueError:
            continue
        (malos if es_est(d) else buenos).append((d, p))

    print(f"dias en disco: {len(dirs)}")
    print(f"  correctos (EDT, ventana 19:55-20:00 UTC = 15:55-16:00 ET): {len(buenos)}")
    for d, _ in buenos:
        print(f"      {d}")
    print(f"  AFECTADOS (EST, la ventana cayo a las 14:55-15:00 ET): {len(malos)}")
    for d, _ in malos:
        print(f"      {d}   <-- se borra")

    if not malos:
        print("\nNada que borrar. Ya puedes re-lanzar la ingesta.")
        return 0
    print(f"\nSe van a borrar {len(malos)} dias. Se vuelven a bajar en la siguiente")
    print(f"corrida, con la ventana correcta (~${len(malos)*1.77:.2f}).")
    if input("Continuar? [s/N]: ").strip().lower() != "s":
        print("Cancelado.")
        return 0
    for d, p in malos:
        shutil.rmtree(p)
        print(f"  borrado {d}")
    print("\nListo. Ahora corre:")
    print("  python3 run_ingesta.py --scope pilot --freq weekly --workers 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
