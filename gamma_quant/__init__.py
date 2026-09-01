"""gamma_quant — contraste de la hipotesis de Gamma Exposure (GEX) de dealers.

QUE ES ESTE PAQUETE
-------------------
Un aparato de medida, no una estrategia. Su trabajo es construir GEX de forma
auditable a partir de cadenas de opciones y, despues, intentar TUMBAR la
hipotesis de que ese GEX predice algo.

LO QUE ESTE PAQUETE NO SABE
---------------------------
No sabe cual es el posicionamiento real de los dealers. Nadie lo sabe: el
inventario de un market maker es privado. Todo el GEX publico -- este incluido --
lo INFIERE del open interest mediante una CONVENCION DE SIGNO que no puede
verificarse con datos publicos (supuesto A1/A2 del PROJECT_PLAN).

Por eso `options.gex` trata la convencion como un objeto intercambiable y por eso
el placebo de invertirla es un test de primera clase. Si la señal sobrevive a
invertir el signo, la señal no venia del signo.

ESTADO DE LOS DATOS (31-08-2026)
--------------------------------
No hay historico de cadenas. Las fuentes gratuitas dan snapshot del presente. El
proyecto archiva hacia delante desde el 2026-08-31; hasta que ese archivo tenga
tamaño, cualquier resultado de estrategia es una prueba del codigo y va marcado
`SINTETICO -- NO ES EVIDENCIA`.

ORGANIZACION
------------
    options/        griegas, motor GEX, gamma flip, muros
    data/           ingesta, limpieza, validacion, almacen
    features/       ingenieria de features
    strategies/     reglas de trading
    backtest/       ejecucion y costes
    research/       contrastes (NADIE del motor importa de aqui)
    models/         ML, solo si el baseline tiene señal
    visualization/  graficos
"""

from __future__ import annotations

__version__ = "0.1.0"

# Fecha en la que arranca el archivo propio de cadenas. Cualquier analisis que
# pretenda usar datos anteriores esta usando datos que no tenemos.
ARCHIVE_START_DATE = "2026-08-31"

__all__ = ["__version__", "ARCHIVE_START_DATE"]
