"""Logging del proyecto.

POR QUE UN MODULO PARA ESTO
---------------------------
Dos razones practicas, ambas aprendidas a base de perder tiempo:

1. WINDOWS. La consola por defecto descodifica en cp1252 y revienta con la
   primera 'σ' o '±' de un mensaje. Aqui se fuerza UTF-8 en el handler.

2. TRAZABILIDAD DEL ARCHIVADOR. El archivo de cadenas es irreemplazable y corre
   desatendido: si un dia falla la descarga y nadie se entera, aparece un hueco
   silencioso en la muestra. Por eso el archivador escribe SIEMPRE a fichero,
   ademas de a consola, y los fallos quedan en `reports/archiver.log`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED: set[str] = set()

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Logger configurado una sola vez por nombre.

    `log_file` añade un handler a disco. Se usa en procesos desatendidos
    (archivador) donde perder el error significa perder el dato.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # Consola. `errors="replace"` para que un caracter raro nunca tumbe el
    # proceso que esta descargando datos.
    stream = logging.StreamHandler(sys.stdout)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8", errors="replace")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger
