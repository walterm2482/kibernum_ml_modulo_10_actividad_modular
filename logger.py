# logger.py
"""
logger.py
---------
Configura un logger consistente para todos los módulos del proyecto.
"""

import logging
import os


def get_logger(name: str = "mlops") -> logging.Logger:
    """
    Crea o recupera un logger configurado.

    Parameters
    ----------
    name : str, optional
        Nombre del logger. Por defecto "mlops".

    Returns
    -------
    logging.Logger
        Instancia configurada de logger.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
