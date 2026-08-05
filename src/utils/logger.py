"""
logger.py
---------
Shared logger. Import `logger` from here in every module.
Keeps formatting consistent and makes it easy to switch log levels.
"""

import logging
import sys


def get_logger(name: str = "copilot") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
