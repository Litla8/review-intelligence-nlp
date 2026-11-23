"""Shared logging setup so every entry point uses the same format and dynamic level."""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with the given level name (INFO, DEBUG, WARNING, ...).

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level!r}")
    logging.basicConfig(level=numeric, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
