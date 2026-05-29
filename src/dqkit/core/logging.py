"""Structured, single-configuration logging for the toolkit."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def _configure_once() -> None:
    """Attach one stderr handler to the ``dqkit`` root logger, idempotently."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("dqkit")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``dqkit``.

    The first call configures a single stderr handler on the ``dqkit`` root
    logger; later calls reuse it, so importing many modules never stacks
    duplicate handlers.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` under the ``dqkit`` namespace.
    """
    _configure_once()
    if name == "dqkit" or name.startswith("dqkit."):
        return logging.getLogger(name)
    return logging.getLogger(f"dqkit.{name}")
