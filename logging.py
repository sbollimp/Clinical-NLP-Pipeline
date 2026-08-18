"""Consistent, quiet-by-default logging for the pipeline."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def configure(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("%s ...", label)
    yield
    logger.info("%s done in %.2fs", label, time.perf_counter() - start)
