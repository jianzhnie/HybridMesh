"""Colored distributed-aware logging with rank information.

hpmesh configures one stdout handler per module logger instead of touching the
root logger, so a module that wants its INFO lines on the console takes its
logger from ``get_logger`` here rather than from ``logging.getLogger``. The
rank decision is made by a filter *at emit time*, not when the logger is
created: engine modules are imported before ``init_dist``, when the real rank
is not knowable yet, so an import-time rank check would hand every rank a
rank-0 configuration.
"""

from __future__ import annotations

import logging
import sys
from logging import Formatter, LogRecord
from typing import ClassVar

from colorama import Fore, Style

logger_initialized: dict[str, bool] = {}


class ColorfulFormatter(Formatter):
    """Formatter that adds ANSI color codes and rank information to log messages."""

    COLORS: ClassVar[dict[str, str]] = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.LIGHTGREEN_EX,
    }

    def format(self, record: LogRecord) -> str:
        # Add rank information to the record
        record.rank = self._get_rank()

        # Format the log message
        log_message = super().format(record)

        # Add color based on log level
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET

    def _get_rank(self) -> int:
        return get_distributed_rank()


class MainProcessFilter(logging.Filter):
    """Decide per record, at emit time, whether it reaches the console.

    Regular lines (below ERROR) pass only on the rank-0 process -- that is the
    whole point of distributed-aware logging. ERROR and above pass on every
    rank: a failure on rank 3 is exactly the line you cannot afford to lose,
    and the old import-time configuration let those through (via the root
    logger's last-resort handler) precisely because non-main loggers carried
    no handler of their own.
    """

    def filter(self, record: LogRecord) -> bool:
        return record.levelno >= logging.ERROR or get_distributed_rank() == 0


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a module logger with a rank-aware stdout handler.

    Below ERROR, only the rank-0 process prints; ERROR and above print on
    every rank. The decision is made by ``_MainProcessFilter`` at emit time,
    so the logger can be created at module import time, before the process
    group exists. The handler is attached once per ``name``; repeat calls
    return the same logger untouched.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    if logger.handlers:
        logger.handlers.clear()

    fmt = (
        "%(asctime)s - [Rank %(rank)d] - "
        "%(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorfulFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(MainProcessFilter())
    logger.addHandler(handler)
    logger.setLevel(log_level)

    logger_initialized[name] = True
    return logger


def get_distributed_rank() -> int:
    """Return the current distributed rank, falling back to the RANK env var or 0.

    Public because components outside the logging stack have to ask the same
    question -- the metrics processor's rank and the report it decorates are
    separate things, and it needs the first to decide the second.
    """
    try:
        from ..accelerator import dist_utils

        if dist_utils.is_distributed():
            return dist_utils.get_rank()
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Failed to query distributed rank, falling back to env var.", exc_info=True
        )

    # Fallback to environment variables
    from ..accelerator.device import get_env_dist_info

    return get_env_dist_info()[0]
