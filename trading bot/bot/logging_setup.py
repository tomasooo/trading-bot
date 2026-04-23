"""Logging setup with Rich console + rotating file."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = RichHandler(rich_tracebacks=True, show_path=False, show_time=True)
    console.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            )
        )
        root.addHandler(fh)

    # Quiet noisy libs
    for name in ("httpx", "urllib3", "alpaca"):
        logging.getLogger(name).setLevel(logging.WARNING)
