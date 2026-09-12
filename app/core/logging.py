"""Structured logging setup for Grounded Customer Support Agent."""

import logging
import sys


class CorrelationFilter(logging.Filter):
    """Logging filter to inject or format correlation IDs (run_id)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return True


def setup_logger(name: str = "customer_support_agent", log_level: str = "INFO") -> logging.Logger:
    """Initialize and configure a structured logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [run_id:%(run_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(CorrelationFilter())

        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger()
