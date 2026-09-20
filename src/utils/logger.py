"""
Structured logging module for the Cognitive Data Processing Pipeline.
Provides formatted console logging with optional color support and configurable log levels.
"""

import sys
import logging
from typing import Optional
from src.utils.config import config

# Custom ANSI color codes for terminal log levels
COLOR_CODES = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",      # Reset formatting
}


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds color highlights to console output."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Create a shallow copy of the record so we don't permanently alter levelname for other formatters
        record_copy = logging.makeLogRecord(record.__dict__)
        if self.use_color and record_copy.levelname in COLOR_CODES:
            color = COLOR_CODES[record_copy.levelname]
            reset = COLOR_CODES["RESET"]
            record_copy.levelname = f"{color}{record_copy.levelname:<8}{reset}"
        else:
            record_copy.levelname = f"{record_copy.levelname:<8}"
        return super().format(record_copy)


def get_logger(name: str = "cognitive_pipeline", level: Optional[str] = None) -> logging.Logger:
    """
    Creates or retrieves a configured logger instance.

    Args:
        name: Name of the logger (typically __name__ or module identifier).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, defaults to config.LOG_LEVEL.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger is already configured
        return logger

    # Determine log level
    log_level_str = (level or config.LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Console handler setup
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(fmt=log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


# Default logger instance
logger = get_logger("cognitive_pipeline")
