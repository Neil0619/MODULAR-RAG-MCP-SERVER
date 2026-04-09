"""Structured logging — stderr only (stdout reserved for MCP protocol messages)."""

import logging
import sys


def get_logger(name: str = "rag") -> logging.Logger:
    """Get a logger that outputs to stderr.

    stdout is reserved for MCP JSON-RPC messages; all logs go to stderr
    to avoid polluting the protocol channel.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
