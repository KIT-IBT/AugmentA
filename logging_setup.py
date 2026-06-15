import logging
import os
import sys
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("AugmentA")
logger.setLevel(logging.DEBUG)  # capture all levels

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

file_handler = RotatingFileHandler("AugmentA.log", maxBytes=1_000_000, backupCount=3)
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
ch.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers if not already added
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(ch)
logger.addHandler(file_handler)

def set_log_file(path: str):
    """Add or update a rotating file handler"""
    global file_handler
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if file_handler:
        logger.removeHandler(file_handler)
    file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    logger.info("This is a test")