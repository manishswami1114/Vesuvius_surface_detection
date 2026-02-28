import logging
import sys

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
