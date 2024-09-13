import sys
from functools import lru_cache

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@lru_cache(maxsize=10)
def warning_once(message: str):
    logger.opt(depth=1).warning(message)


logger.warning_once = warning_once
