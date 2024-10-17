import sys

from loguru import logger


def configure_logger(log_levels=["INFO", "ERROR"]):
    logger.remove()

    for level in log_levels:
        logger.add(sys.stdout, level=level, format=f"{{time:MMMM D, YYYY}} | {level}: {{function}} {{message}}")


configure_logger()


def get_logger():
    return logger
