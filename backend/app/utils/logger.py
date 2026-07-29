from loguru import logger
import sys

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "{message}",
    level="INFO"
)

logger.add(
    "logs/retrieval.log",
    rotation="5 MB",
    retention="7 days",
    level="DEBUG"
)