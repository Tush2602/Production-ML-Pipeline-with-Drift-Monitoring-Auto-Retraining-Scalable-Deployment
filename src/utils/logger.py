import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    level = logging.INFO,
    format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
)
def get_logger(name:str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
    
