import logging 
from datetime import datetime 
import os 
from logging.handlers import RotatingFileHandler
# from from_root import from_root
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

MAX_LOG_SIZE = 5 * 1024 * 1024
MAX_FILES = 3

def configure_logger():
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter=logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    
    file_handler= RotatingFileHandler(LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=MAX_FILES)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
configure_logger()
