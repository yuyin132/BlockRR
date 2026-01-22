import logging
import os
from datetime import datetime

def init_logger(log_dir='logs'):
    logger = logging.getLogger()

    # Remove all existing handlers (to allow reinitialization)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.setLevel(logging.INFO)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a new log file with a timestamp
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    # File handler: write logs to the file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(file_handler)

    # Console handler: also print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(console_handler)

    logging.info(f"✅ Logger initialized. Writing log to {log_file}")