"""
monitoring.py

Provides monitoring and profiling utilities for the anonymization process.
"""

import logging
import psutil
import time
from functools import wraps
from datetime import datetime
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler('anonymization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        duration = end_time - start_time
        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
        
        logger.info(
            f"{func.__name__} completed in {duration:.2f}s. "
            f"Memory delta: {memory_delta:.2f}MB"
        )
        return result
    return wrapper

class ResourceMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self._stop = mp.Event()
        
    def start(self):
        self.process = mp.Process(target=self._monitor)
        self.process.start()
        
    def stop(self):
        self._stop.set()
        self.process.join()
        
    def _monitor(self):
        while not self._stop.is_set():
            cpu_percent = psutil.cpu_percent(interval=self.interval)
            mem = psutil.virtual_memory()
            logger.info(f"CPU: {cpu_percent}% | Memory: {mem.percent}%")
