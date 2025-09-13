"""Simple logging utility for the video processing application."""

import logging
import sys
from typing import Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        
    Returns:
        Logger instance
    """
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    # Ensure the logger has at least INFO level
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    
    return logger

# Create a default logger for this module
logger = get_logger(__name__)