"""
utils/utils.py

This file contains utility functions for generating timestamps and checking the operating system type.
"""

import datetime
import platform

def generate_timestamp():
    """Generates a timestamp for file naming."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def check_os():
    """Checks the operating system type."""
    return platform.system()