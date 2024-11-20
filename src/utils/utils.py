"""
Just some random not very important but necessary functions
"""

import datetime
import platform


def check_os():
    """
    Checks the operating system of the current environment.

    This function determines the operating system being used and returns a string
    indicating the OS type. It recognizes Windows, Linux, and macOS (Darwin).

    Returns:
        str: A string representing the operating system:
            - "Windows" for Windows OS
            - "Linux" for Linux OS
            - "MacOS" for macOS
            - "Unknown" if the OS is not recognized
    """
    if platform.system() == "Windows":
        return "Windows"
    if platform.system() == "Linux":
        return "Linux"
    if platform.system() == "Darwin":
        return "MacOS"
    else:
        return "Unknown"


def generate_timestamp():
    """checks and returns the current datetime"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
