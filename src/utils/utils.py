import sys
import os
import logging
from colorama import init, Fore
import datetime
import platform

def check_os():
    if platform.system() == "Windows":
        return "Windows"
    elif platform.system() == "Linux":
        return "Linux"
    elif platform.system() == "Darwin":
        return "MacOS"
    else:
        return "Unknown"



def generate_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")