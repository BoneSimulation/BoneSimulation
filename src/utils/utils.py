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


def debug_mode():
    return

def generate_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def loggeererar(message, level='info'):
    if debug_mode():
        log_message = f"[{generate_timestamp()}] [{logging.getLogger(__name__)}] [{level.upper()}]: {message}"

        if level.lower() == 'info':
            logging.info(log_message)
            print(Fore.GREEN + log_message + Fore.RESET)
        elif level.lower() == 'debug':
            logging.debug(log_message)
            print(Fore.MAGENTA + log_message + Fore.RESET)
        elif level.lower() == 'warning':
            logging.warning(log_message)
            print(Fore.YELLOW + log_message + Fore.RESET)
        elif level.lower() == 'error':
            logging.error(log_message)
            print(Fore.RED + log_message + Fore.RESET)
        elif level.lower() == '':
            logging.info(log_message)
            print(Fore.WHITE + log_message + Fore.RESET)
        else:
            logging.error("There is a misconfiguration with the logger")
    else:
        if level.lower() == 'debug':
            return
        elif level.lower() == 'info':
            return
        elif level.lower() == '':
            return
        else:
            log_message = f"[{generate_timestamp()}] [{logging.getLogger(__name__)}] [{level.upper()}]: {message}"
            if level.lower() == 'error':
                logging.error(log_message)
                print(Fore.RED + log_message + Fore.RESET)
            else:
                logging.error("There is a misconfiguration with the logger")
