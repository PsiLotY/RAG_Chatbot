import logging
import sys
import os
from datetime import date
from collections import deque
import json
import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

def logger(name:str):
    log = logging.getLogger(name)
    return log


log = logger(__name__)


def current_memory():
    free, total = torch.cuda.mem_get_info()
    free, total= free/1000000000, total/1000000000
    log.info(f" used space {total-free}, free space {free}, total space {total} in GB")


def create_directories_if_not_exist(*directories: str) -> None:
    """

    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_to_file(url: str, title: str, content: str, file_path: str, log) -> None:
    """Saves extracted content to a JSON file.

    Args:
        url (str): The URL of the document.
        title (str): The title of the document.
        content (str): The content of the document.
        file_path (str): The path to save the JSON file.
    """
    try:
        data = {
            "url": url,
            "title": title,
            "accessed": str(date.today()),
            "content": content,
        }
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        log.info(f"Saved content to {file_path}")
    except Exception as e:
        log.error(f"Failed to save content to {file_path}: {e}")
