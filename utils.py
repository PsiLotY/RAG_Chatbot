"""This module contains utility functions that are used throughout the project.
"""

import logging
import sys
import os
from datetime import date
import json
import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def logger(name: str):
    """Create a logger object.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: The Logger object
    """
    return logging.getLogger(name)


log = logger(__name__)


def current_memory() -> None:
    """A function to log the current memory usage of the GPU."""
    free, total = torch.cuda.mem_get_info()
    free, total = free / 1000000000, total / 1000000000
    log.info(" used space %s, free space %s, total space %s in GB", total - free, free, total)


def create_directories_if_not_exist(*directories: str) -> None:
    """If the specified directories do not exist, create them."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_to_file(url: str, title: str, content: str, file_path: str) -> None:
    """Saves a string called content to a JSON file.

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
        log.info("Saved content to %s", file_path)
    except Exception as e:
        log.error("Failed to save content to %s: %s{", file_path, e)
