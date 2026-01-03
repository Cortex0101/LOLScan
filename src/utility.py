import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple
import numpy as np

# import config 
import config as cfg

logger = logging.getLogger(__name__)

def remove_all_files_starting_with(directory: str, prefix: str) -> None:
    """Remove all files in the specified directory that start with the given prefix.
    
    Args:
        directory: Path to the target directory
        prefix: Prefix string to match files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"Directory does not exist: {directory}")
        return
    
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.name.startswith(prefix):
            try:
                file_path.unlink()
                logger.info(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {e}")

remove_all_files_starting_with(
    directory=cfg.Config().get("data.labels_dir", "data/labels"),
    prefix="KESHAEUW_VS_DRUTUTT"
)
remove_all_files_starting_with(
    directory=cfg.Config().get("data.images_dir", "data/images"),
    prefix="KESHAEUW_VS_DRUTUTT"
)