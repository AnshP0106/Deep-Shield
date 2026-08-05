from pathlib import Path

from PIL import Image


def load_image(file_path: Path) -> Image.Image:
    """
    Loads an image from disk.
    """

    return Image.open(file_path)