from PIL import Image

def extract_image_metadata(file_path: str):
    with Image.open(file_path) as image:
        data = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode
        }

        return data