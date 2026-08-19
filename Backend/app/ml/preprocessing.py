from PIL import Image

import torch
from torchvision import transforms


IMAGE_SIZE = 256


image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Prepares an image for Xception inference.
    """

    image = image.convert("RGB")

    tensor = image_transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor