from PIL import Image

import torch
from torchvision import transforms


IMAGE_SIZE = 256


image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Prepares a face image for DeepfakeBench Xception inference.
    """

    image = image.convert("RGB")

    tensor = image_transform(image)

    tensor = tensor.unsqueeze(0)

    print("Input tensor shape:", tensor.shape)

    return tensor