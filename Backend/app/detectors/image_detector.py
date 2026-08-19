from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.managers.model_manager import model_manager


class ImageDetector:

    def __init__(self):
        self.device = model_manager.device

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, file_path: Path):

        # Load image
        image = Image.open(file_path).convert("RGB")

        # Preprocess image
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)

        # Get model
        model = model_manager.get_image_model()

        # Inference
        with torch.no_grad():
            output = model(image_tensor)

            probabilities = torch.softmax(output, dim=1)

        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get confidence
        confidence = probabilities[0][predicted_class].item()

        # DeepfakeBench convention:
        # 0 = real
        # 1 = fake
        label = "FAKE" if predicted_class == 1 else "REAL"

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "real_probability": round(
                probabilities[0][0].item() * 100, 2
            ),
            "fake_probability": round(
                probabilities[0][1].item() * 100, 2
            )
        }


image_detector = ImageDetector()