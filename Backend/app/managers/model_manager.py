import torch
import timm


class ModelManager:
    """
    Responsible for loading and providing AI models.
    """

    def __init__(self):
        self.image_model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_models(self):
        print(f"Using device: {self.device}")

        print("Loading Xception model...")

        self.image_model = timm.create_model(
            "xception",
            pretrained=True
        )

        self.image_model = self.image_model.to(self.device)
        self.image_model.eval()

        print("Xception model loaded successfully!")

    def get_image_model(self):
        return self.image_model

model_manager = ModelManager()