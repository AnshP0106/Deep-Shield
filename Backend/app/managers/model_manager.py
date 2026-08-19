from pathlib import Path

import torch

from app.ml.deepfake_model import DeepfakeXception


class ModelManager:

    def __init__(self):

        self.image_model = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_models(self):

        print(f"Using device: {self.device}")
        print("Loading Deepfake Xception model...")

        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "weights"
            / "xception_best.pth"
        )

        print(f"Model path: {model_path}")

        # Create model architecture
        self.image_model = DeepfakeXception()

        # Load checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Some checkpoints contain a state_dict wrapper
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # --------------------------------------------------
        # Remove "backbone." prefix from DeepfakeBench weights
        # --------------------------------------------------

        cleaned_checkpoint = {}

        for key, weights in checkpoint.items():

            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "", 1)
                cleaned_checkpoint[new_key] = weights

        # --------------------------------------------------
        # Fix pointwise convolution weight dimensions
        # --------------------------------------------------

        for name, weights in list(cleaned_checkpoint.items()):

            if "pointwise" in name and weights.ndim == 2:
                cleaned_checkpoint[name] = (
                    weights.unsqueeze(-1).unsqueeze(-1)
                )

        # --------------------------------------------------
        # Load weights
        # --------------------------------------------------

        missing_keys, unexpected_keys = (
            self.image_model.model.load_state_dict(
                cleaned_checkpoint,
                strict=False
            )
        )

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        # Move model to device
        self.image_model.to(self.device)

        # Evaluation mode
        self.image_model.eval()

        print("Deepfake Xception model loaded successfully!")

    def get_image_model(self):

        return self.image_model


model_manager = ModelManager()