from pathlib import Path

import torch

from app.ml.aasist_net import Model


class AASISTModel:
    """
    Wrapper around the official AASIST-L architecture.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_dir = Path(__file__).resolve().parents[2]
        self.model_path = base_dir / "models" / "weights" / "AASIST-L.pth"

        self.config = {
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 24],
                [24, 24],
            ],
            "gat_dims": [24, 32],
            "pool_ratios": [0.4, 0.5, 0.7, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        print("=" * 60)
        print("Loading AASIST-L")
        print("=" * 60)
        print(f"Checkpoint: {self.model_path}")
        print(f"Device: {self.device}")

        self.model = Model(self.config)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        # Handle different checkpoint formats.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]

        # Remove possible "module." prefix.
        cleaned_state_dict = {}

        for key, value in checkpoint.items():
            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("AASIST-L loaded successfully.")
        print("=" * 60)

    @torch.no_grad()
    def predict(self, waveform):
        """
        waveform:
            torch.Tensor with shape [samples]

        Returns:
            label
            confidence
            probabilities
        """

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        waveform = waveform.float()

        # Make sure waveform is one-dimensional.
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        waveform = waveform.to(self.device)

        # AASIST-L expects 64600 samples.
        target_length = self.config["nb_samp"]

        if waveform.numel() < target_length:
            repeats = (target_length // waveform.numel()) + 1

            waveform = waveform.repeat(repeats)

            waveform = waveform[:target_length]

        else:
            waveform = waveform[:target_length]

        waveform = waveform.unsqueeze(0)

        _, output = self.model(waveform)

        probabilities = torch.softmax(output, dim=1)

        spoof_probability = probabilities[0, 0].item()
        bonafide_probability = probabilities[0, 1].item()

        # AASIST/ASVspoof convention:
        # class 0 = spoof
        # class 1 = bonafide
        if spoof_probability >= bonafide_probability:
            label = "spoof"
            confidence = spoof_probability
        else:
            label = "bonafide"
            confidence = bonafide_probability

        return {
            "classification": label,
            "confidence": round(float(confidence), 4),
            "spoof_probability": round(float(spoof_probability), 4),
            "bonafide_probability": round(float(bonafide_probability), 4),
        }