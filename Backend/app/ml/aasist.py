import os
import sys
import torch
import torch.nn as nn
import numpy as np


class AASISTWrapper:
    """
    Wrapper around the AASIST audio anti-spoofing model.

    Input:
        16 kHz mono waveform

    Output:
        spoof probability
        bona fide probability
    """

    SAMPLE_RATE = 16000
    MAX_SAMPLES = 64600

    def __init__(self, model_path: str, device: str = None):

        self.model_path = model_path

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print("=" * 60)
        print("Loading AASIST")
        print("=" * 60)
        print(f"Checkpoint: {self.model_path}")
        print(f"Device: {self.device}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"AASIST checkpoint not found: {self.model_path}"
            )

        self.model = self._load_model()

        print("✓ AASIST loaded")
        print("=" * 60)

    def _load_model(self):
        """
        Load AASIST.

        The official AASIST implementation contains the actual
        network architecture. This wrapper expects that implementation
        to be available in the Python path.
        """

        try:
            from app.ml.aasist_model import Model
        except ImportError as e:
            raise ImportError(
                "\nAASIST model architecture is missing.\n"
                "Create app/ml/aasist_model.py using the official "
                "AASIST architecture before running inference.\n"
            ) from e

        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 64],
                [64, 64]
            ],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }

        model = Model(model_config).to(self.device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]

            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=True)

        model.eval()

        return model

    @staticmethod
    def prepare_waveform(waveform: np.ndarray) -> np.ndarray:
        """
        Convert arbitrary-length audio into the
        64,600 samples expected by AASIST.
        """

        waveform = np.asarray(waveform, dtype=np.float32)

        # Convert stereo → mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Remove NaN / infinity
        waveform = np.nan_to_num(waveform)

        if len(waveform) == 0:
            raise ValueError("Audio waveform is empty.")

        # Exact AASIST evaluation behavior:
        # truncate long audio
        if len(waveform) >= AASISTWrapper.MAX_SAMPLES:
            waveform = waveform[:AASISTWrapper.MAX_SAMPLES]

        else:
            # Repeat shorter audio until target length
            repeats = (
                AASISTWrapper.MAX_SAMPLES // len(waveform)
            ) + 1

            waveform = np.tile(
                waveform,
                repeats
            )[:AASISTWrapper.MAX_SAMPLES]

        return waveform.astype(np.float32)

    def predict(self, waveform: np.ndarray):
        """
        Run AASIST inference.

        Returns:
            {
                "classification": "real" / "fake",
                "confidence": float,
                "real_probability": float,
                "fake_probability": float
            }
        """

        waveform = self.prepare_waveform(waveform)

        audio_tensor = torch.tensor(
            waveform,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            output = self.model(audio_tensor)

            # AASIST normally returns a 2-class output.
            if isinstance(output, tuple):
                output = output[0]

            probabilities = torch.softmax(
                output,
                dim=1
            )

        probabilities = probabilities.detach().cpu().numpy()[0]

        # AASIST:
        # class 1 = bona fide
        # class 0 = spoof
        fake_probability = float(probabilities[0])
        real_probability = float(probabilities[1])

        if real_probability >= fake_probability:
            classification = "real"
            confidence = real_probability
        else:
            classification = "fake"
            confidence = fake_probability

        return {
            "classification": classification,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability
        }