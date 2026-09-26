from typing import Optional

import torch

from app.ml.aasist_model import AASISTModel


class AudioModel:
    """
    Main audio deepfake detection model.

    Currently:
        AASIST-L

    Future:
        AASIST-L + RawNet2 fusion
    """

    def __init__(self):
        self.aasist: Optional[AASISTModel] = None

    def load(self):
        if self.aasist is None:
            self.aasist = AASISTModel()

    def predict(self, waveform):
        self.load()

        result = self.aasist.predict(waveform)

        return {
            "model": "AASIST-L",
            **result,
        }


audio_model = AudioModel()