import torch
import torch.nn as nn

from app.ml.xception import Xception


class DeepfakeXception(nn.Module):

    def __init__(self):
        super().__init__()

        xception_config = {
            "num_classes": 2,
            "mode": "original",
            "inc": 3,
            "dropout": False
        }

        self.model = Xception(xception_config)

    def forward(self, x):
        output, _ = self.model(x)

        return output