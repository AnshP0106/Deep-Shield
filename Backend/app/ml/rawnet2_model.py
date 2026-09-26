import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SincConv(nn.Module):
    """
    Sinc-based convolution used by RawNet2.
    """

    def __init__(
        self,
        out_channels=128,
        kernel_size=129,
        sample_rate=16000,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Fixed sinc filterbank
        self.conv = nn.Conv1d(
            1,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.shortcut = None

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )

    def forward(self, x):

        identity = x

        out = self.bn1(x)
        out = F.leaky_relu(out, negative_slope=0.3)

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.leaky_relu(out, negative_slope=0.3)

        out = self.conv2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity

        return out


class RawNet2(nn.Module):
    """
    RawNet2-style architecture for audio spoof detection.

    Output:
        [spoof, bonafide]
    """

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Sinc filterbank
        # --------------------------------------------------

        self.sinc = SincConv(
            out_channels=128,
            kernel_size=129,
            sample_rate=16000,
        )

        self.bn0 = nn.BatchNorm1d(128)

        # --------------------------------------------------
        # Residual stages
        # --------------------------------------------------

        self.stage1 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(128, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        # --------------------------------------------------
        # GRU
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=1024,
            num_layers=3,
            batch_first=True,
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.fc = nn.Linear(
            1024,
            2,
        )

    def forward(self, waveform):

        # waveform:
        # [batch, samples]

        x = waveform.unsqueeze(1)

        # [B, 1, T]
        x = self.sinc(x)

        # [B, 128, T]
        x = self.bn0(x)

        x = F.leaky_relu(
            x,
            negative_slope=0.3,
        )

        x = self.stage1(x)

        x = self.stage2(x)

        # [B, C, T]
        x = x.transpose(1, 2)

        # [B, T, C]
        x, _ = self.gru(x)

        # Last temporal feature
        x = x[:, -1, :]

        logits = self.fc(x)

        return logits


class RawNet2Model:

    def __init__(
        self,
        weights_path: str = "models/weights/rawnet2.pth",
        device: str | None = None,
    ):

        self.device = (
            device
            if device is not None
            else (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )

        self.sample_rate = 16000
        self.target_samples = 64000

        print("=" * 60)
        print("Loading RawNet2")
        print("=" * 60)

        print("Device:", self.device)
        print("Weights:", weights_path)

        self.model = RawNet2()

        try:

            checkpoint = torch.load(
                weights_path,
                map_location="cpu",
            )

            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

            self.model.load_state_dict(
                checkpoint,
                strict=True,
            )

            print("✓ RawNet2 weights loaded")

        except Exception as e:

            print("⚠ Could not load RawNet2 checkpoint")
            print("Reason:", e)

            raise

        self.model.to(self.device)
        self.model.eval()

        print("✓ RawNet2 ready")
        print("=" * 60)

    def _load_audio(self, audio_path):

        waveform, sample_rate = torchaudio.load(
            audio_path
        )

        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample
        if sample_rate != self.sample_rate:

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )

            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        return waveform

    def _prepare_audio(self, waveform):

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        waveform = waveform.float()

        max_value = waveform.abs().max()

        if max_value > 0:
            waveform = waveform / max_value

        # --------------------------------------------------
        # Exactly 4 seconds
        # --------------------------------------------------

        if waveform.shape[0] < self.target_samples:

            padding = (
                self.target_samples
                - waveform.shape[0]
            )

            waveform = F.pad(
                waveform,
                (0, padding),
            )

        else:

            waveform = waveform[
                : self.target_samples
            ]

        return waveform

    @torch.no_grad()
    def predict(self, audio_path):

        waveform = self._load_audio(
            audio_path
        )

        waveform = self._prepare_audio(
            waveform
        )

        waveform = waveform.unsqueeze(0)

        waveform = waveform.to(
            self.device
        )

        logits = self.model(
            waveform
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        spoof_probability = probabilities[
            0, 0
        ].item()

        bonafide_probability = probabilities[
            0, 1
        ].item()

        if spoof_probability >= bonafide_probability:
            prediction = "FAKE"
        else:
            prediction = "REAL"

        return {
            "prediction": prediction,
            "fake_probability": round(
                spoof_probability,
                4,
            ),
            "real_probability": round(
                bonafide_probability,
                4,
            ),
        }


# ------------------------------------------------------
# TEST
# ------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m app.ml.rawnet2_model audio.wav"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    detector = RawNet2Model()

    result = detector.predict(
        audio_path
    )

    print()
    print("=" * 60)
    print("AUDIO DETECTION RESULT")
    print("=" * 60)

    print(
        "Prediction:",
        result["prediction"],
    )

    print(
        "Fake probability:",
        result["fake_probability"],
    )

    print(
        "Real probability:",
        result["real_probability"],
    )

    print("=" * 60)