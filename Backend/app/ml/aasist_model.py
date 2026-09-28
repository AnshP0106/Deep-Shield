from pathlib import Path

import torch

from app.ml.aasist_net import Model


class AASISTModel:
    """
    Wrapper around the official AASIST-L architecture.

    Audio is divided into multiple fixed-length chunks.
    AASIST analyzes every chunk independently and the
    probabilities are aggregated into one final result.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_dir = Path(__file__).resolve().parents[2]

        self.model_path = (
            base_dir
            / "models"
            / "weights"
            / "AASIST-L.pth"
        )

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

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        self.model = Model(self.config)

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Remove DataParallel prefix if present
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in checkpoint.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("AASIST-L loaded successfully.")
        print("=" * 60)

    # ========================================================
    # SINGLE CHUNK INFERENCE
    # ========================================================

    @torch.no_grad()
    def _predict_chunk(self, waveform):
        """
        Run AASIST on exactly one 64600-sample chunk.

        Returns:
            spoof_probability
            bonafide_probability
        """

        waveform = waveform.float()

        # Make sure waveform is 1D.
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        target_length = self.config["nb_samp"]

        # ----------------------------------------------------
        # Pad short chunks by repeating the waveform.
        # ----------------------------------------------------

        if waveform.numel() < target_length:

            if waveform.numel() == 0:
                raise ValueError(
                    "Audio chunk contains no samples."
                )

            repeats = (
                target_length // waveform.numel()
            ) + 1

            waveform = waveform.repeat(repeats)

        # ----------------------------------------------------
        # Trim to exactly target length.
        # ----------------------------------------------------

        waveform = waveform[:target_length]

        # ----------------------------------------------------
        # Add batch dimension.
        # Shape:
        #
        # [64600]
        #    ↓
        # [1, 64600]
        # ----------------------------------------------------

        waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        # ----------------------------------------------------
        # AASIST inference
        # ----------------------------------------------------

        _, output = self.model(waveform)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        # AASIST / ASVspoof convention:
        #
        # class 0 = spoof
        # class 1 = bonafide
        # ----------------------------------------------------

        spoof_probability = probabilities[
            0, 0
        ].item()

        bonafide_probability = probabilities[
            0, 1
        ].item()

        return (
            float(spoof_probability),
            float(bonafide_probability),
        )

    # ========================================================
    # MULTI-CHUNK PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(self, waveform):
        """
        Analyze the complete audio using multiple chunks.

        The audio is divided into 64600-sample sections.
        Every section is analyzed independently.

        The final result is the average probability
        across all analyzed chunks.
        """

        if not isinstance(
            waveform,
            torch.Tensor
        ):
            waveform = torch.tensor(
                waveform,
                dtype=torch.float32
            )

        waveform = waveform.float()

        # ----------------------------------------------------
        # Convert to mono / 1D
        # ----------------------------------------------------

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        waveform = waveform.cpu()

        total_samples = waveform.numel()

        if total_samples == 0:
            raise ValueError(
                "Audio contains no samples."
            )

        chunk_size = self.config["nb_samp"]

        # ----------------------------------------------------
        # Determine number of chunks
        # ----------------------------------------------------

        num_chunks = (
            total_samples + chunk_size - 1
        ) // chunk_size

        print()
        print("AASIST CHUNK ANALYSIS")
        print("-" * 60)

        print(
            f"Total samples: {total_samples}"
        )

        print(
            f"Chunk size: {chunk_size}"
        )

        print(
            f"Chunks: {num_chunks}"
        )

        print()

        spoof_scores = []
        bonafide_scores = []

        # ----------------------------------------------------
        # Analyze every chunk
        # ----------------------------------------------------

        for chunk_index in range(num_chunks):

            start = (
                chunk_index
                * chunk_size
            )

            end = min(
                start + chunk_size,
                total_samples
            )

            chunk = waveform[start:end]

            spoof_probability, bonafide_probability = (
                self._predict_chunk(chunk)
            )

            spoof_scores.append(
                spoof_probability
            )

            bonafide_scores.append(
                bonafide_probability
            )

            print(
                f"Chunk {chunk_index + 1}: "
                f"spoof={spoof_probability:.4f} "
                f"bonafide={bonafide_probability:.4f}"
            )

        # ====================================================
        # AGGREGATION
        # ====================================================

        average_spoof = (
            sum(spoof_scores)
            / len(spoof_scores)
        )

        average_bonafide = (
            sum(bonafide_scores)
            / len(bonafide_scores)
        )

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        if average_spoof >= average_bonafide:

            label = "spoof"

            confidence = average_spoof

        else:

            label = "bonafide"

            confidence = average_bonafide

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print()
        print("AASIST AGGREGATED RESULT")
        print("-" * 60)

        print(
            f"chunks_analyzed: {num_chunks}"
        )

        print(
            f"average_spoof_probability: "
            f"{average_spoof:.6f}"
        )

        print(
            f"average_bonafide_probability: "
            f"{average_bonafide:.6f}"
        )

        print(
            f"classification: {label}"
        )

        print(
            f"confidence: {confidence:.6f}"
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "classification": label,

            "confidence": round(
                float(confidence),
                4
            ),

            "spoof_probability": round(
                float(average_spoof),
                4
            ),

            "bonafide_probability": round(
                float(average_bonafide),
                4
            ),

            "chunks_analyzed": int(
                num_chunks
            ),

            "chunk_spoof_probabilities": [
                round(float(x), 4)
                for x in spoof_scores
            ],

            "chunk_bonafide_probabilities": [
                round(float(x), 4)
                for x in bonafide_scores
            ],
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m app.ml.aasist_model "
            "uploads/audio/test.wav"
        )

        sys.exit(1)

    import torchaudio

    audio_path = sys.argv[1]

    print()
    print("=" * 60)
    print("AASIST DIRECT TEST")
    print("=" * 60)

    waveform, sample_rate = torchaudio.load(
        audio_path
    )

    # --------------------------------------------------------
    # Convert stereo → mono
    # --------------------------------------------------------

    if waveform.ndim == 2 and waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    waveform = waveform.squeeze(0)

    # --------------------------------------------------------
    # Resample if required
    # --------------------------------------------------------

    target_sample_rate = 16000

    if sample_rate != target_sample_rate:

        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
        )

        waveform = resampler(
            waveform.unsqueeze(0)
        ).squeeze(0)

        sample_rate = target_sample_rate

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AASISTModel()

    result = model.predict(
        waveform
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )