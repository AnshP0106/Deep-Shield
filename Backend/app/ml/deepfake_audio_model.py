from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


class DeepfakeAudioModel:
    """
    Wrapper for the deepfake-audio-detector model.

    Model input:
        Mel spectrogram
        91 mel bands
        150 time steps

    Model output:
        [AI probability, Human probability]
    """

    def __init__(self):
        # ----------------------------------------------------
        # Device/model path
        # ----------------------------------------------------

        base_dir = Path(__file__).resolve().parents[2]

        self.model_path = (
            base_dir
            / "models"
            / "weights"
            / "model-1.h5"
        )

        self.n_mels = 91
        self.max_time_steps = 150

        print("=" * 60)
        print("Loading Deepfake Audio Model")
        print("=" * 60)

        print(f"Checkpoint: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Deepfake audio model not found: "
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        print("Deepfake audio model loaded successfully.")
        print("=" * 60)

    # ========================================================
    # PREPROCESSING
    # ========================================================

    def preprocess_audio(self, audio_path):
        """
        Reproduce the preprocessing used by the
        original deepfake-audio-detector repository.

        Returns:
            numpy array with shape:
            [1, 91, 150]
        """

        audio_path = str(audio_path)

        # ----------------------------------------------------
        # Load audio using librosa
        # ----------------------------------------------------

        audio, sample_rate = librosa.load(
            audio_path
        )

        if len(audio) == 0:
            raise ValueError(
                f"Audio file is empty or could not be loaded: "
                f"{audio_path}"
            )

        # ----------------------------------------------------
        # Mel spectrogram
        # ----------------------------------------------------

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            n_mels=self.n_mels,
        )

        # ----------------------------------------------------
        # Convert power spectrogram → dB
        # ----------------------------------------------------

        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max,
        )

        # ----------------------------------------------------
        # Force exactly 150 time steps
        # ----------------------------------------------------

        current_time_steps = mel_spectrogram.shape[1]

        if current_time_steps < self.max_time_steps:

            mel_spectrogram = np.pad(
                mel_spectrogram,
                (
                    (0, 0),
                    (
                        0,
                        self.max_time_steps
                        - current_time_steps,
                    ),
                ),
                mode="constant",
            )

        else:

            mel_spectrogram = (
                mel_spectrogram[
                    :,
                    :self.max_time_steps
                ]
            )

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        mel_spectrogram = (
            mel_spectrogram.astype(np.float32)
        )

        # ----------------------------------------------------
        # Add batch dimension
        #
        # [91, 150]
        #      ↓
        # [1, 91, 150]
        # ----------------------------------------------------

        return np.expand_dims(
            mel_spectrogram,
            axis=0,
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, audio_path):
        """
        Analyze one audio file.

        Returns:
            classification
            confidence
            ai_probability
            human_probability
        """

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        input_data = self.preprocess_audio(
            audio_path
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        predictions = self.model.predict(
            input_data,
            verbose=0,
        )

        probabilities = predictions[0]

        # Original repository convention:
        #
        # p[0] = AI
        # p[1] = Human

        ai_probability = float(
            probabilities[0]
        )

        human_probability = float(
            probabilities[1]
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if ai_probability >= human_probability:

            classification = "AI"
            confidence = ai_probability

        else:

            classification = "HUMAN"
            confidence = human_probability

        return {
            "classification": classification,
            "confidence": round(
                confidence,
                4,
            ),
            "ai_probability": round(
                ai_probability,
                4,
            ),
            "human_probability": round(
                human_probability,
                4,
            ),
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python -m app.ml.deepfake_audio_model "
            "uploads/audio/test.wav"
        )

        raise SystemExit(1)

    audio_path = sys.argv[1]

    model = DeepfakeAudioModel()

    result = model.predict(
        audio_path
    )

    print()
    print("=" * 60)
    print("DEEPFAKE AUDIO RESULT")
    print("=" * 60)

    print(
        f"classification: "
        f"{result['classification']}"
    )

    print(
        f"confidence: "
        f"{result['confidence']:.4f}"
    )

    print(
        f"ai_probability: "
        f"{result['ai_probability']:.4f}"
    )

    print(
        f"human_probability: "
        f"{result['human_probability']:.4f}"
    )

    print("=" * 60)