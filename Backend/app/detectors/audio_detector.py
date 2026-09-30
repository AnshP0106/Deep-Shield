from pathlib import Path
import sys

import librosa
import numpy as np

from app.ml.deepfake_audio_model import DeepfakeAudioModel


# ============================================================
# CONFIG
# ============================================================

_audio_model = None


# ============================================================
# MODEL
# ============================================================

def get_audio_model():

    global _audio_model

    if _audio_model is None:
        _audio_model = DeepfakeAudioModel()

    return _audio_model


# ============================================================
# BASIC AUDIO FEATURES
# ============================================================

def calculate_audio_features(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    # Load audio using librosa.
    # This is also consistent with the model's
    # preprocessing pipeline.

    audio, sample_rate = librosa.load(
        str(file_path)
    )

    if len(audio) == 0:
        raise ValueError(
            f"Audio file is empty: {file_path}"
        )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = len(audio) / sample_rate

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    # --------------------------------------------------------
    # Zero crossing rate
    # --------------------------------------------------------

    zero_crossings = np.sum(
        np.abs(
            np.diff(
                np.sign(audio)
            )
        ) > 0
    )

    zcr = (
        zero_crossings
        / max(len(audio), 1)
    )

    return {
        "duration_seconds": float(
            duration
        ),
        "sample_rate": int(
            sample_rate
        ),
        "num_samples": int(
            len(audio)
        ),
        "rms": float(
            rms
        ),
        "zero_crossing_rate": float(
            zcr
        ),
    }


# ============================================================
# DEEPFAKE DETECTION
# ============================================================

def run_audio_detection(file_path):

    model = get_audio_model()

    result = model.predict(
        file_path
    )

    return result


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_audio(file_path):

    file_path = Path(file_path)

    print("=" * 60)
    print("AUDIO DEEPFAKE ANALYSIS")
    print("=" * 60)

    print(
        f"file: {file_path}"
    )

    print()

    try:

        # ----------------------------------------------------
        # Basic audio information
        # ----------------------------------------------------

        features = calculate_audio_features(
            file_path
        )

        print("AUDIO FEATURES")
        print("-" * 60)

        for key, value in features.items():

            print(
                f"{key}: {value}"
            )

        print()

        # ----------------------------------------------------
        # Deepfake model
        # ----------------------------------------------------

        print(
            "RUNNING DEEPFAKE AUDIO MODEL"
        )

        print("-" * 60)

        result = run_audio_detection(
            file_path
        )

        print()

        print(
            "MODEL RESULT"
        )

        print("-" * 60)

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

        # ----------------------------------------------------
        # Return structured result
        # ----------------------------------------------------

        return {

            "file": str(
                file_path
            ),

            "audio": features,

            "classification": result[
                "classification"
            ],

            "confidence": result[
                "confidence"
            ],

            "ai_probability": result[
                "ai_probability"
            ],

            "human_probability": result[
                "human_probability"
            ],

            "detector_status": "success",
        }

    except Exception as e:

        print("=" * 60)
        print(
            "AUDIO ANALYSIS FAILED"
        )
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 60)

        return {

            "file": str(
                file_path
            ),

            "detector_status": "failed",

            "error": str(e),
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python -m app.detectors.audio_detector "
            "uploads/audio/test.wav"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    analyze_audio(
        audio_path
    )