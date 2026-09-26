from pathlib import Path
import sys

import numpy as np
import torch
import soundfile as sf
import torch.nn.functional as F

from app.ml.aasist_model import AASISTModel


# ============================================================
# CONFIG
# ============================================================

TARGET_SAMPLE_RATE = 16000

_aasist_model = None


# ============================================================
# MODEL
# ============================================================

def get_aasist_model():
    global _aasist_model

    if _aasist_model is None:
        _aasist_model = AASISTModel()

    return _aasist_model


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):
    """
    Load audio using SoundFile instead of torchaudio.

    Output:
        waveform: torch.Tensor [1, samples]
        sample_rate: int

    Processing:
        1. Load audio
        2. Convert stereo -> mono
        3. Convert to float32
        4. Resample to 16 kHz
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    try:

        # ----------------------------------------------------
        # Load audio using SoundFile
        # ----------------------------------------------------

        audio, sample_rate = sf.read(
            str(file_path),
            dtype="float32",
            always_2d=True
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to load audio file: {file_path}\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    # --------------------------------------------------------
    # SoundFile returns:
    #
    # [samples, channels]
    #
    # Convert to:
    #
    # [channels, samples]
    # --------------------------------------------------------

    waveform = torch.from_numpy(
        audio.T.copy()
    )

    # --------------------------------------------------------
    # Convert stereo/multi-channel -> mono
    # --------------------------------------------------------

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # --------------------------------------------------------
    # Ensure float32
    # --------------------------------------------------------

    waveform = waveform.float()

    # --------------------------------------------------------
    # Resample -> 16 kHz
    #
    # We use torch interpolation instead of
    # torchaudio.transforms.Resample.
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        original_length = waveform.shape[-1]

        new_length = int(
            original_length
            * TARGET_SAMPLE_RATE
            / sample_rate
        )

        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False
        ).squeeze(0)

        sample_rate = TARGET_SAMPLE_RATE

    return waveform, sample_rate


# ============================================================
# BASIC AUDIO FEATURES
# ============================================================

def calculate_audio_features(waveform, sample_rate):
    """
    Calculate basic audio statistics.

    These features are currently metadata only.

    AASIST remains the actual deepfake classifier.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    # --------------------------------------------------------
    # Zero Crossing Rate
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

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = (
        len(audio)
        / sample_rate
    )

    return {
        "duration_seconds": float(duration),
        "sample_rate": int(sample_rate),
        "num_samples": int(len(audio)),
        "rms": float(rms),
        "zero_crossing_rate": float(zcr),
    }


# ============================================================
# AASIST INFERENCE
# ============================================================

def run_aasist(waveform, sample_rate):
    """
    Run AASIST-L inference.

    AASISTModel.predict() handles the model-specific
    preprocessing internally.

    Returns:
        classification
        confidence
        spoof_probability
        bonafide_probability
    """

    model = get_aasist_model()

    # Make sure waveform is mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = waveform.float()

    # Add batch dimension
    waveform = waveform.unsqueeze(0)

    # AASISTModel.predict() accepts waveform only
    with torch.no_grad():
        result = model.predict(waveform)

    return result


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_audio(file_path):

    print("=" * 60)
    print("AUDIO DEEPFAKE ANALYSIS")
    print("=" * 60)

    print(f"file: {file_path}")
    print()

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        waveform, sample_rate = load_audio(
            file_path
        )

        print("Audio loaded successfully")
        print(
            f"sample_rate: {sample_rate}"
        )
        print(
            f"samples: {waveform.shape[-1]}"
        )
        print()

        # ----------------------------------------------------
        # Basic features
        # ----------------------------------------------------

        features = calculate_audio_features(
            waveform,
            sample_rate
        )

        print("AUDIO FEATURES")
        print("-" * 60)

        for key, value in features.items():

            print(
                f"{key}: {value}"
            )

        print()

        # ----------------------------------------------------
        # AASIST
        # ----------------------------------------------------

        print("RUNNING AASIST-L")
        print("-" * 60)

        result = run_aasist(
            waveform,
            sample_rate
        )

        print()

        # ----------------------------------------------------
        # AASIST RESULT
        # ----------------------------------------------------

        print("AASIST RESULT")
        print("-" * 60)

        classification = result.get(
            "classification"
        )

        confidence = result.get(
            "confidence"
        )

        spoof_probability = result.get(
            "spoof_probability"
        )

        bonafide_probability = result.get(
            "bonafide_probability"
        )

        print(
            f"classification: "
            f"{classification}"
        )

        if confidence is not None:

            print(
                f"confidence: "
                f"{confidence:.6f}"
            )

        if spoof_probability is not None:

            print(
                f"spoof_probability: "
                f"{spoof_probability:.6f}"
            )

        if bonafide_probability is not None:

            print(
                f"bonafide_probability: "
                f"{bonafide_probability:.6f}"
            )

        print("=" * 60)

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {
            "file": str(file_path),

            "audio": features,

            "classification": classification,

            "confidence": confidence,

            "spoof_probability": spoof_probability,

            "bonafide_probability": bonafide_probability,

            "detector_status": "success",
        }

    except Exception as e:

        print("=" * 60)
        print("AUDIO ANALYSIS FAILED")
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 60)

        return {
            "file": str(file_path),

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

    analyze_audio(audio_path)