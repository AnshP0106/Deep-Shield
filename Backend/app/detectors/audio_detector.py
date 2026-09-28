from pathlib import Path
import sys
import wave

import numpy as np
import torch

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
# WAV LOADER
# ============================================================

def load_wav(file_path: Path):
    """
    Load an uncompressed WAV file without torchaudio/TorchCodec.

    Returns:
        waveform: torch.Tensor [1, samples]
        sample_rate: int
    """

    with wave.open(str(file_path), "rb") as wav:

        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        num_frames = wav.getnframes()

        raw_audio = wav.readframes(num_frames)

    # --------------------------------------------------------
    # Convert PCM → numpy
    # --------------------------------------------------------

    if sample_width == 1:

        audio = np.frombuffer(
            raw_audio,
            dtype=np.uint8
        ).astype(np.float32)

        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw_audio,
            dtype=np.int16
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 3:

        # 24-bit PCM
        raw = np.frombuffer(
            raw_audio,
            dtype=np.uint8
        ).reshape(-1, 3)

        audio = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )

        # Sign extension
        audio = np.where(
            audio & 0x800000,
            audio - 0x1000000,
            audio
        )

        audio = audio.astype(np.float32) / 8388608.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw_audio,
            dtype=np.int32
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width} bytes"
        )

    # --------------------------------------------------------
    # Stereo / multi-channel → mono
    # --------------------------------------------------------

    if num_channels > 1:

        audio = audio.reshape(
            -1,
            num_channels
        )

        audio = np.mean(
            audio,
            axis=1
        )

    # --------------------------------------------------------
    # Resample to 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        old_length = len(audio)

        new_length = int(
            old_length
            * TARGET_SAMPLE_RATE
            / sample_rate
        )

        old_indices = np.linspace(
            0,
            old_length - 1,
            old_length
        )

        new_indices = np.linspace(
            0,
            old_length - 1,
            new_length
        )

        audio = np.interp(
            new_indices,
            old_indices,
            audio
        ).astype(np.float32)

        sample_rate = TARGET_SAMPLE_RATE

    waveform = torch.from_numpy(
        audio.astype(np.float32)
    ).unsqueeze(0)

    return waveform, sample_rate


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(file_path):
    """
    Load audio without using torchaudio/TorchCodec.

    Currently supports WAV directly.

    Returns:
        waveform: torch.Tensor [1, samples]
        sample_rate: int
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    # --------------------------------------------------------
    # WAV
    # --------------------------------------------------------

    if extension == ".wav":

        return load_wav(file_path)

    # --------------------------------------------------------
    # Other formats
    # --------------------------------------------------------

    raise RuntimeError(
        f"Unsupported audio format: {extension}. "
        f"For now, use WAV files for AASIST testing."
    )


# ============================================================
# BASIC AUDIO FEATURES
# ============================================================

def calculate_audio_features(
    waveform,
    sample_rate
):

    audio = waveform.squeeze(0).numpy()

    # RMS
    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    # Zero crossing rate
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

    # Duration
    duration = (
        len(audio)
        / sample_rate
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
# AASIST INFERENCE
# ============================================================

def run_aasist(
    waveform,
    sample_rate
):

    model = get_aasist_model()

    # --------------------------------------------------------
    # Ensure mono
    # --------------------------------------------------------

    if waveform.ndim == 2:

        waveform = waveform.mean(
            dim=0
        )

    waveform = waveform.float()

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    waveform = waveform.unsqueeze(0)

    # --------------------------------------------------------
    # AASIST
    # --------------------------------------------------------

    with torch.no_grad():

        result = model.predict(
            waveform
        )

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

        print(
            "Audio loaded successfully"
        )

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

        return {
            "file": str(file_path),

            "audio": features,

            "classification": classification,

            "confidence": confidence,

            "spoof_probability":
                spoof_probability,

            "bonafide_probability":
                bonafide_probability,

            "detector_status":
                "success",
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

            "detector_status":
                "failed",

            "error": str(e),
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python -m "
            "app.detectors.audio_detector "
            "uploads/audio/test.wav"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    analyze_audio(
        audio_path
    )