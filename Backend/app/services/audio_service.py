from pathlib import Path

from app.detectors.audio_detector import analyze_audio


def analyze_audio_file(file_path: str):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    return analyze_audio(str(path))