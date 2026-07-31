from pathlib import Path


def analyze_media(file_path: Path, media_type: str) -> dict:
    print(f"Analyzing {media_type}: {file_path}")

    return {
        "status": "Analysis not implemented yet.",
        "media_type": media_type,
        "file_path": str(file_path)
    }