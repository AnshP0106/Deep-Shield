import sys
from pathlib import Path
import cv2

from app.detectors.scrfd_detector import scrfd_detector


def test_detector(image_path: str = "test.jpg"):
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"Test image not found at: {img_file.resolve()}")
        return

    image = cv2.imread(str(img_file))
    if image is None:
        print(f"Failed to read image at: {img_file.resolve()}")
        return

    print("========================================")
    print("TESTING SCRFD FACE DETECTOR")
    print("========================================")
    print(f"Image Path: {img_file.name}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")

    detections = scrfd_detector.detect_faces(image)

    print(f"Faces Detected: {len(detections)}")
    print("========================================")

    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        print(f"Face {idx}: bbox=[x={x}, y={y}, w={w}, h={h}], confidence={conf * 100:.2f}%")

    print("========================================")


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    test_detector(test_img)