import cv2

from app.detectors.yunet_detector import yunet_detector


image_path = "test_image.png"

image = cv2.imread(image_path)

if image is None:
    raise ValueError(
        "Could not read test_image.png"
    )

faces = yunet_detector.detect(image)

print(
    "YuNet faces detected:",
    len(faces)
)

for i, face in enumerate(faces):

    x, y, w, h = face[:4]

    confidence = face[-1]

    print(
        f"Face {i + 1}: "
        f"x={x:.0f}, "
        f"y={y:.0f}, "
        f"w={w:.0f}, "
        f"h={h:.0f}, "
        f"confidence={confidence:.3f}"
    )

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imwrite(
    "debug_yunet.jpg",
    image
)

print(
    "Saved result to debug_yunet.jpg"
)


#Image Type | Confidence | Faces Detected |
# 1. Nigga | 0.945 | 1
# 2. AI | |