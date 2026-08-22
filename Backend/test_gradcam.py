from pathlib import Path

import cv2
import torch

from app.managers.model_manager import model_manager
from app.ml.preprocessing import preprocess_image
from app.ml.gradcam import GradCAM


# =========================================================
# 1. LOAD MODEL
# =========================================================

model_manager.load_models()

model = (
    model_manager
    .get_image_model()
)

device = (
    model_manager.device
)

print(
    "Model loaded:",
    type(model)
)

print(
    "Device:",
    device
)


# =========================================================
# 2. SELECT TARGET LAYER
# =========================================================

target_layer = model.model.conv4

print(
    "Grad-CAM target layer:",
    target_layer
)


# =========================================================
# 3. CREATE GRAD-CAM
# =========================================================

gradcam = GradCAM(
    model=model,
    target_layer=target_layer
)


# =========================================================
# 4. LOAD TEST IMAGE
# =========================================================

image_path = Path(
    "test_image.png"
)

image = cv2.imread(
    str(image_path)
)

if image is None:

    raise ValueError(
        f"Unable to read image: {image_path}"
    )


# =========================================================
# 5. CONVERT BGR → RGB
# =========================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 6. PREPROCESS
# =========================================================

from PIL import Image

pil_image = Image.fromarray(
    image_rgb
)

input_tensor = (
    preprocess_image(
        pil_image
    )
    .to(device)
)


# =========================================================
# 7. GET MODEL PREDICTION
# =========================================================

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

prediction_class = int(
    torch.argmax(
        probabilities,
        dim=1
    ).item()
)

confidence = float(
    probabilities[
        0,
        prediction_class
    ].item()
)


print(
    "Predicted class:",
    prediction_class
)

print(
    "Confidence:",
    round(
        confidence * 100,
        2
    ),
    "%"
)


# =========================================================
# 8. GENERATE GRAD-CAM
# =========================================================

cam = gradcam.generate(
    input_tensor=input_tensor,
    target_class=prediction_class
)


print(
    "Grad-CAM generated successfully!"
)

print(
    "CAM shape:",
    cam.shape
)


# =========================================================
# 9. CREATE OVERLAY
# =========================================================

overlay = gradcam.overlay(
    image=image,
    cam=cam,
    alpha=0.45
)


# =========================================================
# 10. SAVE RESULT
# =========================================================

output_path = (
    "debug_gradcam.jpg"
)

cv2.imwrite(
    output_path,
    overlay
)


print(
    "Grad-CAM saved to:",
    output_path
)


# =========================================================
# 11. CLEANUP
# =========================================================

gradcam.remove_hooks()

print(
    "Grad-CAM test completed."
)