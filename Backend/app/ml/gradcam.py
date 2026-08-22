import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = (
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_hook = (
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    # =====================================================
    # SAVE ACTIVATIONS
    # =====================================================

    def _save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # =====================================================
    # SAVE GRADIENTS
    # =====================================================

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # =====================================================
    # GENERATE GRAD-CAM
    # =====================================================

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int
    ):

        # -------------------------------------------------
        # Clear previous values
        # -------------------------------------------------

        self.activations = None
        self.gradients = None

        # -------------------------------------------------
        # Model must be in evaluation mode
        # -------------------------------------------------

        self.model.eval()

        # -------------------------------------------------
        # Clear gradients
        # -------------------------------------------------

        self.model.zero_grad()

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        output = self.model(
            input_tensor
        )

        # -------------------------------------------------
        # Select target class
        # -------------------------------------------------

        target_score = output[
            0,
            target_class
        ]

        # -------------------------------------------------
        # Backward pass
        # -------------------------------------------------

        target_score.backward()

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if (
            self.activations is None
            or self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM failed to capture "
                "activations or gradients."
            )

        # -------------------------------------------------
        # Remove batch dimension
        # -------------------------------------------------

        activations = (
            self.activations[0]
        )

        gradients = (
            self.gradients[0]
        )

        # -------------------------------------------------
        # Global average pooling of gradients
        # -------------------------------------------------

        weights = gradients.mean(
            dim=(1, 2)
        )

        # -------------------------------------------------
        # Weighted feature maps
        # -------------------------------------------------

        cam = torch.zeros(
            activations.shape[1:],
            dtype=activations.dtype,
            device=activations.device
        )

        for channel in range(
            activations.shape[0]
        ):

            cam += (
                weights[channel]
                * activations[channel]
            )

        # -------------------------------------------------
        # ReLU
        #
        # Keep only positive influence
        # -------------------------------------------------

        cam = torch.relu(
            cam
        )

        # -------------------------------------------------
        # Move to CPU
        # -------------------------------------------------

        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:

            cam /= max_value

        # -------------------------------------------------
        # Resize to input image size
        # -------------------------------------------------

        height = input_tensor.shape[2]
        width = input_tensor.shape[3]

        cam = cv2.resize(
            cam,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        return cam

    # =====================================================
    # CREATE HEATMAP
    # =====================================================

    def create_heatmap(
        self,
        cam: np.ndarray
    ):

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        return heatmap

    # =====================================================
    # OVERLAY HEATMAP
    # =====================================================

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.45
    ):

        # Resize CAM to original image
        # dimensions

        cam_resized = cv2.resize(
            cam,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap = self.create_heatmap(
            cam_resized
        )

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return overlay

    # =====================================================
    # CLEANUP
    # =====================================================

    def remove_hooks(self):

        self.forward_hook.remove()

        self.backward_hook.remove()