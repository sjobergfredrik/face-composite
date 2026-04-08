"""
Face masking pipeline ported from FaceFusion's approach.
Creates occlusion-aware, region-aware masks for clean face swap blending.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

# Face parser class indices (BiSeNet/face-parsing)
FACE_REGION_INDICES = {
    "skin": 1,
    "left_eyebrow": 2,
    "right_eyebrow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper_lip": 12,
    "lower_lip": 13,
}

# Default regions to include in mask
DEFAULT_FACE_REGIONS = [1, 2, 3, 4, 5, 10, 11, 12, 13]

# ImageNet normalization constants (for BiSeNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceMasker:
    def __init__(self, model_dir: str = MODEL_DIR):
        self._occluder = None
        self._parser = None
        self._model_dir = model_dir

    def _load_occluder(self):
        if self._occluder is not None:
            return
        path = os.path.join(self._model_dir, "xseg_1.onnx")
        if not os.path.exists(path):
            print(f"[FaceMasker] xseg_1.onnx not found at {path}, occlusion masking disabled")
            return
        self._occluder = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._occluder_input = self._occluder.get_inputs()[0].name
        print(f"[FaceMasker] Loaded xseg_1 occluder")

    def _load_parser(self):
        if self._parser is not None:
            return
        path = os.path.join(self._model_dir, "bisenet_resnet_34.onnx")
        if not os.path.exists(path):
            print(f"[FaceMasker] bisenet_resnet_34.onnx not found at {path}, region masking disabled")
            return
        self._parser = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._parser_input = self._parser.get_inputs()[0].name
        print(f"[FaceMasker] Loaded bisenet_resnet_34 parser")

    def create_box_mask(
        self,
        crop_size: tuple[int, int],
        blur: float = 0.3,
        padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Create a padded box mask with feathered edges.
        padding: (top, right, bottom, left) as fractions of crop size.
        """
        h, w = crop_size
        mask = np.ones((h, w), dtype=np.float32)

        # Apply padding (zero out edges)
        top = int(h * padding[0])
        right = int(w * padding[1])
        bottom = int(h * padding[2])
        left = int(w * padding[3])

        if top > 0:
            mask[:top, :] = 0
        if bottom > 0:
            mask[h - bottom:, :] = 0
        if left > 0:
            mask[:, :left] = 0
        if right > 0:
            mask[:, w - right:] = 0

        # Gaussian blur for soft edges
        blur_size = int(min(h, w) * 0.5 * blur)
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size > 1:
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask

    def create_occlusion_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Detect occluding elements (hair, hands, glasses) using xseg model.
        Input: the ORIGINAL warped target crop (before swap), BGR uint8.
        Returns: mask where 1 = face (safe to paste), 0 = occluded (preserve original).
        """
        self._load_occluder()
        if self._occluder is None:
            # Fallback: full ones mask (no occlusion detection)
            return np.ones(crop_bgr.shape[:2], dtype=np.float32)

        h, w = crop_bgr.shape[:2]

        # Resize to model input size (256x256)
        input_img = cv2.resize(crop_bgr, (256, 256))
        # BGR to RGB, normalize to [0, 1]
        input_rgb = input_img[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_rgb, axis=0)

        try:
            outputs = self._occluder.run(None, {self._occluder_input: input_tensor})
            mask = self._normalize_single_mask_output(outputs[0])
        except Exception as exc:
            print(f"[FaceMasker] Occlusion mask failed, using fallback mask: {exc}")
            return np.ones(crop_bgr.shape[:2], dtype=np.float32)

        # Resize back to crop size
        mask = cv2.resize(mask, (w, h))

        # Post-process: smooth and threshold (FaceFusion's formula)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5)
        mask = np.clip(mask, 0.5, 1.0)
        mask = (mask - 0.5) * 2.0

        return mask.clip(0, 1).astype(np.float32)

    def create_region_mask(
        self,
        crop_bgr: np.ndarray,
        regions: list[int] | None = None,
    ) -> np.ndarray:
        """
        Semantic face parsing to create region-aware mask.
        Only includes face parts (skin, eyes, nose, mouth), excludes hair/background/neck.
        Input: the ORIGINAL warped target crop (before swap), BGR uint8.
        """
        self._load_parser()
        if self._parser is None:
            return np.ones(crop_bgr.shape[:2], dtype=np.float32)

        if regions is None:
            regions = DEFAULT_FACE_REGIONS

        h, w = crop_bgr.shape[:2]

        # Resize to 512x512 (BiSeNet input size)
        input_img = cv2.resize(crop_bgr, (512, 512))
        # BGR to RGB, float32
        input_rgb = input_img[:, :, ::-1].astype(np.float32) / 255.0
        # ImageNet normalization
        input_norm = (input_rgb - IMAGENET_MEAN) / IMAGENET_STD
        # HWC to NCHW
        input_tensor = np.expand_dims(input_norm.transpose(2, 0, 1), axis=0)

        try:
            outputs = self._parser.run(None, {self._parser_input: input_tensor})
            parsing = outputs[0][0]  # (num_classes, 512, 512)
        except Exception as exc:
            print(f"[FaceMasker] Region mask failed, using fallback mask: {exc}")
            return np.ones(crop_bgr.shape[:2], dtype=np.float32)

        # Argmax to get class per pixel
        class_map = np.argmax(parsing, axis=0)  # (512, 512)

        # Create binary mask from selected face regions
        mask = np.isin(class_map, regions).astype(np.float32)

        # Resize back to crop size
        mask = cv2.resize(mask, (w, h))

        # Post-process: smooth (same formula as occlusion)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5)
        mask = np.clip(mask, 0.5, 1.0)
        mask = (mask - 0.5) * 2.0

        return mask.clip(0, 1).astype(np.float32)

    def create_composite_mask(
        self,
        original_crop_bgr: np.ndarray,
        blur: float = 0.3,
    ) -> np.ndarray:
        """
        Create the final composite mask by combining all three mask types.
        Uses np.minimum.reduce — any single mask can veto a pixel.
        """
        h, w = original_crop_bgr.shape[:2]

        masks = [
            self.create_box_mask((h, w), blur=blur),
            self.create_occlusion_mask(original_crop_bgr),
            self.create_region_mask(original_crop_bgr),
        ]

        composite = np.minimum.reduce(masks)
        return composite.clip(0, 1)

    def _normalize_single_mask_output(self, output: np.ndarray) -> np.ndarray:
        """Normalize ONNX mask outputs to a 2D float mask."""
        mask = np.asarray(output)

        if mask.ndim == 4:
            mask = mask[0]

        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask.mean(axis=0)

        if mask.ndim == 1:
            side = int(np.sqrt(mask.shape[0]))
            mask = mask.reshape(side, side)

        return mask.astype(np.float32)
