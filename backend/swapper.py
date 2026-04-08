import os
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

from face_masker import FaceMasker

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

# Warp template (arcface_128) — used by both inswapper and hyperswap
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32)


class FaceSwapper:
    def __init__(self):
        os.environ["INSIGHTFACE_HOME"] = MODEL_DIR
        self.app = FaceAnalysis(name="buffalo_l", root=MODEL_DIR)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load inswapper_128
        inswapper_path = os.path.join(MODEL_DIR, "inswapper_128.onnx")
        if os.path.exists(inswapper_path):
            self.inswapper = insightface.model_zoo.get_model(inswapper_path)
        else:
            self.inswapper = None

        # Load hyperswap_1c_256
        hyperswap_path = os.path.join(MODEL_DIR, "hyperswap_1c_256.onnx")
        if os.path.exists(hyperswap_path):
            self.hyperswap = ort.InferenceSession(
                hyperswap_path, providers=["CPUExecutionProvider"],
            )
            print(f"Loaded hyperswap_1c_256")
        else:
            self.hyperswap = None
            print(f"hyperswap_1c_256 not found at {hyperswap_path}")

        # Load face masker (occlusion + region parsing)
        self.face_masker = FaceMasker(MODEL_DIR)

    def detect_faces(self, img: np.ndarray) -> list[dict]:
        faces = self.app.get(img)
        faces = sorted(faces, key=lambda f: f.bbox[0])
        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "index": i,
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                },
                "confidence": float(face.det_score),
            })
        return results

    def swap(
        self,
        scene_img: np.ndarray,
        actor_img: np.ndarray,
        target_face_idx: int,
        model: str = "hyperswap_1c_256",
    ) -> np.ndarray:
        # Detect faces
        scene_faces = self.app.get(scene_img)
        scene_faces = sorted(scene_faces, key=lambda f: f.bbox[0])

        if target_face_idx >= len(scene_faces):
            raise ValueError(
                f"Face index {target_face_idx} out of range. "
                f"Detected {len(scene_faces)} faces."
            )

        actor_faces = self.app.get(actor_img)
        if not actor_faces:
            raise ValueError("No face detected in actor headshot.")
        actor_face = max(actor_faces, key=lambda f: float(f.det_score))
        target_face = scene_faces[target_face_idx]

        if model == "hyperswap_1c_256" and self.hyperswap:
            return self._hyperswap(scene_img, actor_face, target_face)
        elif self.inswapper:
            return self._inswapper(scene_img, actor_face, target_face)
        else:
            raise ValueError("No swap model available.")

    def _get_affine_matrix(self, landmark_5: np.ndarray, crop_size: tuple[int, int]):
        """Calculate affine warp matrix from 5 landmarks to template."""
        template = ARCFACE_128_TEMPLATE * np.array(crop_size, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(
            landmark_5.astype(np.float32), template,
            method=cv2.RANSAC, ransacReprojThreshold=100,
        )
        return matrix

    def _warp_crop(self, img: np.ndarray, matrix: np.ndarray, crop_size: tuple[int, int]):
        """Warp image region using affine matrix."""
        return cv2.warpAffine(
            img, matrix, crop_size,
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA,
        )

    def _inswapper(self, scene_img, actor_face, target_face) -> np.ndarray:
        """InSwapper 128×128 with multi-mask pipeline."""
        crop_size = (128, 128)

        # Run inswapper and use its alignment matrix when available.
        swap_result = self.inswapper.get(
            scene_img.copy(), target_face, actor_face, paste_back=False
        )
        matrix = None
        if isinstance(swap_result, tuple):
            swapped_crop, matrix = swap_result
        else:
            swapped_crop = swap_result

        # If inswapper returns full frame (paste_back=False not supported), fall back.
        if not isinstance(swapped_crop, np.ndarray) or swapped_crop.shape[:2] != crop_size:
            # Some insightface versions always paste back — use old path
            result = self.inswapper.get(
                scene_img.copy(), target_face, actor_face, paste_back=True
            )
            return result

        if matrix is None:
            matrix = self._get_affine_matrix(target_face.kps, crop_size)

        # Warp original target face for mask creation using the same alignment.
        original_crop = self._warp_crop(scene_img, matrix, crop_size)

        # Create composite mask from original crop
        mask = self.face_masker.create_composite_mask(original_crop, blur=0.3)

        # Paste back with multi-mask
        return self._paste_back_optimized(scene_img, swapped_crop, mask, matrix)

    def _hyperswap(self, scene_img, actor_face, target_face) -> np.ndarray:
        """HyperSwap 256×256 with multi-mask pipeline."""
        crop_size = (256, 256)

        # 1. Source embedding (L2-normalized)
        source_emb = actor_face.normed_embedding.reshape(1, -1).astype(np.float32)

        # 2. Warp target face to 256x256
        matrix = self._get_affine_matrix(target_face.kps, crop_size)
        original_crop = self._warp_crop(scene_img, matrix, crop_size)

        # 3. Preprocess for ONNX: BGR→RGB, [0,1], normalize to [-1,1], CHW, batch
        crop_rgb = original_crop[:, :, ::-1].astype(np.float32) / 255.0
        crop_norm = (crop_rgb - 0.5) / 0.5
        crop_input = np.expand_dims(crop_norm.transpose(2, 0, 1), axis=0)

        # 4. Inference
        result = self.hyperswap.run(None, {
            "source": source_emb,
            "target": crop_input,
        })
        swapped = result[0][0]  # (3, 256, 256)

        # 5. Postprocess: CHW→HWC, denormalize, RGB→BGR
        swapped_hwc = swapped.transpose(1, 2, 0)
        swapped_01 = (swapped_hwc * 0.5 + 0.5).clip(0, 1)
        swapped_crop = (swapped_01[:, :, ::-1] * 255).astype(np.uint8)

        # 6. Create composite mask from ORIGINAL crop (before swap)
        # This detects occlusions and face regions in the scene, not the swapped face
        mask = self.face_masker.create_composite_mask(original_crop, blur=0.3)

        # 7. Paste back with optimized bounding-box approach
        return self._paste_back_optimized(scene_img, swapped_crop, mask, matrix)

    def _paste_back_optimized(
        self,
        frame: np.ndarray,
        crop: np.ndarray,
        mask: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Optimized paste-back: inverse-warp only within the tight bounding box
        of the face region, not the entire frame. Reduces edge artifacts and is faster.
        Ported from FaceFusion's face_helper.py approach.
        """
        frame_h, frame_w = frame.shape[:2]
        crop_h, crop_w = crop.shape[:2]
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)

        # Calculate the bounding box of the paste region in frame space
        # by transforming the crop corners through the inverse matrix
        corners = np.array([
            [0, 0],
            [crop_w, 0],
            [crop_w, crop_h],
            [0, crop_h],
        ], dtype=np.float32)

        transformed = cv2.transform(
            corners.reshape(1, -1, 2), inverse_matrix
        )[0]

        # Get tight bounding box, clipped to frame
        x1 = max(0, int(np.floor(transformed[:, 0].min())))
        y1 = max(0, int(np.floor(transformed[:, 1].min())))
        x2 = min(frame_w, int(np.ceil(transformed[:, 0].max())))
        y2 = min(frame_h, int(np.ceil(transformed[:, 1].max())))

        paste_w = x2 - x1
        paste_h = y2 - y1
        if paste_w <= 0 or paste_h <= 0:
            return frame

        # Offset the inverse matrix to the paste region
        local_matrix = inverse_matrix.copy()
        local_matrix[0, 2] -= x1
        local_matrix[1, 2] -= y1

        # Warp crop and mask only to the paste region size
        inv_crop = cv2.warpAffine(
            crop, local_matrix, (paste_w, paste_h),
            borderMode=cv2.BORDER_REPLICATE,
        )
        inv_mask = cv2.warpAffine(
            mask, local_matrix, (paste_w, paste_h),
        ).clip(0, 1)

        # Expand mask to 3 channels
        inv_mask_3ch = inv_mask[:, :, np.newaxis]

        # Alpha blend only within the paste region
        result = frame.copy()
        region = result[y1:y2, x1:x2].astype(np.float32)
        blended = region * (1 - inv_mask_3ch) + inv_crop.astype(np.float32) * inv_mask_3ch
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result
