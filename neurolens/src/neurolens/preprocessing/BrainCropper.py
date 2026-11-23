from __future__ import annotations

import numpy as np
import tensorflow as tf

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from neurolens.preprocessing.ImageOperator import ImageOperator


class BrainCropper(ImageOperator):
    """
    Robust intracranial bounding-box cropper using Otsu thresholding, morphological
    opening, and area-filtered connected components (ignoring corner orientation markers
    and JPEG border artifacts), followed by aspect-preserving square padding and resize.
    """

    def __init__(self, target_size: int = 128, margin_ratio: float = 0.06) -> None:
        self.target_size = int(target_size)
        self.margin_ratio = float(margin_ratio)

    def apply(self, image):
        is_tensor = tf.is_tensor(image)
        in_dtype = image.dtype if is_tensor else None
        arr = np.asarray(image.numpy() if is_tensor else image, dtype=np.float32)
        had_channel = arr.ndim == 3
        arr2d = arr[..., 0] if had_channel else arr

        max_val = float(arr2d.max()) if arr2d.size > 0 else 0.0
        scale = 255.0 if max_val > 1.5 else 1.0
        norm2d = arr2d / scale
        h, w = norm2d.shape
        cropped = norm2d

        if cv2 is not None:
            blurred = cv2.GaussianBlur(norm2d, (5, 5), sigmaX=1.0)
            u8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = max(0.04, 0.55 * (float(otsu_val) / 255.0))
            fg = (blurred > thresh).astype(np.uint8)

            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
            num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(
                fg, connectivity=8
            )

            if num_labels > 1:
                areas = comp_stats[1:, cv2.CC_STAT_AREA]
                max_area = float(areas.max())
                keep_labels = [
                    1 + idx
                    for idx, area in enumerate(areas)
                    if area >= 0.10 * max_area and area >= 32
                ]
                if keep_labels:
                    brain_mask = np.isin(labels, keep_labels)
                    ys, xs = np.where(brain_mask)
                    if ys.size > 32 and xs.size > 32:
                        y0, y1 = int(ys.min()), int(ys.max()) + 1
                        x0, x1 = int(xs.min()), int(xs.max()) + 1
                        box_area = (y1 - y0) * (x1 - x0)
                        if box_area >= 0.05 * (h * w):
                            pad_y = max(2, int(round((y1 - y0) * self.margin_ratio)))
                            pad_x = max(2, int(round((x1 - x0) * self.margin_ratio)))
                            y0 = max(0, y0 - pad_y)
                            y1 = min(h, y1 + pad_y)
                            x0 = max(0, x0 - pad_x)
                            x1 = min(w, x1 + pad_x)
                            cropped = norm2d[y0:y1, x0:x1]
        else:
            thresh = max(0.05, 0.08 * float(norm2d.max()))
            ys, xs = np.where(norm2d > thresh)
            if ys.size > 32 and xs.size > 32:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                pad_y = max(2, int(round((y1 - y0) * self.margin_ratio)))
                pad_x = max(2, int(round((x1 - x0) * self.margin_ratio)))
                cropped = norm2d[
                    max(0, y0 - pad_y) : min(h, y1 + pad_y),
                    max(0, x0 - pad_x) : min(w, x1 + pad_x),
                ]

        ch, cw = cropped.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.float32)
        off_y = (side - ch) // 2
        off_x = (side - cw) // 2
        square[off_y : off_y + ch, off_x : off_x + cw] = cropped

        if side != self.target_size:
            if cv2 is not None:
                interp = cv2.INTER_AREA if side > self.target_size else cv2.INTER_LINEAR
                resized = cv2.resize(
                    square, (self.target_size, self.target_size), interpolation=interp
                )
            else:
                resized_t = tf.image.resize(
                    square[..., np.newaxis],
                    [self.target_size, self.target_size],
                    method="bilinear",
                )
                resized = resized_t.numpy()[..., 0]
        else:
            resized = square

        out = np.clip(resized * scale, 0.0, scale).astype(np.float32)
        if had_channel:
            out = out[..., np.newaxis]
        if is_tensor:
            return tf.convert_to_tensor(out, dtype=in_dtype or tf.float32)
        return out