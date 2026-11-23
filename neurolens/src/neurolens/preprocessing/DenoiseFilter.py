from __future__ import annotations

import warnings
import numpy as np
import tensorflow as tf
from scipy.signal import wiener as scipy_wiener

from neurolens.preprocessing.ImageOperator import ImageOperator

try:
    from skimage.restoration import (
        denoise_nl_means,
        denoise_tv_chambolle,
        estimate_sigma,
    )
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    import bm3d as _bm3d_pkg
    _HAS_BM3D = True
except ImportError:
    _HAS_BM3D = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class DenoiseFilter(ImageOperator):
    """Composable MRI denoising operator supporting TV, Wiener, Perona-Malik, NLM, and BM3D."""

    SUPPORTED_METHODS = ("none", "TV", "Wiener", "PeronaMalik", "NLM", "BM3D")

    def __init__(
        self,
        method: str = "TV",
        *,
        # Total Variation (Chambolle)
        tv_weight: float = 0.08,
        # Wiener
        wiener_size: int = 5,
        # Perona-Malik anisotropic diffusion
        pm_iterations: int = 10,
        pm_kappa: float = 0.15,
        pm_gamma: float = 0.20,
        pm_option: int = 1,
        # Non-Local Means
        nlm_patch_size: int = 5,
        nlm_patch_distance: int = 6,
        nlm_h_factor: float = 1.15,
        # BM3D
        bm3d_sigma_psd: float | None = None,
    ) -> None:
        canonical = self._canonical_name(method)
        if canonical not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported denoise method '{method}'. Expected one of {self.SUPPORTED_METHODS}."
            )
        if not (0.0 < pm_gamma <= 0.25):
            raise ValueError("pm_gamma must be in (0, 0.25] for numerical stability.")

        self.method = canonical
        self.tv_weight = tv_weight
        self.wiener_size = wiener_size
        self.pm_iterations = pm_iterations
        self.pm_kappa = pm_kappa
        self.pm_gamma = pm_gamma
        self.pm_option = pm_option
        self.nlm_patch_size = nlm_patch_size
        self.nlm_patch_distance = nlm_patch_distance
        self.nlm_h_factor = nlm_h_factor
        self.bm3d_sigma_psd = bm3d_sigma_psd

    @staticmethod
    def _canonical_name(name: str) -> str:
        lookup = {
            "none": "none",
            "tv": "TV",
            "wiener": "Wiener",
            "peronamalik": "PeronaMalik",
            "perona_malik": "PeronaMalik",
            "nlm": "NLM",
            "bm3d": "BM3D",
        }
        key = name.strip().lower().replace("-", "").replace(" ", "")
        return lookup.get(key, name)

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        if self.method == "none":
            return input

        shape = input.shape
        out = tf.py_function(self._apply_tf, [input], Tout=tf.float32)
        if shape.rank is not None:
            out.set_shape(shape)
        return out

    def _apply_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        arr = tensor.numpy().astype(np.float32)
        squeeze_channel = arr.ndim == 3 and arr.shape[-1] == 1
        img2d = arr[:, :, 0] if squeeze_channel else arr
        denoised = self.denoise_array(img2d)
        denoised = np.clip(denoised, 0.0, 1.0).astype(np.float32)
        if squeeze_channel:
            denoised = denoised[:, :, np.newaxis]
        return tf.convert_to_tensor(denoised, dtype=tf.float32)

    def denoise_array(self, img: np.ndarray) -> np.ndarray:
        """Apply the configured denoising algorithm to a 2D float32 image in [0, 1]."""
        img = np.asarray(img, dtype=np.float32)
        if self.method == "none":
            return img
        if self.method == "TV":
            return self._denoise_tv(img)
        if self.method == "Wiener":
            return self._denoise_wiener(img)
        if self.method == "PeronaMalik":
            return self._denoise_perona_malik(img)
        if self.method == "NLM":
            return self._denoise_nlm(img)
        if self.method == "BM3D":
            return self._denoise_bm3d(img)
        raise RuntimeError(f"Unhandled method: {self.method}")

    @staticmethod
    def _estimate_noise_sigma(img: np.ndarray) -> float:
        if _HAS_SKIMAGE:
            sigma = float(estimate_sigma(img, channel_axis=None))
            if np.isfinite(sigma) and sigma > 1e-5:
                return sigma
        # Robust MAD estimator on Laplacian response
        lap = (
            -4.0 * img
            + np.roll(img, 1, axis=0)
            + np.roll(img, -1, axis=0)
            + np.roll(img, 1, axis=1)
            + np.roll(img, -1, axis=1)
        )
        mad = float(np.median(np.abs(lap - np.median(lap))))
        sigma = mad / (0.6745 * np.sqrt(20.0))
        return max(sigma, 1e-3)

    def _denoise_tv(self, img: np.ndarray) -> np.ndarray:
        if _HAS_SKIMAGE:
            return denoise_tv_chambolle(img, weight=self.tv_weight, channel_axis=None).astype(np.float32)

        # Pure NumPy Chambolle dual projection fallback
        px = np.zeros_like(img, dtype=np.float32)
        py = np.zeros_like(img, dtype=np.float32)
        tau = 0.248
        for _ in range(30):
            div_p = (
                px - np.roll(px, 1, axis=1)
                + py - np.roll(py, 1, axis=0)
            )
            u = img + self.tv_weight * div_p
            gx = np.roll(u, -1, axis=1) - u
            gy = np.roll(u, -1, axis=0) - u
            norm = np.maximum(1.0, np.sqrt(gx * gx + gy * gy))
            px = (px + (tau / self.tv_weight) * gx) / norm
            py = (py + (tau / self.tv_weight) * gy) / norm
        div_p = px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)
        return (img + self.tv_weight * div_p).astype(np.float32)

    def _denoise_wiener(self, img: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            filtered = scipy_wiener(img.astype(np.float64), mysize=self.wiener_size)
        filtered = np.nan_to_num(filtered, nan=float(np.mean(img)), posinf=1.0, neginf=0.0)
        return filtered.astype(np.float32)

    def _denoise_perona_malik(self, img: np.ndarray) -> np.ndarray:
        """Perona-Malik 2D anisotropic diffusion (4-connected stencil)."""
        u = img.astype(np.float32).copy()
        kappa_sq = float(self.pm_kappa) ** 2 + 1e-12

        for _ in range(self.pm_iterations):
            north = np.pad(u[:-1, :], ((1, 0), (0, 0)), mode="edge") - u
            south = np.pad(u[1:, :], ((0, 1), (0, 0)), mode="edge") - u
            west = np.pad(u[:, :-1], ((0, 0), (1, 0)), mode="edge") - u
            east = np.pad(u[:, 1:], ((0, 0), (0, 1)), mode="edge") - u

            if self.pm_option == 1:
                c_n = np.exp(-(north * north) / kappa_sq)
                c_s = np.exp(-(south * south) / kappa_sq)
                c_w = np.exp(-(west * west) / kappa_sq)
                c_e = np.exp(-(east * east) / kappa_sq)
            else:
                c_n = 1.0 / (1.0 + (north * north) / kappa_sq)
                c_s = 1.0 / (1.0 + (south * south) / kappa_sq)
                c_w = 1.0 / (1.0 + (west * west) / kappa_sq)
                c_e = 1.0 / (1.0 + (east * east) / kappa_sq)

            u += self.pm_gamma * (c_n * north + c_s * south + c_w * west + c_e * east)

        return u

    def _denoise_nlm(self, img: np.ndarray) -> np.ndarray:
        sigma = self._estimate_noise_sigma(img)
        if _HAS_SKIMAGE:
            return denoise_nl_means(
                img,
                h=self.nlm_h_factor * sigma,
                sigma=sigma,
                fast_mode=True,
                patch_size=self.nlm_patch_size,
                patch_distance=self.nlm_patch_distance,
                channel_axis=None,
            ).astype(np.float32)
        if _HAS_CV2:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            h_param = max(1.0, self.nlm_h_factor * sigma * 255.0)
            out = cv2.fastNlMeansDenoising(
                u8,
                None,
                h=h_param,
                templateWindowSize=self.nlm_patch_size,
                searchWindowSize=2 * self.nlm_patch_distance + 1,
            )
            return (out.astype(np.float32) / 255.0)
        raise ImportError("Install 'scikit-image' or 'opencv-python' for NLM denoising.")

    def _denoise_bm3d(self, img: np.ndarray) -> np.ndarray:
        sigma = self.bm3d_sigma_psd if self.bm3d_sigma_psd is not None else self._estimate_noise_sigma(img)
        if _HAS_BM3D:
            out = _bm3d_pkg.bm3d(
                img.astype(np.float64),
                sigma_psd=float(sigma),
                stage_arg=_bm3d_pkg.BM3DStages.ALL_STAGES,
            )
            return out.astype(np.float32)

        warnings.warn(
            "Package 'bm3d' is not installed; falling back to NLM denoising. Run `pip install bm3d`.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._denoise_nlm(img)