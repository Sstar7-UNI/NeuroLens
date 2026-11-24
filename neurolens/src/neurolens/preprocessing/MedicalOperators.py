from __future__ import annotations

import warnings
from typing import Callable, List, Sequence

import numpy as np
import tensorflow as tf
from scipy import ndimage

from neurolens.data_management.ImageRecord import ImageRecord
from neurolens.preprocessing.ImageOperator import ImageOperator
from neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor
from neurolens.preprocessing.OtsuSegmenter import OtsuSegmenter

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False

try:
    from skimage.filters import sato as skimage_sato
    from skimage.filters import threshold_multiotsu
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - vmin) / (vmax - vmin)).astype(np.float32)


class _NumpyImageOperator(ImageOperator):
    """Base adapter bridging 3D TensorFlow tensors (H, W, 1) to 2D NumPy functions."""

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        shape = input.shape
        out = tf.py_function(self._wrapped, [input], Tout=tf.float32)
        if shape.rank is not None:
            out.set_shape(shape)
        return out

    def _wrapped(self, tensor: tf.Tensor) -> tf.Tensor:
        arr = tensor.numpy().astype(np.float32)
        has_channel = arr.ndim == 3 and arr.shape[-1] == 1
        img2d = arr[:, :, 0] if has_channel else arr
        res = np.clip(self.apply_numpy(img2d), 0.0, 1.0).astype(np.float32)
        if has_channel:
            res = res[:, :, np.newaxis]
        return tf.convert_to_tensor(res, dtype=tf.float32)



class LoGFilter(_NumpyImageOperator):
    """Laplacian of Gaussian edge/blob enhancement."""

    def __init__(self, sigma: float = 1.2, alpha: float = 0.5, residual: bool = True) -> None:
        self.sigma = sigma
        self.alpha = alpha
        self.residual = residual

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        smoothed = ndimage.gaussian_filter(img2d.astype(np.float32), sigma=self.sigma)
        log_resp = ndimage.laplace(smoothed)
        if self.residual:
            # Subtract negative Laplacian peaks to sharpen boundaries while keeping anatomy
            enhanced = img2d - self.alpha * log_resp
            return _normalize01(enhanced)
        return _normalize01(log_resp)


class DoGFilter(_NumpyImageOperator):
    """Difference of Gaussians multi-scale band-pass enhancement."""

    def __init__(
        self,
        sigma_low: float = 1.0,
        sigma_high: float = 2.2,
        alpha: float = 0.6,
        residual: bool = True,
    ) -> None:
        if sigma_high <= sigma_low:
            raise ValueError("sigma_high must be greater than sigma_low.")
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.alpha = alpha
        self.residual = residual

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        g1 = ndimage.gaussian_filter(img2d.astype(np.float32), sigma=self.sigma_low)
        g2 = ndimage.gaussian_filter(img2d.astype(np.float32), sigma=self.sigma_high)
        dog = g1 - g2
        if self.residual:
            return _normalize01(img2d + self.alpha * dog)
        return _normalize01(dog)


class CLAHEFilter(_NumpyImageOperator):
    """Contrast Limited Adaptive Histogram Equalization."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        if _HAS_CV2:
            u8 = np.clip(img2d * 255.0, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=float(self.clip_limit),
                tileGridSize=tuple(self.tile_grid_size),
            )
            return clahe.apply(u8).astype(np.float32) / 255.0
        if _HAS_SKIMAGE:
            from skimage.exposure import equalize_adapthist
            return equalize_adapthist(img2d, clip_limit=self.clip_limit / 100.0).astype(np.float32)
        raise ImportError("Install 'opencv-python' or 'scikit-image' for CLAHE.")




class NyulNormalizer(_NumpyImageOperator):
    """Piecewise-linear MRI histogram landmark standardization (Nyul & Udupa)."""

    DEFAULT_PERCENTILES = (1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0)

    def __init__(
        self,
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
        standard_scale: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.percentiles = np.asarray(percentiles, dtype=np.float32)
        self.standard_scale = standard_scale
        self.standard_landmarks: np.ndarray = np.linspace(
            standard_scale[0], standard_scale[1], len(self.percentiles), dtype=np.float32
        )

    @staticmethod
    def _foreground_landmarks(img2d: np.ndarray, percentiles: np.ndarray) -> np.ndarray | None:
        fg = img2d[img2d > 0.02]
        if fg.size < 32:
            fg = img2d.ravel()
        lm = np.percentile(fg, percentiles).astype(np.float32)
        # Ensure strictly increasing landmarks for piecewise interpolation
        for i in range(1, len(lm)):
            if lm[i] <= lm[i - 1]:
                lm[i] = lm[i - 1] + 1e-5
        if lm[-1] - lm[0] < 1e-6:
            return None
        return lm

    def fit(self, images: Sequence[np.ndarray]) -> "NyulNormalizer":
        """Learn average landmark intensities on the training set (no data leakage)."""
        all_mapped = []
        s_min, s_max = self.standard_scale
        for img in images:
            lm = self._foreground_landmarks(img, self.percentiles)
            if lm is None:
                continue
            mapped = s_min + (lm - lm[0]) / (lm[-1] - lm[0]) * (s_max - s_min)
            all_mapped.append(mapped)

        if all_mapped:
            self.standard_landmarks = np.mean(np.stack(all_mapped, axis=0), axis=0).astype(np.float32)
        return self

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        lm = self._foreground_landmarks(img2d, self.percentiles)
        if lm is None:
            return _normalize01(img2d)
        standardized = np.interp(img2d.ravel(), lm, self.standard_landmarks).reshape(img2d.shape)
        return np.clip(standardized, 0.0, 1.0).astype(np.float32)


class N4BiasFieldCorrector(_NumpyImageOperator):
    """N4ITK MRI intensity inhomogeneity (bias field) correction via SimpleITK."""

    def __init__(
        self,
        shrink_factor: int = 2,
        num_fitting_levels: int = 3,
        max_iterations_per_level: int = 30,
    ) -> None:
        self.shrink_factor = max(1, int(shrink_factor))
        self.num_fitting_levels = int(num_fitting_levels)
        self.max_iterations_per_level = int(max_iterations_per_level)

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        if _HAS_SITK:
            # Shift strictly positive so log-bias estimation never encounters zeros
            shifted = img2d.astype(np.float32) + 1e-3
            sitk_img = sitk.GetImageFromArray(shifted)
            mask = sitk.OtsuThreshold(sitk_img, 0, 1, 128)

            if self.shrink_factor > 1:
                work_img = sitk.Shrink(sitk_img, [self.shrink_factor] * 2)
                work_mask = sitk.Shrink(mask, [self.shrink_factor] * 2)
            else:
                work_img = sitk_img
                work_mask = mask

            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(
                [self.max_iterations_per_level] * self.num_fitting_levels
            )
            corrector.Execute(work_img, work_mask)
            log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
            corrected_full = sitk_img / sitk.Exp(log_bias)
            out = sitk.GetArrayFromImage(corrected_full).astype(np.float32) - 1e-3
            return _normalize01(np.maximum(out, 0.0))

        warnings.warn(
            "SimpleITK is not installed; using low-frequency Gaussian bias-field approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Multiplicative low-frequency illumination field fallback
        field = ndimage.gaussian_filter(img2d + 1e-2, sigma=max(img2d.shape) / 6.0)
        return _normalize01((img2d + 1e-2) / np.maximum(field, 1e-3))


class MultiOtsuSegmenter(_NumpyImageOperator):
    """Multi-level Otsu intensity segmentation (K=3 tissue classes)."""

    def __init__(self, classes: int = 3, mask_intensities: bool = False) -> None:
        self.classes = classes
        self.mask_intensities = mask_intensities

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        if _HAS_SKIMAGE:
            try:
                thresholds = threshold_multiotsu(img2d, classes=self.classes)
                regions = np.digitize(img2d, bins=thresholds).astype(np.float32)
                norm_regions = regions / max(float(self.classes - 1), 1.0)
                if self.mask_intensities:
                    # Keep pixels above background class 0
                    return (img2d * (regions > 0)).astype(np.float32)
                return norm_regions
            except ValueError:
                pass

        # Fallback: quantile partitioning
        qs = np.linspace(0.0, 1.0, self.classes + 1)[1:-1]
        thresholds = np.quantile(img2d, qs)
        regions = np.digitize(img2d, bins=thresholds).astype(np.float32)
        return regions / max(float(self.classes - 1), 1.0)


class KMeansSegmenter(_NumpyImageOperator):
    """Deterministic intensity K-Means segmentation with sorted cluster centroids."""

    def __init__(self, k: int = 4, seed: int = 42, mask_background: bool = False) -> None:
        self.k = k
        self.seed = seed
        self.mask_background = mask_background

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        h, w = img2d.shape
        pixels = img2d.reshape(-1, 1).astype(np.float32)

        if _HAS_CV2:
            cv2.setRNGSeed(self.seed)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            _, labels, centers = cv2.kmeans(
                pixels, self.k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            labels = labels.ravel()
            centers = centers.ravel()
        else:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.k, n_init=3, max_iter=40, random_state=self.seed)
            labels = km.fit_predict(pixels)
            centers = km.cluster_centers_.ravel()

        # Sort clusters from darkest (0) to brightest (k - 1) so cluster IDs are consistent
        order = np.argsort(centers)
        rank_map = np.empty_like(order)
        rank_map[order] = np.arange(self.k)
        sorted_labels = rank_map[labels].reshape(h, w).astype(np.float32)

        if self.mask_background:
            return (img2d * (sorted_labels > 0)).astype(np.float32)
        return sorted_labels / max(float(self.k - 1), 1.0)




class SatoFilter(_NumpyImageOperator):
    """Sato multi-scale tubeness/ridge structure enhancement filter."""

    def __init__(
        self,
        sigmas: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
        black_ridges: bool = False,
        blend_alpha: float = 0.5,
    ) -> None:
        self.sigmas = list(sigmas)
        self.black_ridges = black_ridges
        self.blend_alpha = blend_alpha

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        if not _HAS_SKIMAGE:
            raise ImportError("Install 'scikit-image' for Sato structure enhancement.")
        ridges = skimage_sato(
            img2d.astype(np.float32),
            sigmas=self.sigmas,
            black_ridges=self.black_ridges,
            mode="reflect",
        )
        ridges = _normalize01(ridges)
        if self.blend_alpha > 0.0:
            return _normalize01((1.0 - self.blend_alpha) * img2d + self.blend_alpha * ridges)
        return ridges


class GaborFilterBank(_NumpyImageOperator):
    """Multi-orientation, multi-scale 2D Gabor texture/boundary enhancement bank."""

    def __init__(
        self,
        orientations: int = 6,
        lambdas: Sequence[float] = (4.0, 8.0),
        sigma: float = 2.0,
        gamma: float = 0.5,
        blend_alpha: float = 0.5,
    ) -> None:
        self.orientations = orientations
        self.lambdas = list(lambdas)
        self.sigma = sigma
        self.gamma = gamma
        self.blend_alpha = blend_alpha

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        responses = []
        ksize = int(6 * self.sigma + 1) | 1
        for lambd in self.lambdas:
            for i in range(self.orientations):
                theta = i * np.pi / self.orientations
                if _HAS_CV2:
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize),
                        sigma=self.sigma,
                        theta=theta,
                        lambd=lambd,
                        gamma=self.gamma,
                        psi=0.0,
                        ktype=cv2.CV_32F,
                    )
                    resp = cv2.filter2D(img2d.astype(np.float32), cv2.CV_32F, kernel)
                else:
                    half = ksize // 2
                    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
                    xr = x * np.cos(theta) + y * np.sin(theta)
                    yr = -x * np.sin(theta) + y * np.cos(theta)
                    kernel = np.exp(-(xr ** 2 + (self.gamma * yr) ** 2) / (2.0 * self.sigma ** 2)) * np.cos(
                        2.0 * np.pi * xr / lambd
                    )
                    resp = ndimage.convolve(img2d.astype(np.float32), kernel, mode="reflect")
                responses.append(np.abs(resp))

        energy = _normalize01(np.max(np.stack(responses, axis=0), axis=0))
        if self.blend_alpha > 0.0:
            return _normalize01((1.0 - self.blend_alpha) * img2d + self.blend_alpha * energy)
        return energy




class SkullStripper(_NumpyImageOperator):
    """2D MRI brain extraction via Otsu thresholding, morphological opening, and LCC masking."""

    def __init__(self, open_radius: int = 3, close_radius: int = 5) -> None:
        self.open_radius = open_radius
        self.close_radius = close_radius

    def apply_numpy(self, img2d: np.ndarray) -> np.ndarray:
        norm = _normalize01(img2d)
        u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

        if _HAS_CV2:
            _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (binary > 0).astype(np.uint8)
            k_open = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.open_radius + 1, 2 * self.open_radius + 1)
            )
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

            if num_labels > 1:
                # Ignore background label 0 and select largest intracranial component
                largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                brain_mask = (labels == largest_label).astype(np.uint8)
            else:
                brain_mask = mask

            k_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.close_radius + 1, 2 * self.close_radius + 1)
            )
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, k_close)
            brain_mask = ndimage.binary_fill_holes(brain_mask > 0).astype(np.float32)
        else:
            thresh = float(np.mean(norm[norm > 0.02])) if np.any(norm > 0.02) else 0.1
            mask = norm > (0.65 * thresh)
            opened = ndimage.binary_opening(mask, iterations=self.open_radius)
            labeled, n_comp = ndimage.label(opened)
            if n_comp > 0:
                sizes = ndimage.sum(opened, labeled, range(1, n_comp + 1))
                brain_mask = labeled == (1 + int(np.argmax(sizes)))
            else:
                brain_mask = mask
            brain_mask = ndimage.binary_closing(brain_mask, iterations=self.close_radius)
            brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.float32)

        # Guard against stripping an entire low-contrast slice
        if float(np.mean(brain_mask)) < 0.02:
            return img2d.astype(np.float32)

        return (img2d * brain_mask).astype(np.float32)
