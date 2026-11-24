from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import gc
import hashlib
import inspect
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score

try:
    import cv2

    cv2.setNumThreads(1)
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from scipy import ndimage, stats
except ImportError:  # pragma: no cover
    ndimage = None
    stats = None

try:
    from skimage.restoration import denoise_wavelet, estimate_sigma
except ImportError:  # pragma: no cover
    denoise_wavelet = None
    estimate_sigma = None

try:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
except ImportError:  # pragma: no cover
    sitk = None

from neurolens.data_management.DatasetManager import DatasetManager
from neurolens.data_management.DatasetSplit import DatasetSplit
from neurolens.data_management.ImageRecord import ImageRecord
from neurolens.data_management.SplitType import SplitType
from neurolens.data_management.TumorClass import TumorClass
from neurolens.data_storage.ModelConfig import ModelConfig
from neurolens.network.CNNClassifier import CNNClassifier
from neurolens.network.DecisionRule import DecisionRule
from neurolens.preprocessing.BrainCropper import BrainCropper
from neurolens.preprocessing.DenoiseFilter import DenoiseFilter
from neurolens.preprocessing.ImageOperator import ImageOperator
from neurolens.preprocessing.MedicalOperators import (
    CLAHEFilter,
    N4BiasFieldCorrector,
    NyulNormalizer,
)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    mins, rem = divmod(seconds, 60.0)
    if mins < 60.0:
        return f"{int(mins):02d}m {rem:05.2f}s"
    hours, rem_mins = divmod(mins, 60.0)
    return f"{int(hours):d}h {int(rem_mins):02d}m {rem:04.1f}s"


class _LivePhaseTimer:
    """Prints a periodic heartbeat while long CPU/GPU operations run."""

    def __init__(
        self,
        label: str,
        log_fn: Callable[[str], None],
        *,
        enabled: bool = True,
        heartbeat_s: float = 5.0,
        indent: str = "    ",
    ) -> None:
        self.label = label
        self.log_fn = log_fn
        self.enabled = enabled
        self.heartbeat_s = heartbeat_s
        self.indent = indent
        self.start_time = 0.0
        self.elapsed_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            elapsed = time.perf_counter() - self.start_time
            self.log_fn(f"{self.indent}... still {self.label} ({elapsed:.1f}s elapsed)...")

    def __enter__(self) -> "_LivePhaseTimer":
        self.start_time = time.perf_counter()
        if self.enabled:
            self.log_fn(f"{self.indent}[TIMER] {self.label}...")
            self._stop.clear()
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_s = time.perf_counter() - self.start_time
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            status = "failed" if exc_type is not None else "done"
            self.log_fn(f"{self.indent}[TIMER] {self.label} {status} in {self.elapsed_s:.2f}s.")


def _extract_split_records(
    split: DatasetSplit,
) -> tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    for tr_m, va_m, te_m in (
        ("getTrainingSet", "getValidationSet", "getTestSet"),
        ("getTrainRecords", "getValidationRecords", "getTestRecords"),
        ("getTrain", "getVal", "getTest"),
    ):
        if callable(getattr(split, tr_m, None)) and callable(getattr(split, va_m, None)):
            te_fn = getattr(split, te_m, None)
            return (
                list(getattr(split, tr_m)()),
                list(getattr(split, va_m)()),
                list(te_fn()) if callable(te_fn) else [],
            )

    for tr_a, va_a, te_a in (
        ("_trainingSamples", "_validationSamples", "_testSamples"),
        ("trainRecords", "validationRecords", "testRecords"),
        ("train", "val", "test"),
    ):
        if hasattr(split, tr_a) and hasattr(split, va_a) and hasattr(split, te_a):
            return (
                list(getattr(split, tr_a)),
                list(getattr(split, va_a)),
                list(getattr(split, te_a)),
            )

    list_fields = [v for v in vars(split).values() if isinstance(v, (list, tuple))]
    if len(list_fields) >= 3:
        return list(list_fields[0]), list(list_fields[1]), list(list_fields[2])

    raise AttributeError(
        f"Cannot extract records from DatasetSplit ({list(vars(split).keys())})"
    )



def _to_numpy_2d(image) -> tuple[np.ndarray, bool, tf.DType | None, bool, float]:
    is_tensor = tf.is_tensor(image)
    in_dtype = image.dtype if is_tensor else None
    arr = np.asarray(image.numpy() if is_tensor else image, dtype=np.float32)
    had_channel = arr.ndim == 3
    arr2d = arr[..., 0] if had_channel else arr
    max_val = float(arr2d.max()) if arr2d.size > 0 else 0.0
    scale = 255.0 if max_val > 1.5 else 1.0
    return arr2d / scale, is_tensor, in_dtype, had_channel, scale


def _from_numpy_2d(
    arr2d: np.ndarray,
    is_tensor: bool,
    in_dtype: tf.DType | None,
    had_channel: bool,
    scale: float,
):
    out = np.clip(arr2d * scale, 0.0, scale).astype(np.float32)
    if had_channel:
        out = out[..., np.newaxis]
    if is_tensor:
        return tf.convert_to_tensor(out, dtype=in_dtype or tf.float32)
    return out


def _box_mean(img: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * max(1, int(radius)) + 1
    if cv2 is not None:
        return cv2.boxFilter(
            img.astype(np.float32), ddepth=-1, ksize=(k, k), borderType=cv2.BORDER_REFLECT
        )
    if ndimage is not None:
        return ndimage.uniform_filter(img.astype(np.float32), size=k, mode="reflect")
    return img.astype(np.float32)


def _estimate_noise_sigma_2d(img: np.ndarray) -> float:
    if estimate_sigma is not None:
        try:
            sig = float(estimate_sigma(img, channel_axis=None))
            if math.isfinite(sig) and sig > 1e-5:
                return sig
        except Exception:
            pass
    lap = (
        -4.0 * img
        + np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    )
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return max(mad / (0.6745 * math.sqrt(20.0)), 1e-3)


class ForegroundLCCCleaner(ImageOperator):
    """
    Removes disconnected border watermarks/banners (e.g., top-edge URL headers)
    by retaining only the largest connected head/brain component.
    """

    def __init__(self, threshold: float = 0.04, close_radius: int = 3) -> None:
        self.threshold = float(threshold)
        self.close_radius = int(close_radius)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        mask = (arr2d > self.threshold).astype(np.uint8)
        if int(mask.sum()) < 32:
            return _from_numpy_2d(arr2d, is_tensor, in_dtype, had_channel, scale)

        k = 2 * self.close_radius + 1
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
                closed, connectivity=8
            )
            if n_labels > 1:
                largest_lbl = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
                keep = labels == largest_lbl
            else:
                keep = mask > 0
        elif ndimage is not None:
            closed = ndimage.binary_closing(mask > 0, iterations=self.close_radius)
            labeled, n_comp = ndimage.label(closed)
            if n_comp > 0:
                sizes = ndimage.sum(closed, labeled, range(1, n_comp + 1))
                keep = labeled == (1 + int(np.argmax(sizes)))
            else:
                keep = mask > 0
        else:
            keep = mask > 0

        if ndimage is not None:
            keep = ndimage.binary_fill_holes(keep)

        if float(np.mean(keep)) >= 0.05:
            out = np.where(keep, arr2d, 0.0)
        else:
            out = arr2d
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class SUSANDenoiseFilter(ImageOperator):
    """
    SUSAN (Smallest Univalue Segment Assimilating Nucleus) edge-preserving denoiser.
    Averages only spatial neighbors within a brightness-similarity threshold of the center pixel.
    """

    def __init__(self, radius: int = 2, sigma_spatial: float = 1.5) -> None:
        self.radius = int(radius)
        self.sigma_spatial = float(sigma_spatial)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        sigma_n = _estimate_noise_sigma_2d(arr2d)
        brightness_thresh = max(2.8 * sigma_n, 0.045)

        r = self.radius
        padded = np.pad(arr2d, r, mode="reflect")
        h, w = arr2d.shape
        acc_num = np.zeros_like(arr2d, dtype=np.float32)
        acc_den = np.zeros_like(arr2d, dtype=np.float32)

        inv_2s2 = 1.0 / (2.0 * self.sigma_spatial * self.sigma_spatial)
        inv_t = 1.0 / brightness_thresh

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2_spatial = float(dx * dx + dy * dy)
                if d2_spatial > float(r * r):
                    continue
                w_spatial = math.exp(-d2_spatial * inv_2s2)
                patch = padded[r + dy : r + dy + h, r + dx : r + dx + w]
                diff_norm = (patch - arr2d) * inv_t
                d2 = diff_norm * diff_norm
                d6 = np.minimum(d2 * d2 * d2, 40.0)
                w_total = w_spatial * np.exp(-d6)
                acc_num += w_total * patch
                acc_den += w_total

        out = acc_num / np.maximum(acc_den, 1e-6)
        out[arr2d <= 1e-4] = 0.0
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class GuidedImageFilter(ImageOperator):
    """O(1) self-guided local linear edge-preserving denoiser (He et al.)."""

    def __init__(self, radius: int = 3, eps: float = 2.5e-4) -> None:
        self.radius = int(radius)
        self.eps = float(eps)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        mean_i = _box_mean(arr2d, self.radius)
        mean_ii = _box_mean(arr2d * arr2d, self.radius)
        var_i = np.maximum(0.0, mean_ii - mean_i * mean_i)

        a = var_i / (var_i + self.eps)
        b = mean_i - a * mean_i
        out = _box_mean(a, self.radius) * arr2d + _box_mean(b, self.radius)
        out[arr2d <= 1e-4] = 0.0
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class WhiteStripeNormalizer(ImageOperator):
    """
    WhiteStripe MRI intensity harmonization: centers and scales intensities using the
    Normal-Appearing White Matter (NAWM) histogram mode rather than tumor-skewed whole-brain stats.
    """

    def __init__(self, bins: int = 128, smooth_sigma: float = 2.0) -> None:
        self.bins = int(bins)
        self.smooth_sigma = float(smooth_sigma)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        fg_mask = arr2d > 0.03
        fg = arr2d[fg_mask]
        if fg.size < 32:
            return _from_numpy_2d(arr2d, is_tensor, in_dtype, had_channel, scale)

        lo, hi = float(np.percentile(fg, 1.0)), float(np.percentile(fg, 99.5))
        if hi <= lo + 1e-5:
            return _from_numpy_2d(arr2d, is_tensor, in_dtype, had_channel, scale)

        hist, edges = np.histogram(fg, bins=self.bins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_f = hist.astype(np.float32)
        if ndimage is not None:
            hist_f = ndimage.gaussian_filter1d(hist_f, sigma=self.smooth_sigma)

        start_bin = max(1, int(0.15 * self.bins))
        end_bin = max(start_bin + 1, int(0.90 * self.bins))
        peak_idx = start_bin + int(np.argmax(hist_f[start_bin:end_bin]))
        mu_nawm = float(centers[peak_idx])

        half_max = 0.5 * float(hist_f[peak_idx])
        left_idx = peak_idx
        while left_idx > 0 and hist_f[left_idx] > half_max:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(hist_f) - 1 and hist_f[right_idx] > half_max:
            right_idx += 1

        fwhm = max(float(centers[right_idx] - centers[left_idx]), 1e-4)
        global_std = max(float(np.std(fg)), 1e-4)
        sigma_nawm = float(np.clip(fwhm / 2.355, 0.35 * global_std, 1.5 * global_std))

        z_nawm = (arr2d - mu_nawm) / sigma_nawm
        out = np.clip(0.5 + 0.16 * z_nawm, 0.005, 1.0).astype(np.float32)
        out[~fg_mask] = 0.0
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class PercentileClipNormalizer(ImageOperator):
    """Foreground Winsorization (0.5th-99.9th percentiles) before Z-score."""

    def __init__(self, low_pct: float = 0.5, high_pct: float = 99.9) -> None:
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        fg_mask = arr2d > 1e-4
        fg = arr2d[fg_mask]
        if fg.size > 16:
            lo = float(np.percentile(fg, self.low_pct))
            hi = float(np.percentile(fg, self.high_pct))
            if hi > lo + 1e-6:
                clipped = np.clip(arr2d, lo, hi)
                out = (clipped - lo) / (hi - lo)
                out[~fg_mask] = 0.0
            else:
                out = arr2d
        else:
            out = arr2d
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class TopHatBottomHatFilter(ImageOperator):
    """Morphological White Top-Hat + Black Bottom-Hat local rim/core contrast enhancement."""

    def __init__(
        self, radius: int = 4, alpha_top: float = 0.35, alpha_bot: float = 0.20
    ) -> None:
        self.radius = int(radius)
        self.alpha_top = float(alpha_top)
        self.alpha_bot = float(alpha_bot)

    def apply(self, image):
        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        k = 2 * self.radius + 1
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            tophat = cv2.morphologyEx(arr2d, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(arr2d, cv2.MORPH_BLACKHAT, kernel)
        elif ndimage is not None:
            y, x = np.ogrid[-self.radius : self.radius + 1, -self.radius : self.radius + 1]
            disk = (x * x + y * y) <= (self.radius * self.radius)
            tophat = np.maximum(0.0, arr2d - ndimage.grey_opening(arr2d, footprint=disk))
            blackhat = np.maximum(0.0, ndimage.grey_closing(arr2d, footprint=disk) - arr2d)
        else:
            tophat = np.zeros_like(arr2d)
            blackhat = np.zeros_like(arr2d)

        out = arr2d + self.alpha_top * tophat - self.alpha_bot * blackhat
        out[arr2d <= 1e-4] = 0.0
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class FastDenoiseFilter(ImageOperator):
    """Edge-preserving MRI denoisers ('median', 'SUSAN', 'GIF', 'wavelet')."""

    def __init__(self, method: str = "median") -> None:
        self.method = str(method)
        self._susan = (
            SUSANDenoiseFilter(radius=2, sigma_spatial=1.5) if self.method == "SUSAN" else None
        )
        self._gif = GuidedImageFilter(radius=3, eps=2.5e-4) if self.method == "GIF" else None
        self._fallback = (
            DenoiseFilter(method=self.method)
            if self.method not in {"median", "SUSAN", "GIF", "wavelet"}
            else None
        )

    def apply(self, image):
        if self._susan is not None:
            return self._susan.apply(image)
        if self._gif is not None:
            return self._gif.apply(image)
        if self._fallback is not None:
            return self._fallback.apply(image)

        arr2d, is_tensor, in_dtype, had_channel, scale = _to_numpy_2d(image)
        if self.method == "median":
            if cv2 is not None:
                u8 = np.clip(arr2d * 255.0, 0, 255).astype(np.uint8)
                out = cv2.medianBlur(u8, 3).astype(np.float32) / 255.0
            elif ndimage is not None:
                out = ndimage.median_filter(arr2d, size=3)
            else:
                out = arr2d
        elif self.method == "wavelet":
            if denoise_wavelet is not None:
                out = denoise_wavelet(
                    arr2d,
                    method="BayesShrink",
                    mode="soft",
                    rescale_sigma=True,
                    channel_axis=None,
                ).astype(np.float32)
            elif cv2 is not None:
                out = cv2.bilateralFilter(arr2d, d=5, sigmaColor=0.06, sigmaSpace=2.5)
            else:
                out = arr2d
        else:
            out = arr2d

        out[arr2d <= 1e-4] = 0.0
        return _from_numpy_2d(out, is_tensor, in_dtype, had_channel, scale)


class ParallelN4BiasFieldCorrector(ImageOperator):
    """Thread-local N4ITK bias field corrector (1 ITK thread per worker)."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _get_worker_instance(self) -> N4BiasFieldCorrector:
        inst = getattr(self._local, "corrector", None)
        if inst is None:
            if sitk is not None:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            sig = inspect.signature(N4BiasFieldCorrector).parameters
            if "shrink_factor" in sig:
                inst = N4BiasFieldCorrector(
                    shrink_factor=4,
                    num_fitting_levels=3,
                    max_iterations_per_level=20,
                )
            else:
                inst = N4BiasFieldCorrector()
            self._local.corrector = inst
        return inst

    def apply(self, image):
        return self._get_worker_instance().apply(image)


def _compute_lbp_channel(img2d: np.ndarray) -> np.ndarray:
    """Foreground-masked 8-neighbor Local Binary Pattern texture map in [0, 1]."""
    fg_mask = img2d > 1e-4
    if ndimage is not None:
        interior_mask = ndimage.binary_erosion(fg_mask, iterations=1)
        if int(interior_mask.sum()) < 32:
            interior_mask = fg_mask
    else:
        interior_mask = fg_mask

    padded = np.pad(img2d.astype(np.float32), 1, mode="reflect")
    center = padded[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros_like(center, dtype=np.float32)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        lbp += ((neighbor >= center).astype(np.float32)) * float(1 << bit)
    lbp /= 255.0
    lbp[~interior_mask] = 0.0
    return lbp


def _compute_log_channels(
    img2d: np.ndarray, sigmas: Sequence[float] = (1.0, 2.5)
) -> List[np.ndarray]:
    """Foreground-masked multi-scale Laplacian of Gaussian maps (sigma=1.0, 2.5)."""
    channels: List[np.ndarray] = []
    fg_mask = img2d > 1e-4
    for sigma in sigmas:
        if cv2 is not None:
            ksize = int(6 * sigma + 1) | 1
            smoothed = cv2.GaussianBlur(img2d.astype(np.float32), (ksize, ksize), sigmaX=sigma)
            resp = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3) * (sigma * sigma)
        elif ndimage is not None:
            smoothed = ndimage.gaussian_filter(img2d.astype(np.float32), sigma=sigma)
            resp = ndimage.laplace(smoothed) * (sigma * sigma)
        else:
            resp = np.zeros_like(img2d, dtype=np.float32)

        fg_vals = resp[fg_mask]
        if fg_vals.size > 16:
            vmin, vmax = float(np.percentile(fg_vals, 0.5)), float(np.percentile(fg_vals, 99.5))
        else:
            vmin, vmax = float(resp.min()), float(resp.max())

        if vmax - vmin > 1e-6:
            norm_resp = np.clip((resp - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
        else:
            norm_resp = np.zeros_like(resp, dtype=np.float32)
        norm_resp[~fg_mask] = 0.0
        channels.append(norm_resp)
    return channels


_NUMPY_NATIVE_OPS = (
    BrainCropper,
    ForegroundLCCCleaner,
    SUSANDenoiseFilter,
    GuidedImageFilter,
    WhiteStripeNormalizer,
    PercentileClipNormalizer,
    TopHatBottomHatFilter,
    FastDenoiseFilter,
)



@dataclass
class ExperimentSpec:
    stage: str
    name: str
    crop_to_brain: bool = True
    n4_bias: bool = True
    input_size: int = 128
    aug_copies: int = 8
    aug_preset: str = "mri_full"
    denoise_method: str = "median"        # "none" | "median" | "SUSAN" | "GIF" | "wavelet"
    intensity_norm: str = "none"          # "none" | "percentile_clip" | "WhiteStripe" | "Nyul"
    contrast_method: str = "CLAHE_2.0_8"  # "none" | "CLAHE_1.0_8" | "CLAHE_2.0_8"
    edge_filter: str = "none"             # "none" | "TopHatBottomHat"
    feature_channels: str = "none"        # "none" | "lbp" | "log" | "lbp_log"
    foreground_zscore: bool = True
    seed: int = 42

    def num_channels(self) -> int:
        return {"none": 1, "lbp": 2, "log": 3, "lbp_log": 4}.get(self.feature_channels, 1)

    def common_prefix_signature(self) -> str:
        return f"crop={int(self.crop_to_brain)}|n4={int(self.n4_bias)}|sz={self.input_size}"

    def single_channel_signature(self) -> str:
        return (
            f"{self.common_prefix_signature()}|denoise={self.denoise_method}|"
            f"norm={self.intensity_norm}|contrast={self.contrast_method}|"
            f"edge={self.edge_filter}"
        )

    def pipeline_signature(self) -> str:
        return (
            f"{self.single_channel_signature()}|"
            f"feat_ch={self.feature_channels}|fg_z={int(self.foreground_zscore)}"
        )

    def config_signature(self) -> str:
        return f"aug={self.aug_copies}:{self.aug_preset}|{self.pipeline_signature()}"


class _ExperimentRepository:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint = str(self.run_dir / "model.weights.h5")
        self._confPath = str(self.run_dir / "model.json")
        self._config: ModelConfig | None = None
        self._in_channels: int = 1

    def getCheckpointPath(self) -> str:
        return self._checkpoint

    def saveConfig(self, config: ModelConfig, in_channels: int | None = None) -> None:
        self._config = config
        if in_channels is not None:
            self._in_channels = int(in_channels)
        payload = {
            "inputWidth": config.inputWidth,
            "inputHeight": config.inputHeight,
            "classNames": list(config.classNames),
            "filterCounts": list(config.filterCounts),
            "normalizationMean": float(config.normalizationMean),
            "normalizationStd": float(config.normalizationStd),
            "randomSeed": int(config.randomSeed),
            "epochsTrained": int(config.epochsTrained),
            "parameterCount": int(config.parameterCount),
            "inChannels": int(self._in_channels),
        }
        Path(self._confPath).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def loadConfig(self) -> ModelConfig:
        if self._config is not None:
            return self._config
        raw = json.loads(Path(self._confPath).read_text(encoding="utf-8"))
        self._in_channels = int(raw.pop("inChannels", 1))
        self._config = ModelConfig(**raw)
        return self._config

    def loadModel(
        self, config: ModelConfig | None = None, in_channels: int | None = None
    ) -> CNNClassifier:
        cfg = config or self.loadConfig()
        ch = int(in_channels) if in_channels is not None else int(self._in_channels)
        model = CNNClassifier(cfg)
        dummy = tf.zeros((1, cfg.inputHeight, cfg.inputWidth, ch), dtype=tf.float32)
        model(dummy, training=False)
        model.load_weights(self._checkpoint)
        return model


class Testbench:

    CSV_FIELDS = [
        "run_id",
        "stage",
        "experiment_name",
        "seed",
        "train_per_class",
        "val_total",
        "test_total",
        "crop_to_brain",
        "n4_bias",
        "input_size",
        "aug_copies",
        "aug_preset",
        "denoise_method",
        "edge_filter",
        "contrast_method",
        "intensity_norm",
        "feature_channels",
        "foreground_zscore",
        "pipeline_signature",
        "epochs_completed",
        "best_epoch",
        "train_time_s",
        "mean_epoch_ms",
        "inference_time_ms_per_img",
        "parameter_count",
        "final_train_loss",
        "best_val_loss",
        "best_val_macro_f1",
        "best_val_min_class_f1",
        "best_val_f1_std",
        "best_val_ema_balanced_f1",
        "test_accuracy",
        "test_empirical_error",
        "test_macro_precision",
        "test_macro_sensitivity",
        "test_macro_specificity",
        "test_macro_f1",
        "test_min_class_f1",
        "test_f1_std",
        "test_cohen_kappa",
        "test_mcc",
        "test_roc_auc_macro",
        "glioma_precision",
        "glioma_sensitivity",
        "glioma_specificity",
        "glioma_f1",
        "meningioma_precision",
        "meningioma_sensitivity",
        "meningioma_specificity",
        "meningioma_f1",
        "no_tumor_precision",
        "no_tumor_sensitivity",
        "no_tumor_specificity",
        "no_tumor_f1",
        "pituitary_precision",
        "pituitary_sensitivity",
        "pituitary_specificity",
        "pituitary_f1",
        *[f"cm_{r}{c}" for r in range(4) for c in range(4)],
    ]

    EPOCH_CSV_FIELDS = [
        "run_id",
        "stage",
        "experiment_name",
        "seed",
        "denoise_method",
        "edge_filter",
        "contrast_method",
        "intensity_norm",
        "feature_channels",
        "pipeline_signature",
        "epoch",
        "is_best_epoch",
        "duration_ms",
        "train_loss",
        "train_macro_f1",
        "val_loss",
        "val_macro_f1",
        "val_min_class_f1",
        "val_f1_std",
        "val_ema_balanced_f1",
    ]

    _INT_FIELDS = {
        "run_id",
        "seed",
        "train_per_class",
        "val_total",
        "test_total",
        "crop_to_brain",
        "n4_bias",
        "input_size",
        "aug_copies",
        "foreground_zscore",
        "epochs_completed",
        "best_epoch",
        "parameter_count",
        "epoch",
        "is_best_epoch",
        *(f"cm_{r}{c}" for r in range(4) for c in range(4)),
    }

    _STR_FIELDS = {
        "stage",
        "experiment_name",
        "aug_preset",
        "denoise_method",
        "edge_filter",
        "contrast_method",
        "intensity_norm",
        "feature_channels",
        "pipeline_signature",
    }

    _OPTIONAL_FLOAT_FIELDS = {
        "inference_time_ms_per_img",
        "test_accuracy",
        "test_empirical_error",
        "test_macro_precision",
        "test_macro_sensitivity",
        "test_macro_specificity",
        "test_macro_f1",
        "test_min_class_f1",
        "test_f1_std",
        "test_cohen_kappa",
        "test_mcc",
        "test_roc_auc_macro",
        "glioma_precision",
        "glioma_sensitivity",
        "glioma_specificity",
        "glioma_f1",
        "meningioma_precision",
        "meningioma_sensitivity",
        "meningioma_specificity",
        "meningioma_f1",
        "no_tumor_precision",
        "no_tumor_sensitivity",
        "no_tumor_specificity",
        "no_tumor_f1",
        "pituitary_precision",
        "pituitary_sensitivity",
        "pituitary_specificity",
        "pituitary_f1",
        "train_loss",
        "train_macro_f1",
        "val_loss",
        "val_macro_f1",
        "val_min_class_f1",
        "val_f1_std",
        "val_ema_balanced_f1",
    }

    _SERIALIZED_OP_CLASSES = {"DenoiseFilter"}
    _MAX_OP_MEM_ENTRIES = 25000
    BASELINE_NAME = "baseline_aug8"

    def __init__(
        self,
        dataset_path: str,
        output_csv: str = "results/testbench_filters_metrics.csv",
        epoch_csv: str | None = None,
        stats_csv: str | None = None,
        work_dir: str = "results/testbench_filter_runs",
        *,
        train_per_class: int = 150,
        val_total: int = 400,
        input_size: int = 128,
        aug_copies: int = 8,
        filter_counts: Sequence[int] = (16, 32, 64),
        epochs: int = 35,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
        patience: int = 6,
        ema_alpha: float = 0.5,
        seeds: Sequence[int] = (42, 100, 2026),
        max_cached_tensors: int = 12000,
        num_workers: int | None = None,
        resume: bool = True,
        verbose: int = 1,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_csv = Path(output_csv)
        if epoch_csv is not None:
            self.epoch_csv = Path(epoch_csv)
        else:
            stem = self.output_csv.stem
            epoch_stem = (
                stem.replace("_metrics", "_epochs")
                if "_metrics" in stem
                else f"{stem}_epochs"
            )
            self.epoch_csv = self.output_csv.with_name(
                f"{epoch_stem}{self.output_csv.suffix or '.csv'}"
            )
        if stats_csv is not None:
            self.stats_csv = Path(stats_csv)
        else:
            self.stats_csv = self.output_csv.with_name("statistical_comparison.csv")

        self.work_dir = Path(work_dir)
        self.train_per_class = int(train_per_class)
        self.val_total = int(val_total)
        self.input_size = int(input_size)
        self.aug_copies = int(aug_copies)
        self.filter_counts = list(filter_counts)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.patience = int(patience)
        self.ema_alpha = float(ema_alpha)
        self.seeds = [int(s) for s in seeds] or [42]
        self.max_cached_tensors = int(max_cached_tensors)
        self.num_workers = max(1, int(num_workers or (os.cpu_count() or 4)))
        self.resume = bool(resume)
        self.verbose = int(verbose)

        if cv2 is not None:
            cv2.setNumThreads(1)
        if sitk is not None:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        try:
            tf.config.optimizer.set_jit("autoclustering")
        except Exception:
            pass

        self._class_order = list(TumorClass)
        self._class_to_idx = {c: idx for idx, c in enumerate(self._class_order)}

        for p in (
            self.output_csv.parent,
            self.epoch_csv.parent,
            self.stats_csv.parent,
            self.work_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

        self._op_disk_cache_dir = self.work_dir / "_op_cache"
        self._op_disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._op_mem_cache: dict[str, np.ndarray] = {}
        self._op_cache_lock = threading.Lock()
        self._op_exec_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self._SERIALIZED_OP_CLASSES
        }
        self._op_cache_hits = 0
        self._op_cache_misses = 0


        self._common_prefix_cache: dict[tuple[str, str], np.ndarray] = {}
        self._single_ch_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._feature_ch_cache: dict[tuple[str, str, str, str], np.ndarray] = {}
        self._fitted_nyul_cache: dict[tuple[str, str, int], NyulNormalizer] = {}
        self._cache_lock = threading.Lock()

        self._raw_decode_cache: dict[tuple[int, str], np.ndarray] = {}
        self._raw_decode_lock = threading.Lock()
        self._manifest_records: List[ImageRecord] | None = None
        self._splits_by_seed: dict[int, DatasetSplit] = {}


        self._baseline_ema: float | None = None
        self._current_max_ema: float | None = None
        self._current_max_name: str = ""
        self._verify_base_f1_by_seed: dict[int, float] = {}
        self._verify_max_f1_by_seed: dict[int, tuple[float, str]] = {}

        self._suite_start = time.perf_counter()
        self._run_counter = 0
        self._completed_runs: dict[tuple[str, str, int], dict] = {}
        self._completed_by_sig: dict[tuple[str, int], dict] = {}
        self._run_dirs_by_sig: dict[tuple[str, int], Path] = {}

        if self.resume:
            self._load_and_sanitize_existing_results()
            self._sanitize_existing_epoch_results()
        else:
            for p in (self.output_csv, self.epoch_csv, self.stats_csv):
                if p.exists():
                    p.unlink()

    def _suite_elapsed_str(self) -> str:
        return _format_duration(time.perf_counter() - self._suite_start)

    def _log(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _phase_timer(self, label: str, *, indent: str = "    ") -> _LivePhaseTimer:
        return _LivePhaseTimer(
            label,
            lambda msg: self._log(msg, level=1),
            enabled=self.verbose >= 1,
            heartbeat_s=5.0,
            indent=indent,
        )



    def _archive_incompatible_csv(self, csv_path: Path) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = csv_path.with_name(f"{csv_path.stem}.legacy_{ts}{csv_path.suffix}")
        os.replace(csv_path, backup_path)
        self._log(f"[RESUME] Archived incompatible CSV '{csv_path}' -> '{backup_path}'.")

    def _parse_csv_row(
        self, raw_row: dict, expected_fields: Sequence[str]
    ) -> dict | None:
        if None in raw_row:
            return None
        parsed: dict = {}
        for field in expected_fields:
            if field not in raw_row or raw_row[field] is None:
                return None
            val = raw_row[field]
            if field in self._STR_FIELDS:
                if val == "":
                    return None
                parsed[field] = str(val)
            elif field in self._INT_FIELDS:
                try:
                    parsed[field] = int(val)
                except (ValueError, TypeError):
                    return None
            else:
                if field in self._OPTIONAL_FLOAT_FIELDS and val == "":
                    parsed[field] = ""
                else:
                    try:
                        fval = float(val)
                        if not math.isfinite(fval):
                            return None
                        parsed[field] = fval
                    except (ValueError, TypeError):
                        return None
        return parsed

    def _load_and_sanitize_existing_results(self) -> None:
        if not self.output_csv.exists() or self.output_csv.stat().st_size == 0:
            return
        valid_rows: List[dict] = []
        had_corrupt = False
        with self.output_csv.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != self.CSV_FIELDS:
                fh.close()
                self._archive_incompatible_csv(self.output_csv)
                return
            for raw_row in reader:
                parsed = self._parse_csv_row(raw_row, self.CSV_FIELDS)
                if parsed is None:
                    had_corrupt = True
                    continue
                valid_rows.append(parsed)

        if had_corrupt:
            tmp = self.output_csv.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                w.writeheader()
                w.writerows(valid_rows)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.output_csv)

        for row in valid_rows:
            self._register_completed_row(row)
        if valid_rows:
            self._log(
                f"[RESUME] Loaded {len(valid_rows)} completed run(s) from '{self.output_csv}'."
            )

    def _register_completed_row(self, row: dict) -> None:
        key = (row["stage"], row["experiment_name"], int(row["seed"]))
        self._completed_runs[key] = row
        cfg_sig = f"aug={row['aug_copies']}:{row['aug_preset']}|{row['pipeline_signature']}"
        self._completed_by_sig[(cfg_sig, int(row["seed"]))] = row
        run_id = int(row["run_id"])
        self._run_counter = max(self._run_counter, run_id)
        cand_dir = (
            self.work_dir
            / f"run_{run_id:04d}_{row['stage']}_{row['experiment_name']}_s{int(row['seed'])}"
        )
        if cand_dir.exists():
            self._run_dirs_by_sig[(cfg_sig, int(row["seed"]))] = cand_dir

    def _sanitize_existing_epoch_results(self) -> None:
        if not self.epoch_csv.exists() or self.epoch_csv.stat().st_size == 0:
            return
        valid_rows: List[dict] = []
        needs_rewrite = False
        with self.epoch_csv.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != self.EPOCH_CSV_FIELDS:
                fh.close()
                self._archive_incompatible_csv(self.epoch_csv)
                return
            for raw_row in reader:
                parsed = self._parse_csv_row(raw_row, self.EPOCH_CSV_FIELDS)
                if parsed is None or (
                    parsed["stage"],
                    parsed["experiment_name"],
                    int(parsed["seed"]),
                ) not in self._completed_runs:
                    needs_rewrite = True
                    continue
                valid_rows.append(parsed)

        if needs_rewrite:
            tmp = self.epoch_csv.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.EPOCH_CSV_FIELDS)
                w.writeheader()
                w.writerows(valid_rows)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.epoch_csv)

    def _append_csv_rows(
        self, path: Path, fieldnames: Sequence[str], rows: List[dict]
    ) -> None:
        if not rows:
            return
        write_header = (not path.exists()) or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())


    def build_stratified_split(self, seed: int) -> DatasetSplit:
        if seed in self._splits_by_seed:
            return self._splits_by_seed[seed]

        t0 = time.perf_counter()
        if self._manifest_records is None:
            with self._phase_timer(
                f"Scanning dataset manifest (seed={seed})", indent="[DATASET] "
            ):
                manager = DatasetManager(self.dataset_path)
                manifest = manager.loadDataset()
                self._manifest_records = list(manifest.getSamples())

        all_records = self._manifest_records
        n_classes = len(self._class_order)
        val_per_class = self.val_total // n_classes
        train_pool: Dict[TumorClass, List[ImageRecord]] = {c: [] for c in self._class_order}
        test_records: List[ImageRecord] = []

        for rec in all_records:
            if rec.split == SplitType.TEST:
                test_records.append(rec)
            else:
                train_pool[rec.label].append(rec)

        rng = random.Random(seed)
        train_records: List[ImageRecord] = []
        val_records: List[ImageRecord] = []
        for c in self._class_order:
            pool = list(train_pool[c])
            rng.shuffle(pool)
            required = self.train_per_class + val_per_class
            if len(pool) < required:
                raise ValueError(
                    f"Class {c.name} has {len(pool)} non-test images, need {required}."
                )
            for rec in pool[: self.train_per_class]:
                train_records.append(ImageRecord(rec.imagePath, c, SplitType.TRAINING))
            for rec in pool[self.train_per_class : required]:
                val_records.append(ImageRecord(rec.imagePath, c, SplitType.VALIDATION))

        if not test_records:
            for c in self._class_order:
                pool = list(train_pool[c])
                rng.shuffle(pool)
                required = self.train_per_class + val_per_class
                for rec in pool[required:]:
                    test_records.append(ImageRecord(rec.imagePath, c, SplitType.TEST))

        split = DatasetSplit(train_records, val_records, test_records)
        self._splits_by_seed[seed] = split
        elapsed_s = time.perf_counter() - t0
        self._log(
            f"[DATASET] Split (seed={seed}) ready in {elapsed_s:.2f}s -> "
            f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
        )
        return split


    @staticmethod
    def build_common_prefix_ops(spec: ExperimentSpec) -> List[ImageOperator]:
        ops: List[ImageOperator] = []
        if spec.crop_to_brain:
            ops.append(BrainCropper(target_size=spec.input_size, margin_ratio=0.06))
            ops.append(ForegroundLCCCleaner(threshold=0.04, close_radius=3))
        if spec.n4_bias:
            ops.append(ParallelN4BiasFieldCorrector())
        return ops

    @staticmethod
    def build_filter_ops(spec: ExperimentSpec) -> List[ImageOperator]:
        ops: List[ImageOperator] = []
        if spec.denoise_method != "none":
            ops.append(FastDenoiseFilter(method=spec.denoise_method))

        if spec.intensity_norm == "percentile_clip":
            ops.append(PercentileClipNormalizer(low_pct=0.5, high_pct=99.9))
        elif spec.intensity_norm == "WhiteStripe":
            ops.append(WhiteStripeNormalizer())
        elif spec.intensity_norm == "Nyul":
            ops.append(NyulNormalizer())

        if spec.edge_filter == "TopHatBottomHat":
            ops.append(TopHatBottomHatFilter(radius=4, alpha_top=0.35, alpha_bot=0.20))

        if spec.contrast_method == "CLAHE":
            ops.append(CLAHEFilter(clip_limit=2.0, tile_grid_size=(8, 8)))
        elif spec.contrast_method.startswith("CLAHE_"):
            _, clip_str, grid_str = spec.contrast_method.split("_")
            grid = int(grid_str)
            ops.append(CLAHEFilter(clip_limit=float(clip_str), tile_grid_size=(grid, grid)))
        return ops

    @classmethod
    def build_pipeline(cls, spec: ExperimentSpec) -> List[ImageOperator]:
        return cls.build_common_prefix_ops(spec) + cls.build_filter_ops(spec)

    def _op_cache_tag(self, op: ImageOperator) -> str:
        cls_name = type(op).__name__
        attrs = [
            f"{k}={v}"
            for k, v in sorted(getattr(op, "__dict__", {}).items())
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, tuple, list))
        ]
        return f"{cls_name}({','.join(attrs)})"

    def _memoize_stateless_operators(self, ops: List[ImageOperator]) -> List[ImageOperator]:
        for op in ops:
            if isinstance(op, NyulNormalizer) or hasattr(op, "fit"):
                continue
            orig_apply = getattr(op, "apply", None)
            if not callable(orig_apply):
                continue

            cls_name = type(op).__name__
            tag = self._op_cache_tag(op)
            exec_lock = self._op_exec_locks.get(cls_name)
            numpy_fn = getattr(op, "apply_numpy", None)

            def _make_cached_apply(
                fn: Callable,
                np_fn: Callable | None,
                op_tag: str,
                op_lock: threading.Lock | None,
                op_obj: ImageOperator,
            ):
                def _cached_apply(image):
                    is_tensor = tf.is_tensor(image)
                    in_dtype = image.dtype if is_tensor else None
                    arr = np.ascontiguousarray(image.numpy() if is_tensor else image)

                    h = hashlib.blake2b(op_tag.encode("utf-8"), digest_size=16)
                    h.update(str(arr.shape).encode("utf-8"))
                    h.update(arr.tobytes())
                    hex_key = h.hexdigest()

                    def _wrap_out(np_arr: np.ndarray):
                        if is_tensor:
                            return tf.convert_to_tensor(np_arr, dtype=in_dtype)
                        if isinstance(op_obj, _NUMPY_NATIVE_OPS) or np_fn is not None:
                            return np_arr.copy()
                        return tf.convert_to_tensor(np_arr, dtype=tf.float32)

                    with self._op_cache_lock:
                        hit = self._op_mem_cache.get(hex_key)
                        if hit is not None:
                            self._op_cache_hits += 1
                    if hit is not None:
                        return _wrap_out(hit)

                    disk_file = self._op_disk_cache_dir / f"{hex_key}.npy"
                    if disk_file.exists():
                        try:
                            loaded = np.load(disk_file)
                            with self._op_cache_lock:
                                self._op_cache_hits += 1
                                if len(self._op_mem_cache) < self._MAX_OP_MEM_ENTRIES:
                                    self._op_mem_cache[hex_key] = loaded
                            return _wrap_out(loaded)
                        except Exception:
                            pass

                    def _compute():
                        if callable(np_fn):
                            had_ch = arr.ndim == 3 and arr.shape[-1] == 1
                            img2d = arr[:, :, 0] if had_ch else arr
                            res2d = np.clip(np_fn(img2d.astype(np.float32)), 0.0, 1.0).astype(
                                np.float32
                            )
                            return res2d[:, :, np.newaxis] if had_ch else res2d

                        call_inp = (
                            arr
                            if isinstance(op_obj, _NUMPY_NATIVE_OPS) and not is_tensor
                            else (
                                image
                                if is_tensor
                                else tf.convert_to_tensor(arr, dtype=tf.float32)
                            )
                        )
                        res = fn(call_inp)
                        return np.array(
                            res.numpy() if tf.is_tensor(res) else res, copy=True
                        )

                    if op_lock is not None:
                        with op_lock:
                            with self._op_cache_lock:
                                hit = self._op_mem_cache.get(hex_key)
                                if hit is not None:
                                    self._op_cache_hits += 1
                            if hit is not None:
                                return _wrap_out(hit)
                            out_np = _compute()
                    else:
                        out_np = _compute()

                    with self._op_cache_lock:
                        self._op_cache_misses += 1
                        if len(self._op_mem_cache) >= self._MAX_OP_MEM_ENTRIES:
                            evict_n = max(1, self._MAX_OP_MEM_ENTRIES // 10)
                            for old_k in list(self._op_mem_cache.keys())[:evict_n]:
                                del self._op_mem_cache[old_k]
                        self._op_mem_cache[hex_key] = out_np

                    tmp_npy = (
                        self._op_disk_cache_dir
                        / f"{hex_key}_{threading.get_ident()}.tmp.npy"
                    )
                    try:
                        np.save(tmp_npy, out_np)
                        os.replace(tmp_npy, disk_file)
                    except Exception:
                        tmp_npy.unlink(missing_ok=True)

                    return _wrap_out(out_np)

                return _cached_apply

            op.apply = _make_cached_apply(orig_apply, numpy_fn, tag, exec_lock, op)
        return ops

    def _decode_raw_image(self, image_path: str, target_size: int) -> np.ndarray:
        key = (int(target_size), str(image_path))
        with self._raw_decode_lock:
            cached = self._raw_decode_cache.get(key)
        if cached is not None:
            return cached.copy()

        if cv2 is not None:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                arr = img.astype(np.float32) / 255.0
                h, w = arr.shape
                if h != target_size or w != target_size:
                    scale = min(target_size / max(1, h), target_size / max(1, w))
                    nh = max(1, int(round(h * scale)))
                    nw = max(1, int(round(w * scale)))
                    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    resized = cv2.resize(arr, (nw, nh), interpolation=interp)
                    canvas = np.zeros((target_size, target_size), dtype=np.float32)
                    y0 = (target_size - nh) // 2
                    x0 = (target_size - nw) // 2
                    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
                    arr = canvas
                out = arr[..., np.newaxis]
                with self._raw_decode_lock:
                    if len(self._raw_decode_cache) < self.max_cached_tensors:
                        self._raw_decode_cache[key] = out
                return out.copy()

        raw_bytes = tf.io.read_file(str(image_path))
        img_t = tf.io.decode_image(raw_bytes, channels=1, expand_animations=False)
        img_t = tf.image.convert_image_dtype(img_t, tf.float32)
        img_t = tf.image.resize_with_pad(img_t, target_size, target_size)
        out = img_t.numpy()
        with self._raw_decode_lock:
            if len(self._raw_decode_cache) < self.max_cached_tensors:
                self._raw_decode_cache[key] = out
        return out.copy()

    @staticmethod
    def _attach_feature_channels(img: np.ndarray, feature_channels: str) -> np.ndarray:
        if feature_channels == "none":
            return img
        base2d = img[..., 0]
        if feature_channels == "lbp":
            return np.stack([base2d, _compute_lbp_channel(base2d)], axis=-1).astype(np.float32)
        if feature_channels == "log":
            log1, log25 = _compute_log_channels(base2d, sigmas=(1.0, 2.5))
            return np.stack([base2d, log1, log25], axis=-1).astype(np.float32)
        if feature_channels == "lbp_log":
            lbp2d = _compute_lbp_channel(base2d)
            log1, log25 = _compute_log_channels(base2d, sigmas=(1.0, 2.5))
            return np.stack([base2d, lbp2d, log1, log25], axis=-1).astype(np.float32)
        return img

    @staticmethod
    def _put_bounded_cache(cache: dict, key, val: np.ndarray, max_entries: int) -> None:
        if len(cache) >= max_entries:
            evict_n = max(1, max_entries // 8)
            for old_k in list(cache.keys())[:evict_n]:
                del cache[old_k]
        cache[key] = val

    def _prepare_split_arrays(
        self,
        split: DatasetSplit,
        spec: ExperimentSpec,
        *,
        include_test: bool = False,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float,
        float,
    ]:
        target_size = int(spec.input_size)
        prefix_sig = spec.common_prefix_signature()
        single_ch_sig = spec.single_channel_signature()
        nyul_scope = f"seed={spec.seed}" if spec.intensity_norm == "Nyul" else "global"

        common_ops = self._memoize_stateless_operators(self.build_common_prefix_ops(spec))
        filter_ops = self._memoize_stateless_operators(self.build_filter_ops(spec))

        stateful_idx = next(
            (
                i
                for i, op in enumerate(filter_ops)
                if isinstance(op, NyulNormalizer) or hasattr(op, "fit")
            ),
            None,
        )
        pre_nyul_ops = filter_ops[:stateful_idx] if stateful_idx is not None else filter_ops
        nyul_op = filter_ops[stateful_idx] if stateful_idx is not None else None
        post_nyul_ops = filter_ops[stateful_idx + 1 :] if stateful_idx is not None else []

        def _apply_seq(arr: np.ndarray, op_list: List[ImageOperator]) -> np.ndarray:
            cur = arr
            for op in op_list:
                inp = (
                    cur
                    if isinstance(op, _NUMPY_NATIVE_OPS)
                    else tf.convert_to_tensor(cur, dtype=tf.float32)
                )
                res = op.apply(inp)
                cur = np.asarray(res.numpy() if tf.is_tensor(res) else res, dtype=np.float32)
                if cur.ndim == 2:
                    cur = cur[..., np.newaxis]
            if cur.shape[0] != target_size or cur.shape[1] != target_size:
                if cv2 is not None:
                    cur = cv2.resize(
                        cur[..., 0],
                        (target_size, target_size),
                        interpolation=cv2.INTER_AREA,
                    )[..., np.newaxis]
                else:
                    cur = tf.image.resize_with_pad(cur, target_size, target_size).numpy()
            return cur

        train_recs, val_recs, test_recs = _extract_split_records(split)
        records = list(train_recs) + list(val_recs) + (list(test_recs) if include_test else [])
        n_train, n_val = len(train_recs), len(val_recs)

        total = len(records)
        done_count = 0
        progress_lock = threading.Lock()
        t_start = time.perf_counter()
        last_log_t = t_start

        def _tick_progress() -> None:
            nonlocal done_count, last_log_t
            if self.verbose < 1:
                return
            with progress_lock:
                done_count += 1
                now = time.perf_counter()
                if done_count == total or (now - last_log_t) >= 10.0:
                    last_log_t = now
                    elapsed = max(1e-3, now - t_start)
                    rate = done_count / elapsed
                    eta = (total - done_count) / max(rate, 1e-6)
                    self._log(
                        f"    [PREP] {done_count:4d}/{total} ({100.0 * done_count / total:5.1f}%) | "
                        f"{rate:6.1f} img/s | elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s"
                    )

        def _get_common_prefix(rec: ImageRecord) -> np.ndarray:
            p_str = str(rec.imagePath)
            pkey = (prefix_sig, p_str)
            with self._cache_lock:
                cached = self._common_prefix_cache.get(pkey)
            if cached is not None:
                return cached
            raw = self._decode_raw_image(p_str, target_size)
            out = _apply_seq(raw, common_ops)
            with self._cache_lock:
                self._put_bounded_cache(
                    self._common_prefix_cache, pkey, out, self.max_cached_tensors
                )
            return out

        def _prepare_single_channel_fast(rec: ImageRecord) -> np.ndarray:
            p_str = str(rec.imagePath)
            skey = (single_ch_sig, nyul_scope, p_str)
            with self._cache_lock:
                cached = self._single_ch_cache.get(skey)
            if cached is not None:
                _tick_progress()
                return cached
            base = _get_common_prefix(rec)
            out = _apply_seq(base, pre_nyul_ops)
            with self._cache_lock:
                self._put_bounded_cache(
                    self._single_ch_cache, skey, out, self.max_cached_tensors
                )
            _tick_progress()
            return out

        if nyul_op is None:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                single_imgs = list(pool.map(_prepare_single_channel_fast, records))
        else:
            # Stateful Nyul path: check single-channel cache first, otherwise fit once per (denoise, seed)
            nyul_cache_key = (prefix_sig, spec.denoise_method, int(spec.seed))

            def _get_pre_nyul(rec: ImageRecord) -> np.ndarray:
                base = _get_common_prefix(rec)
                return _apply_seq(base, pre_nyul_ops)

            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                pre_imgs = list(pool.map(_get_pre_nyul, records))

            with self._cache_lock:
                fitted_nyul = self._fitted_nyul_cache.get(nyul_cache_key)
            if fitted_nyul is None:
                nyul_op.fit(
                    [im[..., 0] if im.ndim == 3 else im for im in pre_imgs[:n_train]]
                )
                with self._cache_lock:
                    self._fitted_nyul_cache[nyul_cache_key] = nyul_op
                fitted_nyul = nyul_op

            seq_after = [fitted_nyul] + post_nyul_ops

            def _apply_post_nyul(item: tuple[ImageRecord, np.ndarray]) -> np.ndarray:
                rec, im = item
                p_str = str(rec.imagePath)
                skey = (single_ch_sig, nyul_scope, p_str)
                with self._cache_lock:
                    cached = self._single_ch_cache.get(skey)
                if cached is not None:
                    _tick_progress()
                    return cached
                out = _apply_seq(im, seq_after)
                with self._cache_lock:
                    self._put_bounded_cache(
                        self._single_ch_cache, skey, out, self.max_cached_tensors
                    )
                _tick_progress()
                return out

            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                single_imgs = list(pool.map(_apply_post_nyul, zip(records, pre_imgs)))

        if spec.feature_channels == "none":
            final_imgs = [im.copy() for im in single_imgs]
        else:

            def _get_multi_channel(item: tuple[ImageRecord, np.ndarray]) -> np.ndarray:
                rec, base_im = item
                fkey = (single_ch_sig, nyul_scope, spec.feature_channels, str(rec.imagePath))
                with self._cache_lock:
                    cached = self._feature_ch_cache.get(fkey)
                if cached is not None:
                    return cached.copy()
                stacked = self._attach_feature_channels(base_im, spec.feature_channels)
                with self._cache_lock:
                    self._put_bounded_cache(
                        self._feature_ch_cache,
                        fkey,
                        stacked,
                        max(2000, self.max_cached_tensors // 2),
                    )
                return stacked.copy()

            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                final_imgs = list(pool.map(_get_multi_channel, zip(records, single_imgs)))

        X_train = np.stack(final_imgs[:n_train], axis=0).astype(np.float32)
        X_val = np.stack(final_imgs[n_train : n_train + n_val], axis=0).astype(np.float32)
        X_test = (
            np.stack(final_imgs[n_train + n_val :], axis=0).astype(np.float32)
            if include_test
            else None
        )

        y_train = np.array([self._class_to_idx[r.label] for r in train_recs], dtype=np.int32)
        y_val = np.array([self._class_to_idx[r.label] for r in val_recs], dtype=np.int32)
        y_test = (
            np.array([self._class_to_idx[r.label] for r in test_recs], dtype=np.int32)
            if include_test
            else None
        )

        fg_train = X_train[..., 0] > 1e-4
        fg_val = X_val[..., 0] > 1e-4
        fg_test = X_test[..., 0] > 1e-4 if X_test is not None else None

        base_mean, base_std = 0.0, 1.0
        for ch in range(X_train.shape[-1]):
            tr_ch = X_train[..., ch]
            fg_vals = tr_ch[fg_train] if spec.foreground_zscore else tr_ch.ravel()
            c_mean = float(fg_vals.mean()) if fg_vals.size > 0 else float(tr_ch.mean())
            c_std = max(float(fg_vals.std()) if fg_vals.size > 0 else float(tr_ch.std()), 1e-6)
            for arr, fg in ((X_train, fg_train), (X_val, fg_val), (X_test, fg_test)):
                if arr is None:
                    continue
                arr_ch = arr[..., ch]
                if spec.foreground_zscore:
                    arr_ch[fg] = (arr_ch[fg] - c_mean) / c_std
                    arr_ch[~fg] = 0.0
                else:
                    arr[..., ch] = (arr_ch - c_mean) / c_std
            if ch == 0:
                base_mean, base_std = c_mean, c_std

        return X_train, y_train, X_val, y_val, X_test, y_test, base_mean, base_std


    @staticmethod
    def _class_f1_stats_from_cm(cm: np.ndarray) -> tuple[float, float, float, float]:
        f1s = []
        for c in range(cm.shape[0]):
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)
            denom = 2.0 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
        arr = np.asarray(f1s, dtype=np.float64)
        macro_f1 = float(np.mean(arr))
        min_f1 = float(np.min(arr))
        f1_std = float(np.std(arr))
        balanced_f1 = 0.70 * macro_f1 + 0.30 * min_f1
        return macro_f1, min_f1, f1_std, balanced_f1

    @staticmethod
    def _augment_batch_gpu(imgs: tf.Tensor, preset: str) -> tf.Tensor:
        """Vectorized batch-level augmentation executed directly on GPU."""
        shape = tf.shape(imgs)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]

        flip_mask = tf.random.uniform([b, 1, 1, 1]) < 0.5
        imgs = tf.where(flip_mask, tf.reverse(imgs, axis=[2]), imgs)

        max_dy = tf.maximum(1, tf.cast(tf.cast(h, tf.float32) * 0.06, tf.int32))
        max_dx = tf.maximum(1, tf.cast(tf.cast(w, tf.float32) * 0.06, tf.int32))
        dy = tf.random.uniform([b, 1], -max_dy, max_dy + 1, dtype=tf.int32)
        dx = tf.random.uniform([b, 1], -max_dx, max_dx + 1, dtype=tf.int32)

        y_idx = tf.math.mod(tf.range(h)[None, :] - dy, h)
        x_idx = tf.math.mod(tf.range(w)[None, :] - dx, w)
        imgs = tf.gather(imgs, y_idx, batch_dims=1, axis=1)
        imgs = tf.gather(imgs, x_idx, batch_dims=1, axis=2)

        if preset == "mri_full":
            ch0 = imgs[..., :1]
            ch0_mean = tf.reduce_mean(ch0, axis=[1, 2], keepdims=True)
            contrast = tf.random.uniform([b, 1, 1, 1], 0.88, 1.12)
            brightness = tf.random.uniform([b, 1, 1, 1], -0.12, 0.12)
            noise_std = tf.random.uniform([b, 1, 1, 1], 0.0, 0.04)

            ch0 = (ch0 - ch0_mean) * contrast + ch0_mean + brightness
            ch0 = ch0 + tf.random.normal(tf.shape(ch0)) * noise_std
            imgs = tf.cond(
                c > 1,
                lambda: tf.concat([ch0, imgs[..., 1:]], axis=-1),
                lambda: ch0,
            )

            do_erase = tf.random.uniform([b, 1, 1, 1]) < 0.5
            cut_h = tf.maximum(2, tf.cast(tf.cast(h, tf.float32) * 0.12, tf.int32))
            cut_w = tf.maximum(2, tf.cast(tf.cast(w, tf.float32) * 0.12, tf.int32))
            y0 = tf.random.uniform([b, 1, 1, 1], 0, tf.maximum(1, h - cut_h), dtype=tf.int32)
            x0 = tf.random.uniform([b, 1, 1, 1], 0, tf.maximum(1, w - cut_w), dtype=tf.int32)
            ys = tf.reshape(tf.range(h), [1, h, 1, 1])
            xs = tf.reshape(tf.range(w), [1, 1, w, 1])
            patch_mask = (ys >= y0) & (ys < y0 + cut_h) & (xs >= x0) & (xs < x0 + cut_w)
            imgs = tf.where(do_erase & patch_mask, tf.zeros_like(imgs), imgs)

        return imgs

    def _build_model(
        self, spec: ExperimentSpec, mean: float, std: float
    ) -> tuple[CNNClassifier, ModelConfig]:
        class_names = [c.name for c in self._class_order]
        cfg_kwargs = {
            "inputWidth": int(spec.input_size),
            "inputHeight": int(spec.input_size),
            "classNames": class_names,
            "filterCounts": list(self.filter_counts),
            "normalizationMean": float(mean),
            "normalizationStd": float(std),
            "randomSeed": int(spec.seed),
            "epochsTrained": 0,
            "parameterCount": 0,
        }
        sig_params = inspect.signature(ModelConfig).parameters
        cfg = ModelConfig(**{k: v for k, v in cfg_kwargs.items() if k in sig_params})

        cnn_params = inspect.signature(CNNClassifier).parameters
        if "dropoutRate" in cnn_params:
            model = CNNClassifier(cfg, dropoutRate=self.dropout_rate)
        elif "dropout_rate" in cnn_params:
            model = CNNClassifier(cfg, dropout_rate=self.dropout_rate)
        else:
            model = CNNClassifier(cfg)

        dummy = tf.zeros(
            (1, spec.input_size, spec.input_size, spec.num_channels()), dtype=tf.float32
        )
        model(dummy, training=False)
        cfg.parameterCount = int(model.count_params())
        return model, cfg

    @staticmethod
    def _empty_test_metrics() -> dict:
        class_keys = ["glioma", "meningioma", "no_tumor", "pituitary"]
        per_class = {
            f"{cname}_{m}": ""
            for cname in class_keys
            for m in ("precision", "sensitivity", "specificity", "f1")
        }
        cm_flat = {f"cm_{r}{c}": 0 for r in range(4) for c in range(4)}
        return {
            "test_total": 0,
            "inference_time_ms_per_img": "",
            "test_accuracy": "",
            "test_empirical_error": "",
            "test_macro_precision": "",
            "test_macro_sensitivity": "",
            "test_macro_specificity": "",
            "test_macro_f1": "",
            "test_min_class_f1": "",
            "test_f1_std": "",
            "test_cohen_kappa": "",
            "test_mcc": "",
            "test_roc_auc_macro": "",
            **per_class,
            **cm_flat,
        }

    def _train_and_evaluate_spec(
        self, spec: ExperimentSpec, run_id: int, *, include_test: bool = False
    ) -> dict:
        tf.keras.utils.set_random_seed(spec.seed)
        np.random.seed(spec.seed)
        random.seed(spec.seed)

        n_classes = len(self._class_order)
        in_channels = spec.num_channels()
        split = self.build_stratified_split(spec.seed)
        X_train, y_train, X_val, y_val, X_test, y_test, mean, std = (
            self._prepare_split_arrays(split, spec, include_test=include_test)
        )

        run_dir = self.work_dir / f"run_{run_id:04d}_{spec.stage}_{spec.name}_s{spec.seed}"
        repo = _ExperimentRepository(run_dir)
        model, model_cfg = self._build_model(spec, mean, std)

        X_train_t = tf.constant(X_train, dtype=tf.float32)
        y_train_t = tf.constant(y_train, dtype=tf.int32)
        X_val_t = tf.constant(X_val, dtype=tf.float32)
        y_val_t = tf.constant(y_val, dtype=tf.int32)

        copies = max(0, int(spec.aug_copies))
        n_base = int(X_train.shape[0])
        n_train_total = n_base * (copies + 1)
        n_val_total = int(X_val.shape[0])
        batch_size = int(self.batch_size)
        steps_per_epoch = max(1, math.ceil(n_train_total / batch_size))

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=max(1, self.epochs * steps_per_epoch),
            alpha=0.01,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.weight_decay,
            clipnorm=1.0,
        )
        ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        @tf.function
        def _run_train_epoch():
            base_idx = tf.tile(tf.range(n_base, dtype=tf.int32), [copies + 1])
            shuffled_idx = tf.random.shuffle(base_idx)
            loss_sum = tf.constant(0.0, dtype=tf.float64)
            batches = tf.constant(0, dtype=tf.int32)
            cm = tf.zeros((n_classes, n_classes), dtype=tf.int64)

            for start in tf.range(0, n_train_total, batch_size):
                end = tf.minimum(start + batch_size, n_train_total)
                b_idx = shuffled_idx[start:end]
                bx = tf.gather(X_train_t, b_idx)
                by = tf.gather(y_train_t, b_idx)

                if copies > 0:
                    is_aug = tf.reshape(b_idx >= n_base, [-1, 1, 1, 1])
                    bx = tf.where(is_aug, self._augment_batch_gpu(bx, spec.aug_preset), bx)

                with tf.GradientTape() as tape:
                    logits = model(bx, training=True)
                    loss = ce_loss(tf.one_hot(by, depth=n_classes), logits)
                    if model.losses:
                        loss = loss + tf.add_n(model.losses)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                loss_sum += tf.cast(loss, tf.float64)
                batches += 1
                preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
                cm += tf.math.confusion_matrix(
                    by, preds, num_classes=n_classes, dtype=tf.int64
                )

            return loss_sum / tf.cast(tf.maximum(1, batches), tf.float64), cm

        @tf.function
        def _run_val_epoch():
            loss_sum = tf.constant(0.0, dtype=tf.float64)
            batches = tf.constant(0, dtype=tf.int32)
            cm = tf.zeros((n_classes, n_classes), dtype=tf.int64)

            for start in tf.range(0, n_val_total, batch_size):
                end = tf.minimum(start + batch_size, n_val_total)
                bx = X_val_t[start:end]
                by = y_val_t[start:end]
                logits = model(bx, training=False)
                loss = ce_loss(tf.one_hot(by, depth=n_classes), logits)

                loss_sum += tf.cast(loss, tf.float64)
                batches += 1
                preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
                cm += tf.math.confusion_matrix(
                    by, preds, num_classes=n_classes, dtype=tf.int64
                )

            return loss_sum / tf.cast(tf.maximum(1, batches), tf.float64), cm

        best_ema_bal = -1.0
        best_val_macro = -1.0
        best_val_min_c = -1.0
        best_val_std = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        ema_raw = 0.0
        epochs_no_improve = 0
        epoch_rows: List[dict] = []
        epoch_durations_ms: List[float] = []
        final_train_loss = 0.0

        train_t0 = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            ep_t0 = time.perf_counter()

            tr_loss_t, tr_cm_t = _run_train_epoch()
            val_loss_t, val_cm_t = _run_val_epoch()

            train_loss = float(tr_loss_t.numpy())
            val_loss = float(val_loss_t.numpy())
            tr_cm = tr_cm_t.numpy()
            val_cm = val_cm_t.numpy()

            train_macro, _, _, _ = self._class_f1_stats_from_cm(tr_cm)
            val_macro, val_min_c, val_std, val_bal = self._class_f1_stats_from_cm(val_cm)

            ema_raw = self.ema_alpha * val_bal + (1.0 - self.ema_alpha) * ema_raw
            ema_bal = ema_raw / (1.0 - (1.0 - self.ema_alpha) ** epoch)

            ep_ms = (time.perf_counter() - ep_t0) * 1000.0
            epoch_durations_ms.append(ep_ms)
            final_train_loss = train_loss

            is_best = (ema_bal > best_ema_bal + 1e-5) or (
                abs(ema_bal - best_ema_bal) <= 1e-5 and val_loss < best_val_loss
            )
            if is_best:
                best_ema_bal = ema_bal
                best_val_macro = val_macro
                best_val_min_c = val_min_c
                best_val_std = val_std
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                model.save_weights(repo.getCheckpointPath())
                model_cfg.epochsTrained = epoch
                repo.saveConfig(model_cfg, in_channels=in_channels)
            else:
                epochs_no_improve += 1

            epoch_rows.append(
                {
                    "run_id": run_id,
                    "stage": spec.stage,
                    "experiment_name": spec.name,
                    "seed": spec.seed,
                    "denoise_method": spec.denoise_method,
                    "edge_filter": spec.edge_filter,
                    "contrast_method": spec.contrast_method,
                    "intensity_norm": spec.intensity_norm,
                    "feature_channels": spec.feature_channels,
                    "pipeline_signature": spec.pipeline_signature(),
                    "epoch": epoch,
                    "is_best_epoch": int(is_best),
                    "duration_ms": round(ep_ms, 2),
                    "train_loss": round(train_loss, 6),
                    "train_macro_f1": round(train_macro, 6),
                    "val_loss": round(val_loss, 6),
                    "val_macro_f1": round(val_macro, 6),
                    "val_min_class_f1": round(val_min_c, 6),
                    "val_f1_std": round(val_std, 6),
                    "val_ema_balanced_f1": round(ema_bal, 6),
                }
            )

            self._log(
                f"      ep {epoch:02d}/{self.epochs:02d} ({ep_ms:5.0f}ms) | "
                f"tr_loss={train_loss:.4f} | val_f1={val_macro:.4f} "
                f"min_c={val_min_c:.4f} f1_ema={ema_bal:.4f}{'*' if is_best else ''}",
                level=2,
            )

            if epoch >= 10 and epochs_no_improve >= self.patience:
                break

        train_time_s = time.perf_counter() - train_t0
        epochs_completed = len(epoch_rows)

        if include_test and X_test is not None and y_test is not None:
            best_model = repo.loadModel(model_cfg, in_channels=in_channels)
            test_metrics = self._evaluate_test_set(best_model, X_test, y_test)
        else:
            test_metrics = self._empty_test_metrics()

        summary_row = {
            "run_id": run_id,
            "stage": spec.stage,
            "experiment_name": spec.name,
            "seed": spec.seed,
            "train_per_class": self.train_per_class,
            "val_total": self.val_total,
            "crop_to_brain": int(spec.crop_to_brain),
            "n4_bias": int(spec.n4_bias),
            "input_size": int(spec.input_size),
            "aug_copies": int(spec.aug_copies),
            "aug_preset": spec.aug_preset,
            "denoise_method": spec.denoise_method,
            "edge_filter": spec.edge_filter,
            "contrast_method": spec.contrast_method,
            "intensity_norm": spec.intensity_norm,
            "feature_channels": spec.feature_channels,
            "foreground_zscore": int(spec.foreground_zscore),
            "pipeline_signature": spec.pipeline_signature(),
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "train_time_s": round(train_time_s, 3),
            "mean_epoch_ms": round(float(np.mean(epoch_durations_ms)), 2),
            "parameter_count": int(model_cfg.parameterCount),
            "final_train_loss": round(final_train_loss, 6),
            "best_val_loss": round(best_val_loss, 6),
            "best_val_macro_f1": round(best_val_macro, 6),
            "best_val_min_class_f1": round(best_val_min_c, 6),
            "best_val_f1_std": round(best_val_std, 6),
            "best_val_ema_balanced_f1": round(best_ema_bal, 6),
            **test_metrics,
        }

        self._append_csv_rows(self.epoch_csv, self.EPOCH_CSV_FIELDS, epoch_rows)
        self._append_csv_rows(self.output_csv, self.CSV_FIELDS, [summary_row])
        self._register_completed_row(summary_row)

        del model, optimizer, X_train_t, y_train_t, X_val_t, y_val_t
        del X_train, X_val, X_test
        tf.keras.backend.clear_session()
        gc.collect()
        return summary_row

    def _evaluate_test_set(
        self, model: CNNClassifier, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        n_samples = len(y_test)
        n_classes = len(self._class_order)
        _ = model(tf.convert_to_tensor(X_test[: min(4, n_samples)]), training=False)

        inf_t0 = time.perf_counter()
        probs_list: List[np.ndarray] = []
        for start in range(0, n_samples, self.batch_size):
            batch_x = tf.convert_to_tensor(X_test[start : start + self.batch_size])
            logits = model(batch_x, training=False)
            probs_list.append(tf.nn.softmax(logits, axis=-1).numpy())
        inf_ms_per_img = ((time.perf_counter() - inf_t0) * 1000.0) / max(1, n_samples)

        probs = np.concatenate(probs_list, axis=0)
        y_pred = np.asarray(DecisionRule.predict(probs), dtype=np.int32)

        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (y_test, y_pred), 1)

        acc = float(np.trace(cm)) / max(1, n_samples)
        kappa = float(cohen_kappa_score(y_test, y_pred))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        try:
            roc_auc = round(
                float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro")), 6
            )
        except Exception:
            roc_auc = ""

        alias_map = {
            "glioma": "glioma",
            "meningioma": "meningioma",
            "notumor": "no_tumor",
            "no_tumor": "no_tumor",
            "pituitary": "pituitary",
        }
        per_class_out: dict = {}
        precisions, sensitivities, specificities, f1s = [], [], [], []
        for idx, tc in enumerate(self._class_order):
            norm_key = tc.name.lower().replace(" ", "_")
            cname = alias_map.get(norm_key, norm_key)
            tp = float(cm[idx, idx])
            fp = float(cm[:, idx].sum() - tp)
            fn = float(cm[idx, :].sum() - tp)
            tn = float(n_samples - tp - fp - fn)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_v = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

            precisions.append(prec)
            sensitivities.append(sens)
            specificities.append(spec_v)
            f1s.append(f1)

            per_class_out[f"{cname}_precision"] = round(prec, 6)
            per_class_out[f"{cname}_sensitivity"] = round(sens, 6)
            per_class_out[f"{cname}_specificity"] = round(spec_v, 6)
            per_class_out[f"{cname}_f1"] = round(f1, 6)

        f1_arr = np.asarray(f1s, dtype=np.float64)
        return {
            "test_total": n_samples,
            "inference_time_ms_per_img": round(inf_ms_per_img, 4),
            "test_accuracy": round(acc, 6),
            "test_empirical_error": round(1.0 - acc, 6),
            "test_macro_precision": round(float(np.mean(precisions)), 6),
            "test_macro_sensitivity": round(float(np.mean(sensitivities)), 6),
            "test_macro_specificity": round(float(np.mean(specificities)), 6),
            "test_macro_f1": round(float(np.mean(f1_arr)), 6),
            "test_min_class_f1": round(float(np.min(f1_arr)), 6),
            "test_f1_std": round(float(np.std(f1_arr)), 6),
            "test_cohen_kappa": round(kappa, 6),
            "test_mcc": round(mcc, 6),
            "test_roc_auc_macro": roc_auc,
            **per_class_out,
            **{f"cm_{r}{c}": int(cm[r, c]) for r in range(n_classes) for c in range(n_classes)},
        }

    def _format_val_comparison(self, name: str, ema: float) -> str:
        if name == self.BASELINE_NAME or self._baseline_ema is None:
            self._baseline_ema = ema
            if self._current_max_ema is None or ema >= self._current_max_ema:
                self._current_max_ema = ema
                self._current_max_name = name
            return "[BASELINE] Δ Base=+0.0000 | Δ Max=+0.0000"

        delta_base = ema - self._baseline_ema
        prev_max = self._current_max_ema if self._current_max_ema is not None else ema
        prev_name = self._current_max_name or self.BASELINE_NAME
        delta_max = ema - prev_max

        if ema > prev_max + 1e-6:
            self._current_max_ema = ema
            self._current_max_name = name
            return f"Δ Base={delta_base:+.4f} | Δ Max={delta_max:+.4f} [NEW MAX vs {prev_name}]"
        return (
            f"Δ Base={delta_base:+.4f} | Δ Max={delta_max:+.4f} "
            f"(max: {prev_name}={prev_max:.4f})"
        )

    def _format_test_comparison(self, name: str, seed: int, test_f1: float) -> str:
        if name == self.BASELINE_NAME or seed not in self._verify_base_f1_by_seed:
            self._verify_base_f1_by_seed[seed] = test_f1
            self._verify_max_f1_by_seed[seed] = (test_f1, name)
            return f"[BASELINE s={seed}] Δ Base=+0.0000 | Δ Max=+0.0000"

        base_f1 = self._verify_base_f1_by_seed[seed]
        prev_max, prev_name = self._verify_max_f1_by_seed[seed]
        delta_base = test_f1 - base_f1
        delta_max = test_f1 - prev_max

        if test_f1 > prev_max + 1e-6:
            self._verify_max_f1_by_seed[seed] = (test_f1, name)
            return (
                f"Δ Base(s={seed})={delta_base:+.4f} | "
                f"Δ Max(s={seed})={delta_max:+.4f} [NEW SEED MAX vs {prev_name}]"
            )
        return (
            f"Δ Base(s={seed})={delta_base:+.4f} | "
            f"Δ Max(s={seed})={delta_max:+.4f} (max: {prev_name}={prev_max:.4f})"
        )

    def _run_or_reuse_spec(
        self, spec: ExperimentSpec, *, include_test: bool = False
    ) -> dict:
        exact_key = (spec.stage, spec.name, int(spec.seed))
        if exact_key in self._completed_runs:
            row = self._completed_runs[exact_key]
            if not include_test or int(row.get("test_total", 0)) > 0:
                ema = float(row["best_val_ema_balanced_f1"])
                if include_test:
                    tf1 = float(row["test_macro_f1"])
                    cmp_str = self._format_test_comparison(spec.name, int(spec.seed), tf1)
                    self._log(
                        f"  [CACHED] {spec.stage}/{spec.name} (s={spec.seed}) -> "
                        f"test_f1={tf1:.4f} (min_c={row['test_min_class_f1']:.4f}) | {cmp_str}"
                    )
                else:
                    cmp_str = self._format_val_comparison(spec.name, ema)
                    self._log(
                        f"  [CACHED] {spec.stage}/{spec.name} (s={spec.seed}) -> "
                        f"f1_ema={ema:.4f} | {cmp_str}"
                    )
                return row

        sig_key = (spec.config_signature(), int(spec.seed))
        if sig_key in self._completed_by_sig:
            prior = self._completed_by_sig[sig_key]
            if (
                include_test
                and int(prior.get("test_total", 0)) == 0
                and sig_key in self._run_dirs_by_sig
            ):
                self._run_counter += 1
                run_id = self._run_counter
                self._log(
                    f"\n[RUN {run_id:02d}] {spec.stage} :: {spec.name} (seed={spec.seed}) [test eval on saved weights]"
                )
                split = self.build_stratified_split(spec.seed)
                _, _, _, _, X_test, y_test, _, _ = self._prepare_split_arrays(
                    split, spec, include_test=True
                )
                repo = _ExperimentRepository(self._run_dirs_by_sig[sig_key])
                best_model = repo.loadModel(in_channels=spec.num_channels())
                test_metrics = self._evaluate_test_set(best_model, X_test, y_test)

                cloned = dict(prior)
                cloned["run_id"] = run_id
                cloned["stage"] = spec.stage
                cloned["experiment_name"] = spec.name
                cloned.update(test_metrics)
                self._append_csv_rows(self.output_csv, self.CSV_FIELDS, [cloned])
                self._register_completed_row(cloned)

                tf1 = float(cloned["test_macro_f1"])
                cmp_str = self._format_test_comparison(spec.name, int(spec.seed), tf1)
                self._log(
                    f"  -> Test F1={tf1:.4f} | min_c={cloned['test_min_class_f1']:.4f} | "
                    f"MCC={cloned['test_mcc']:.4f} | {cmp_str}"
                )
                return cloned

            if not include_test or int(prior.get("test_total", 0)) > 0:
                self._run_counter += 1
                cloned = dict(prior)
                cloned["run_id"] = self._run_counter
                cloned["stage"] = spec.stage
                cloned["experiment_name"] = spec.name
                self._append_csv_rows(self.output_csv, self.CSV_FIELDS, [cloned])
                self._register_completed_row(cloned)

                ema = float(cloned["best_val_ema_balanced_f1"])
                cmp_str = self._format_val_comparison(spec.name, ema)
                self._log(
                    f"  [REUSED] {spec.stage}/{spec.name} (s={spec.seed}) -> "
                    f"f1_ema={ema:.4f} | {cmp_str}"
                )
                return cloned

        self._run_counter += 1
        run_id = self._run_counter
        self._log(
            f"\n[RUN {run_id:02d}] {spec.stage} :: {spec.name} (seed={spec.seed}) "
            f"[elapsed {self._suite_elapsed_str()}]"
        )
        row = self._train_and_evaluate_spec(spec, run_id, include_test=include_test)
        ema = float(row["best_val_ema_balanced_f1"])

        if include_test:
            tf1 = float(row["test_macro_f1"])
            cmp_str = self._format_test_comparison(spec.name, int(spec.seed), tf1)
            self._log(
                f"  -> Done in {row['train_time_s']:.1f}s (ep={row['best_epoch']}/{row['epochs_completed']}) | "
                f"val_f1={row['best_val_macro_f1']:.4f} | f1_ema={ema:.4f} | "
                f"test_f1={tf1:.4f} | min_c={row['test_min_class_f1']:.4f} | {cmp_str}"
            )
        else:
            cmp_str = self._format_val_comparison(spec.name, ema)
            self._log(
                f"  -> Done in {row['train_time_s']:.1f}s (ep={row['best_epoch']}/{row['epochs_completed']}) | "
                f"val_f1={row['best_val_macro_f1']:.4f} | min_c={row['best_val_min_class_f1']:.4f} | "
                f"f1_ema={ema:.4f} | {cmp_str}"
            )
        return row

    def _print_stage_table(
        self, title: str, entries: List[tuple[str, str, dict]]
    ) -> None:
        base_ema = self._baseline_ema if self._baseline_ema is not None else 0.0
        ranked = sorted(
            entries, key=lambda x: float(x[2]["best_val_ema_balanced_f1"]), reverse=True
        )
        table_max_ema = float(ranked[0][2]["best_val_ema_balanced_f1"]) if ranked else 0.0
        global_max_ema = max(self._current_max_ema or 0.0, table_max_ema)

        self._log(f"\n--- {title} ---")
        self._log(
            f"{'Rank':<5} {'Slot/Stage':<12} {'Config':<26} {'Val F1':>8} "
            f"{'Min-C F1':>9} {'f1_ema':>8} {'Δ Base':>9} {'Δ Max':>9} {'BestEp':>7}"
        )
        for rank, (slot, name, row) in enumerate(ranked, start=1):
            ema = float(row["best_val_ema_balanced_f1"])
            vf1 = float(row["best_val_macro_f1"])
            mc = float(row["best_val_min_class_f1"])
            ep = int(row["best_epoch"])
            self._log(
                f"{rank:<5} {slot:<12} {name:<26} {vf1:8.4f} {mc:9.4f} "
                f"{ema:8.4f} {ema - base_ema:+9.4f} {ema - global_max_ema:+9.4f} {ep:7d}"
            )

    def run(self) -> None:
        primary_seed = self.seeds[0]
        self._log(
            f"=== NeuroLens Domain MRI Filter & Combination Testbench ({self.num_workers} CPU workers) ===\n"
            f"Single Baseline: {self.BASELINE_NAME} (aug_copies={self.aug_copies}, mri_full) | "
            f"Primary seed: {primary_seed} | Verification seeds: {self.seeds}\n"
            f"Metrics CSV: {self.output_csv} | Stats CSV: {self.stats_csv}"
        )
        self._log("\n=== Stage 1: Reference Baseline & Domain MRI Filter Screen (1_filters) ===")
        baseline_spec = ExperimentSpec(
            stage="1_filters",
            name=self.BASELINE_NAME,
            crop_to_brain=True,
            n4_bias=True,
            input_size=self.input_size,
            aug_copies=self.aug_copies,
            aug_preset="mri_full",
            denoise_method="median",
            intensity_norm="none",
            contrast_method="CLAHE_2.0_8",
            edge_filter="none",
            feature_channels="none",
            foreground_zscore=True,
            seed=primary_seed,
        )

        base_row = self._run_or_reuse_spec(baseline_spec, include_test=False)
        base_ema = float(base_row["best_val_ema_balanced_f1"])

        slot_candidates: Dict[str, List[tuple[str, dict]]] = {
            "X_denoise": [
                ("X1_susan", {"denoise_method": "SUSAN"}),
                ("X2_gif", {"denoise_method": "GIF"}),
                ("X3_wavelet", {"denoise_method": "wavelet"}),
            ],
            "Y_harmonize": [
                ("Y1_pct_0.5_99.9", {"intensity_norm": "percentile_clip"}),
                ("Y2_whitestripe", {"intensity_norm": "WhiteStripe"}),
                ("Y3_nyul", {"intensity_norm": "Nyul"}),
            ],
            "Z_contrast": [
                (
                    "Z1_tophat_bottomhat",
                    {"edge_filter": "TopHatBottomHat", "contrast_method": "CLAHE_1.0_8"},
                ),
            ],
            "W_channels": [
                ("W1_plus_lbp", {"feature_channels": "lbp"}),
                ("W2_plus_log", {"feature_channels": "log"}),
                ("W3_plus_lbp_log", {"feature_channels": "lbp_log"}),
            ],
        }

        stage1_entries: List[tuple[str, str, dict]] = [("Baseline", self.BASELINE_NAME, base_row)]
        leaderboard: List[tuple[float, str, ExperimentSpec]] = [
            (base_ema, self.BASELINE_NAME, baseline_spec)
        ]
        slot_best: Dict[str, tuple[str, dict, float, ExperimentSpec]] = {}

        for slot_name, cands in slot_candidates.items():
            best_in_slot: tuple[str, dict, float, ExperimentSpec] | None = None
            for cand_name, patch in cands:
                cand_spec = replace(
                    baseline_spec,
                    stage="1_filters",
                    name=cand_name,
                    seed=primary_seed,
                    **patch,
                )
                row = self._run_or_reuse_spec(cand_spec, include_test=False)
                ema = float(row["best_val_ema_balanced_f1"])
                stage1_entries.append((slot_name, cand_name, row))
                leaderboard.append((ema, cand_name, cand_spec))
                if best_in_slot is None or ema > best_in_slot[2]:
                    best_in_slot = (cand_name, patch, ema, cand_spec)
            if best_in_slot is not None:
                slot_best[slot_name] = best_in_slot

        self._print_stage_table("Stage 1 Filter Leaderboard (seed=42)", stage1_entries)

        self._log("\n=== Stage 2: Cross-Slot Synergistic Filter Combinations (2_combos) ===")
        nx, px, _, _ = slot_best["X_denoise"]
        ny, py, _, _ = slot_best["Y_harmonize"]
        nz, pz, _, _ = slot_best["Z_contrast"]
        nw, pw, _, _ = slot_best["W_channels"]

        combo_specs_raw: List[tuple[str, dict]] = [
            (f"{nx}+{ny}", {**px, **py}),
            (f"{ny}+{nz}", {**py, **pz}),
            (f"{nx}+{nw}", {**px, **pw}),
            (f"{ny}+{nw}", {**py, **pw}),
            (f"{nx}+{ny}+{nw}", {**px, **py, **pw}),
            (f"{nx}+{ny}+{nz}+{nw}", {**px, **py, **pz, **pw}),
        ]

        stage2_entries: List[tuple[str, str, dict]] = [("Baseline", self.BASELINE_NAME, base_row)]
        seen_combo_sigs = {baseline_spec.config_signature()}
        for _, _, _, sp in slot_best.values():
            seen_combo_sigs.add(sp.config_signature())

        for combo_name, merged_patch in combo_specs_raw:
            combo_spec = replace(
                baseline_spec,
                stage="2_combos",
                name=combo_name,
                seed=primary_seed,
                **merged_patch,
            )
            if combo_spec.config_signature() in seen_combo_sigs:
                continue
            seen_combo_sigs.add(combo_spec.config_signature())
            row = self._run_or_reuse_spec(combo_spec, include_test=False)
            ema = float(row["best_val_ema_balanced_f1"])
            stage2_entries.append(("Combo", combo_name, row))
            leaderboard.append((ema, combo_name, combo_spec))

        self._print_stage_table(
            "Stage 2 Combination vs Baseline Leaderboard (seed=42)", stage2_entries
        )

        unique_finalists: List[tuple[str, ExperimentSpec]] = [
            (self.BASELINE_NAME, baseline_spec)
        ]
        seen_final_sigs = {baseline_spec.config_signature()}

        for _, label, spec in sorted(leaderboard, key=lambda x: x[0], reverse=True):
            if len(unique_finalists) >= 4:
                break
            sig = spec.config_signature()
            if sig not in seen_final_sigs:
                seen_final_sigs.add(sig)
                unique_finalists.append((label, spec))

        self._log(
            f"\n=== Stage 3: Multi-Seed Held-Out Test Verification "
            f"({len(unique_finalists)} configs x {len(self.seeds)} seeds) ==="
        )
        for label, proto_spec in unique_finalists:
            for s in self.seeds:
                verify_spec = replace(
                    proto_spec,
                    stage="3_verify",
                    name=label,
                    seed=s,
                )
                self._run_or_reuse_spec(verify_spec, include_test=True)

        self.run_statistical_analysis(unique_finalists)
        self._log(
            f"\n=== Complete in {self._suite_elapsed_str()} | "
            f"Op cache hits={self._op_cache_hits}, misses={self._op_cache_misses} ==="
        )

    def run_statistical_analysis(
        self, finalists: List[tuple[str, ExperimentSpec]]
    ) -> None:
        if len(self.seeds) < 2:
            self._log("[STATS] Need >= 2 seeds for paired hypothesis tests.")
            return

        by_name_seed: dict[tuple[str, int], dict] = {
            (row["experiment_name"], int(row["seed"])): row
            for row in self._completed_runs.values()
            if row["stage"] == "3_verify"
        }

        base_name = finalists[0][0]
        base_f1 = np.array(
            [float(by_name_seed[(base_name, s)]["test_macro_f1"]) for s in self.seeds]
        )

        stat_fields = [
            "config_name",
            "n_seeds",
            "test_f1_mean",
            "test_f1_std",
            "test_min_class_f1_mean",
            "test_class_f1_spread_mean",
            "delta_f1_vs_baseline",
            "delta_f1_vs_max",
            "ci95_low",
            "ci95_high",
            "paired_t_pval",
            "wilcoxon_pval",
            "test_mcc_mean",
            "test_mcc_std",
            "mean_train_time_s",
        ]
        stat_rows: List[dict] = []

        for label, _ in finalists:
            f1_arr = np.array(
                [float(by_name_seed[(label, s)]["test_macro_f1"]) for s in self.seeds]
            )
            min_c_arr = np.array(
                [float(by_name_seed[(label, s)]["test_min_class_f1"]) for s in self.seeds]
            )
            spread_arr = np.array(
                [float(by_name_seed[(label, s)]["test_f1_std"]) for s in self.seeds]
            )
            mcc_arr = np.array(
                [float(by_name_seed[(label, s)]["test_mcc"]) for s in self.seeds]
            )
            time_arr = np.array(
                [float(by_name_seed[(label, s)]["train_time_s"]) for s in self.seeds]
            )

            diffs = f1_arr - base_f1
            mean_diff = float(np.mean(diffs))
            n = len(self.seeds)

            if label == base_name or np.allclose(diffs, 0.0):
                t_pval, w_pval, ci_lo, ci_hi = 1.0, 1.0, 0.0, 0.0
            else:
                se = float(np.std(diffs, ddof=1)) / math.sqrt(n)
                if stats is not None:
                    t_res = stats.ttest_rel(f1_arr, base_f1)
                    t_pval = float(t_res.pvalue) if math.isfinite(t_res.pvalue) else 1.0
                    t_crit = float(stats.t.ppf(0.975, df=n - 1))
                    try:
                        w_pval = float(stats.wilcoxon(f1_arr, base_f1).pvalue)
                    except Exception:
                        w_pval = 1.0
                else:
                    t_stat = mean_diff / max(se, 1e-12)
                    t_pval = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
                    t_crit = 4.303 if n == 3 else 1.96
                    w_pval = t_pval
                ci_lo = mean_diff - t_crit * se
                ci_hi = mean_diff + t_crit * se

            stat_rows.append(
                {
                    "config_name": label,
                    "n_seeds": n,
                    "test_f1_mean": round(float(np.mean(f1_arr)), 6),
                    "test_f1_std": round(float(np.std(f1_arr, ddof=1)), 6),
                    "test_min_class_f1_mean": round(float(np.mean(min_c_arr)), 6),
                    "test_class_f1_spread_mean": round(float(np.mean(spread_arr)), 6),
                    "delta_f1_vs_baseline": round(mean_diff, 6),
                    "delta_f1_vs_max": 0.0,
                    "ci95_low": round(ci_lo, 6),
                    "ci95_high": round(ci_hi, 6),
                    "paired_t_pval": round(t_pval, 6),
                    "wilcoxon_pval": round(w_pval, 6),
                    "test_mcc_mean": round(float(np.mean(mcc_arr)), 6),
                    "test_mcc_std": round(float(np.std(mcc_arr, ddof=1)), 6),
                    "mean_train_time_s": round(float(np.mean(time_arr)), 2),
                }
            )

        max_mean_f1 = max(r["test_f1_mean"] for r in stat_rows) if stat_rows else 0.0
        for r in stat_rows:
            r["delta_f1_vs_max"] = round(r["test_f1_mean"] - max_mean_f1, 6)

        self._log("\n=== Stage 3 Paired Statistical & 4-Class Balance Comparison ===")
        self._log(
            f"{'Config':<28} {'Test F1 (±SD)':>17} {'Δ Base':>9} {'Δ Max':>9} "
            f"{'p-val':>8} {'Min-C F1':>9} {'MCC':>8}"
        )
        for row in stat_rows:
            f1_str = f"{row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f}"
            self._log(
                f"{row['config_name']:<28} {f1_str:>17} "
                f"{row['delta_f1_vs_baseline']:+9.4f} {row['delta_f1_vs_max']:+9.4f} "
                f"{row['paired_t_pval']:8.4f} {row['test_min_class_f1_mean']:9.4f} "
                f"{row['test_mcc_mean']:8.4f}"
            )

        tmp_stats = self.stats_csv.with_suffix(".csv.tmp")
        with tmp_stats.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=stat_fields)
            writer.writeheader()
            writer.writerows(stat_rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_stats, self.stats_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroLens Domain MRI Filter & Combination Testbench"
    )
    parser.add_argument("--dataset", type=str, default="../data/images")
    parser.add_argument("--output", type=str, default="results/testbench_filters_metrics.csv")
    parser.add_argument("--epoch-output", type=str, default=None)
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default="results/testbench_filter_runs")
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--val-total", type=int, default=400)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--aug-copies", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tb = Testbench(
        dataset_path=args.dataset,
        output_csv=args.output,
        epoch_csv=args.epoch_output,
        stats_csv=args.stats_output,
        work_dir=args.work_dir,
        train_per_class=args.train_per_class,
        val_total=args.val_total,
        input_size=args.input_size,
        aug_copies=args.aug_copies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        patience=args.patience,
        seeds=args.seeds,
        num_workers=args.workers,
        resume=not args.fresh,
        verbose=args.verbose,
    )
    tb.run()


if __name__ == "__main__":
    main()