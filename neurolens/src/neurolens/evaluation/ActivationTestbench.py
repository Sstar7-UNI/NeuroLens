from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

try:
    from scipy import stats
except ImportError:
    stats = None

from neurolens.data_management.TumorClass import TumorClass
from neurolens.data_storage.ModelConfig import ModelConfig
from neurolens.evaluation.Testbench import (
    ExperimentSpec,
    Testbench,
    _format_duration,
)
from neurolens.network.CNNClassifier import ActivationSpecType, CNNClassifier
@dataclass
class ActivationSpec:
    stage: str
    name: str
    family: str
    conv_activation: str
    head_activation: str
    preprocess_sig: str
    seed: int = 42

    @property
    def activation_arg(self) -> ActivationSpecType:
        if self.conv_activation == self.head_activation:
            return self.conv_activation
        return (self.conv_activation, self.head_activation)

    @property
    def activation_label(self) -> str:
        if self.conv_activation == self.head_activation:
            return self.conv_activation
        return f"{self.conv_activation}+{self.head_activation}"

    def config_signature(self) -> str:
        return (
            f"conv_act={self.conv_activation}|head_act={self.head_activation}|"
            f"{self.preprocess_sig}"
        )


class _ActivationRepository:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint = str(self.run_dir / "model.weights.h5")
        self._confPath = str(self.run_dir / "model.json")
        self._config: ModelConfig | None = None
        self._in_channels: int = 1
        self._conv_activation: str = "relu"
        self._head_activation: str = "relu"

    def getCheckpointPath(self) -> str:
        return self._checkpoint

    def saveConfig(
        self,
        config: ModelConfig,
        in_channels: int | None = None,
        conv_activation: str | None = None,
        head_activation: str | None = None,
    ) -> None:
        self._config = config
        if in_channels is not None:
            self._in_channels = int(in_channels)
        if conv_activation is not None:
            self._conv_activation = str(conv_activation)
        if head_activation is not None:
            self._head_activation = str(head_activation)
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
            "convActivation": self._conv_activation,
            "headActivation": self._head_activation,
        }
        Path(self._confPath).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def loadConfig(self) -> ModelConfig:
        if self._config is not None:
            return self._config
        raw = json.loads(Path(self._confPath).read_text(encoding="utf-8"))
        self._in_channels = int(raw.pop("inChannels", 1))
        self._conv_activation = str(raw.pop("convActivation", "relu"))
        self._head_activation = str(raw.pop("headActivation", "relu"))
        self._config = ModelConfig(**raw)
        return self._config

    def loadModel(
        self,
        activation: ActivationSpecType,
        dropout_rate: float = 0.2,
        config: ModelConfig | None = None,
        in_channels: int | None = None,
    ) -> CNNClassifier:
        cfg = config or self.loadConfig()
        ch = int(in_channels) if in_channels is not None else int(self._in_channels)
        model = CNNClassifier(cfg, dropoutRate=dropout_rate, activation=activation)
        dummy = tf.zeros((1, cfg.inputHeight, cfg.inputWidth, ch), dtype=tf.float32)
        model(dummy, training=False)
        model.load_weights(self._checkpoint)
        return model
class ActivationTestbench:

    ALL_ACTIVATIONS: List[tuple[str, str]] = [
        ("Baseline", "relu"),
        ("Rectified", "leaky_relu"),
        ("Rectified", "prelu"),
        ("Rectified", "relu6"),
        ("Exponential", "elu"),
        ("Exponential", "selu"),
        ("Exponential", "celu"),
        ("SmoothGated", "gelu"),
        ("SmoothGated", "swish"),
        ("SmoothGated", "mish"),
        ("SmoothGated", "hard_swish"),
        ("Parametric", "trainable_swish"),
        ("CustomSine", "csinelu"),
    ]

    CSV_FIELDS = [
        "run_id",
        "stage",
        "experiment_name",
        "activation_family",
        "conv_activation",
        "head_activation",
        "activation_name",
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
        "activation_family",
        "conv_activation",
        "head_activation",
        "activation_name",
        "seed",
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
        "activation_family",
        "conv_activation",
        "head_activation",
        "activation_name",
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

    BASELINE_NAME = "relu"

    def __init__(
        self,
        dataset_path: str,
        output_csv: str = "results/testbench_activations_metrics.csv",
        epoch_csv: str | None = None,
        stats_csv: str | None = None,
        work_dir: str = "results/testbench_activation_runs",
        *,
        train_per_class: int = 150,
        val_total: int = 400,
        all_data: bool = False,
        input_size: int = 128,
        aug_copies: int = 8,
        aug_preset: str = "mri_full",
        crop_to_brain: bool = True,
        n4_bias: bool = True,
        denoise_method: str = "median",
        intensity_norm: str = "none",
        contrast_method: str = "CLAHE_2.0_8",
        edge_filter: str = "none",
        feature_channels: str = "none",
        foreground_zscore: bool = True,
        filter_counts: Sequence[int] = (16, 32, 64),
        epochs: int = 35,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
        patience: int = 6,
        ema_alpha: float = 0.5,
        combo_top_k: int = 4,
        full_grid: bool = False,
        seeds: Sequence[int] = (42, 100, 2026),
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
            self.stats_csv = self.output_csv.with_name("activation_statistical_comparison.csv")

        self.work_dir = Path(work_dir)
        self.all_data = bool(all_data)
        self.train_per_class = int(train_per_class)
        self.val_total = int(val_total)
        self.input_size = int(input_size)
        self.aug_copies = int(aug_copies)
        self.aug_preset = str(aug_preset)
        self.filter_counts = list(filter_counts)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.patience = int(patience)
        self.ema_alpha = float(ema_alpha)
        self.combo_top_k = max(2, int(combo_top_k))
        self.full_grid = bool(full_grid)
        self.seeds = [int(s) for s in seeds] or [42]
        self.num_workers = max(1, int(num_workers or (os.cpu_count() or 4)))
        self.resume = bool(resume)
        self.verbose = int(verbose)

        self._base_preprocess = ExperimentSpec(
            stage="prep",
            name="fixed_mri_pipeline",
            crop_to_brain=bool(crop_to_brain),
            n4_bias=bool(n4_bias),
            input_size=self.input_size,
            aug_copies=self.aug_copies,
            aug_preset=self.aug_preset,
            denoise_method=str(denoise_method),
            intensity_norm=str(intensity_norm),
            contrast_method=str(contrast_method),
            edge_filter=str(edge_filter),
            feature_channels=str(feature_channels),
            foreground_zscore=bool(foreground_zscore),
            seed=self.seeds[0],
        )

        self._class_order = list(TumorClass)
        self._prep_engine = self._build_prep_engine(self.train_per_class)

        for p in (
            self.output_csv.parent,
            self.epoch_csv.parent,
            self.stats_csv.parent,
            self.work_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

        self._prepared_arrays_cache: dict[tuple[int, bool], tuple] = {}
        self._effective_train_per_class: dict[int, int] = {}

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

    def _build_prep_engine(self, per_class_cap: int) -> Testbench:
        return Testbench(
            dataset_path=self.dataset_path,
            output_csv=str(self.work_dir / "_prep_helper.csv"),
            work_dir=str(self.work_dir),
            train_per_class=int(per_class_cap),
            val_total=self.val_total,
            input_size=self.input_size,
            aug_copies=self.aug_copies,
            seeds=self.seeds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            resume=False,
            verbose=self.verbose,
        )

    def _suite_elapsed_str(self) -> str:
        return _format_duration(time.perf_counter() - self._suite_start)

    def _log(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)
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
        cfg_sig = (
            f"conv_act={row['conv_activation']}|head_act={row['head_activation']}|"
            f"aug={int(row['aug_copies'])}:{row['aug_preset']}|{row['pipeline_signature']}"
        )
        self._completed_by_sig[(cfg_sig, int(row["seed"]))] = row
        run_id = int(row["run_id"])
        self._run_counter = max(self._run_counter, run_id)
        safe_name = row["experiment_name"].replace("+", "_plus_")
        cand_dir = (
            self.work_dir
            / f"run_{run_id:04d}_{row['stage']}_{safe_name}_s{int(row['seed'])}"
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
    def _build_split_for_seed(self, seed: int):
        n_classes = max(1, len(self._class_order))
        engine = self._prep_engine

        if not self.all_data:
            split = engine.build_stratified_split(seed)
            self._effective_train_per_class[seed] = int(
                len(split.getTrainingSet()) // n_classes
            )
            return split

        lo, hi = 1, 100_000
        best_per_class = 0
        best_split = None
        while lo <= hi:
            mid = (lo + hi) // 2
            engine.train_per_class = mid
            engine._splits_by_seed.pop(seed, None)
            try:
                candidate = engine.build_stratified_split(seed)
            except ValueError:
                hi = mid - 1
                continue
            best_per_class = mid
            best_split = candidate
            lo = mid + 1

        if best_split is None:
            raise ValueError(
                "No valid stratified split found (dataset too small for val_total)."
            )

        engine.train_per_class = best_per_class
        engine._splits_by_seed[seed] = best_split
        self._effective_train_per_class[seed] = int(
            len(best_split.getTrainingSet()) // n_classes
        )
        return best_split

    def _get_seed_arrays(self, seed: int, *, include_test: bool) -> tuple:
        key = (int(seed), bool(include_test))
        if key in self._prepared_arrays_cache:
            return self._prepared_arrays_cache[key]

        if not include_test and (int(seed), True) in self._prepared_arrays_cache:
            return self._prepared_arrays_cache[(int(seed), True)]

        split = self._build_split_for_seed(int(seed))
        prep_spec = replace(self._base_preprocess, seed=int(seed))
        arrays = self._prep_engine._prepare_split_arrays(
            split, prep_spec, include_test=include_test
        )
        self._prepared_arrays_cache[key] = arrays
        return arrays

    def _build_model(
        self, spec: ActivationSpec, mean: float, std: float
    ) -> tuple[CNNClassifier, ModelConfig]:
        class_names = [c.name for c in self._class_order]
        cfg = ModelConfig(
            inputWidth=int(self.input_size),
            inputHeight=int(self.input_size),
            classNames=class_names,
            filterCounts=list(self.filter_counts),
            normalizationMean=float(mean),
            normalizationStd=float(std),
            randomSeed=int(spec.seed),
            epochsTrained=0,
            parameterCount=0,
        )
        model = CNNClassifier(
            cfg,
            dropoutRate=self.dropout_rate,
            activation=spec.activation_arg,
        )
        dummy = tf.zeros(
            (
                1,
                self.input_size,
                self.input_size,
                self._base_preprocess.num_channels(),
            ),
            dtype=tf.float32,
        )
        model(dummy, training=False)
        cfg.parameterCount = int(model.count_params())
        return model, cfg

    def _train_and_evaluate_spec(
        self, spec: ActivationSpec, run_id: int, *, include_test: bool = False
    ) -> dict:
        tf.keras.utils.set_random_seed(spec.seed)
        np.random.seed(spec.seed)
        random.seed(spec.seed)

        n_classes = len(self._class_order)
        in_channels = self._base_preprocess.num_channels()
        X_train, y_train, X_val, y_val, X_test, y_test, mean, std = (
            self._get_seed_arrays(spec.seed, include_test=include_test)
        )
        effective_tpc = self._effective_train_per_class.get(
            spec.seed, int(len(y_train) // max(1, n_classes))
        )

        safe_name = spec.name.replace("+", "_plus_")
        run_dir = (
            self.work_dir / f"run_{run_id:04d}_{spec.stage}_{safe_name}_s{spec.seed}"
        )
        repo = _ActivationRepository(run_dir)
        model, model_cfg = self._build_model(spec, mean, std)

        X_train_t = tf.constant(X_train, dtype=tf.float32)
        y_train_t = tf.constant(y_train, dtype=tf.int32)
        X_val_t = tf.constant(X_val, dtype=tf.float32)
        y_val_t = tf.constant(y_val, dtype=tf.int32)

        copies = max(0, int(self.aug_copies))
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
                    bx = tf.where(
                        is_aug,
                        Testbench._augment_batch_gpu(bx, self.aug_preset),
                        bx,
                    )

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

        prep = self._base_preprocess
        train_t0 = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            ep_t0 = time.perf_counter()

            tr_loss_t, tr_cm_t = _run_train_epoch()
            val_loss_t, val_cm_t = _run_val_epoch()

            train_loss = float(tr_loss_t.numpy())
            val_loss = float(val_loss_t.numpy())
            tr_cm = tr_cm_t.numpy()
            val_cm = val_cm_t.numpy()

            train_macro, _, _, _ = Testbench._class_f1_stats_from_cm(tr_cm)
            val_macro, val_min_c, val_std, val_bal = Testbench._class_f1_stats_from_cm(
                val_cm
            )

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
                repo.saveConfig(
                    model_cfg,
                    in_channels=in_channels,
                    conv_activation=spec.conv_activation,
                    head_activation=spec.head_activation,
                )
            else:
                epochs_no_improve += 1

            epoch_rows.append(
                {
                    "run_id": run_id,
                    "stage": spec.stage,
                    "experiment_name": spec.name,
                    "activation_family": spec.family,
                    "conv_activation": spec.conv_activation,
                    "head_activation": spec.head_activation,
                    "activation_name": spec.activation_label,
                    "seed": spec.seed,
                    "pipeline_signature": prep.pipeline_signature(),
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
            best_model = repo.loadModel(
                activation=spec.activation_arg,
                dropout_rate=self.dropout_rate,
                config=model_cfg,
                in_channels=in_channels,
            )
            test_metrics = self._prep_engine._evaluate_test_set(
                best_model, X_test, y_test
            )
        else:
            test_metrics = Testbench._empty_test_metrics()

        summary_row = {
            "run_id": run_id,
            "stage": spec.stage,
            "experiment_name": spec.name,
            "activation_family": spec.family,
            "conv_activation": spec.conv_activation,
            "head_activation": spec.head_activation,
            "activation_name": spec.activation_label,
            "seed": spec.seed,
            "train_per_class": effective_tpc,
            "val_total": int(X_val.shape[0]),
            "crop_to_brain": int(prep.crop_to_brain),
            "n4_bias": int(prep.n4_bias),
            "input_size": int(prep.input_size),
            "aug_copies": int(prep.aug_copies),
            "aug_preset": prep.aug_preset,
            "denoise_method": prep.denoise_method,
            "edge_filter": prep.edge_filter,
            "contrast_method": prep.contrast_method,
            "intensity_norm": prep.intensity_norm,
            "feature_channels": prep.feature_channels,
            "foreground_zscore": int(prep.foreground_zscore),
            "pipeline_signature": prep.pipeline_signature(),
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
        tf.keras.backend.clear_session()
        gc.collect()
        return summary_row
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
        self, spec: ActivationSpec, *, include_test: bool = False
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
                _, _, _, _, X_test, y_test, _, _ = self._get_seed_arrays(
                    spec.seed, include_test=True
                )
                repo = _ActivationRepository(self._run_dirs_by_sig[sig_key])
                best_model = repo.loadModel(
                    activation=spec.activation_arg,
                    dropout_rate=self.dropout_rate,
                    in_channels=self._base_preprocess.num_channels(),
                )
                test_metrics = self._prep_engine._evaluate_test_set(
                    best_model, X_test, y_test
                )

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
            f"\n[RUN {run_id:02d}] {spec.stage} :: {spec.name} "
            f"(conv={spec.conv_activation}, head={spec.head_activation}, seed={spec.seed}) "
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
        table_max_ema = (
            float(ranked[0][2]["best_val_ema_balanced_f1"]) if ranked else 0.0
        )
        global_max_ema = max(self._current_max_ema or 0.0, table_max_ema)

        self._log(f"\n--- {title} ---")
        self._log(
            f"{'Rank':<5} {'Family':<14} {'Conv + Head Activation':<30} {'Val F1':>8} "
            f"{'Min-C F1':>9} {'f1_ema':>8} {'Δ Base':>9} {'Δ Max':>9} {'Ep(ms)':>8} {'BestEp':>7}"
        )
        for rank, (family, name, row) in enumerate(ranked, start=1):
            ema = float(row["best_val_ema_balanced_f1"])
            vf1 = float(row["best_val_macro_f1"])
            mc = float(row["best_val_min_class_f1"])
            ep_ms = float(row["mean_epoch_ms"])
            ep = int(row["best_epoch"])
            self._log(
                f"{rank:<5} {family:<14} {name:<30} {vf1:8.4f} {mc:9.4f} "
                f"{ema:8.4f} {ema - base_ema:+9.4f} {ema - global_max_ema:+9.4f} "
                f"{ep_ms:8.1f} {ep:7d}"
            )

    def run(self) -> None:
        primary_seed = self.seeds[0]
        X_tr_p, _, X_va_p, _, _, _, _, _ = self._get_seed_arrays(
            primary_seed, include_test=False
        )
        eff_tpc = self._effective_train_per_class.get(
            primary_seed, int(X_tr_p.shape[0] // max(1, len(self._class_order)))
        )
        eff_train_epoch = int(X_tr_p.shape[0]) * (max(0, self.aug_copies) + 1)

        prep_sig = self._base_preprocess.config_signature()
        self._log(
            f"=== NeuroLens CNN Activation & All-Combinations Testbench ===\n"
            f"Baseline: {self.BASELINE_NAME} | Custom: CSineLU | CNN Filters: {self.filter_counts}\n"
            f"Data mode: {'ALL AVAILABLE DATA' if self.all_data else f'per_class={eff_tpc}'} "
            f"(base_train={int(X_tr_p.shape[0])}, val={int(X_va_p.shape[0])}, "
            f"aug_copies={self.aug_copies} -> {eff_train_epoch} samples/epoch)\n"
            f"Primary seed: {primary_seed} | Verification seeds: {self.seeds} | "
            f"Combo mode: {'FULL N×N GRID' if self.full_grid else f'ALL PAIRS OF TOP-{self.combo_top_k} + relu + csinelu'}\n"
            f"Metrics CSV: {self.output_csv} | Stats CSV: {self.stats_csv}"
        )
        self._log(
            "\n=== Stage 1: All Individual Activation Functions (1_activations) ==="
        )
        stage1_entries: List[tuple[str, str, dict]] = []
        leaderboard: List[tuple[float, str, ActivationSpec]] = []
        baseline_spec: ActivationSpec | None = None
        base_row: dict | None = None

        for family_name, act_name in self.ALL_ACTIVATIONS:
            spec = ActivationSpec(
                stage="1_activations",
                name=act_name,
                family=family_name,
                conv_activation=act_name,
                head_activation=act_name,
                preprocess_sig=prep_sig,
                seed=primary_seed,
            )
            row = self._run_or_reuse_spec(spec, include_test=False)
            ema = float(row["best_val_ema_balanced_f1"])
            stage1_entries.append((family_name, act_name, row))
            leaderboard.append((ema, act_name, spec))
            if act_name == self.BASELINE_NAME:
                baseline_spec = spec
                base_row = row

        assert baseline_spec is not None and base_row is not None
        self._print_stage_table("Stage 1 All Activations Leaderboard (seed=42)", stage1_entries)
        if self.full_grid:
            combo_pool = [act_name for _, act_name in self.ALL_ACTIVATIONS]
        else:
            combo_pool: List[str] = ["relu", "csinelu"]
            for _, act_name, _ in sorted(leaderboard, key=lambda x: x[0], reverse=True):
                if act_name not in combo_pool:
                    combo_pool.append(act_name)
                if len(combo_pool) >= self.combo_top_k + 2:
                    break

        self._log(
            f"\n=== Stage 2: All Cross-Stage Combinations (2_combos: "
            f"{len(combo_pool)}×{len(combo_pool)} pool = {combo_pool}) ==="
        )
        stage2_entries: List[tuple[str, str, dict]] = [
            ("Baseline", self.BASELINE_NAME, base_row)
        ]
        seen_sigs = {sp.config_signature() for _, _, sp in leaderboard}

        for conv_act in combo_pool:
            for head_act in combo_pool:
                if conv_act == head_act:
                    continue
                combo_name = f"{conv_act}+{head_act}"
                combo_spec = ActivationSpec(
                    stage="2_combos",
                    name=combo_name,
                    family="Combo",
                    conv_activation=conv_act,
                    head_activation=head_act,
                    preprocess_sig=prep_sig,
                    seed=primary_seed,
                )
                if combo_spec.config_signature() in seen_sigs:
                    continue
                seen_sigs.add(combo_spec.config_signature())
                row = self._run_or_reuse_spec(combo_spec, include_test=False)
                ema = float(row["best_val_ema_balanced_f1"])
                stage2_entries.append(("Combo", combo_name, row))
                leaderboard.append((ema, combo_name, combo_spec))

        self._print_stage_table(
            "Stage 2 All Combinations vs Baseline Leaderboard (seed=42)",
            stage2_entries,
        )
        unique_finalists: List[tuple[str, ActivationSpec]] = [
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
            f"\n=== Activation Testbench Complete in {self._suite_elapsed_str()} ==="
        )

    def run_statistical_analysis(
        self, finalists: List[tuple[str, ActivationSpec]]
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
            f"{'Activation Config':<30} {'Test F1 (±SD)':>17} {'Δ Base':>9} {'Δ Max':>9} "
            f"{'p-val':>8} {'Min-C F1':>9} {'MCC':>8}"
        )
        for row in stat_rows:
            f1_str = f"{row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f}"
            self._log(
                f"{row['config_name']:<30} {f1_str:>17} "
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
        description="NeuroLens CNN Activation & All-Combinations Testbench"
    )
    parser.add_argument("--dataset", type=str, default="../data/images")
    parser.add_argument(
        "--output", type=str, default="results/testbench_activations_metrics.csv"
    )
    parser.add_argument("--epoch-output", type=str, default=None)
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default="results/testbench_activation_runs")
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--val-total", type=int, default=400)
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use all available per-class training samples with aug_copies=8",
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--aug-copies", type=int, default=8)
    parser.add_argument("--denoise", type=str, default="median")
    parser.add_argument("--intensity-norm", type=str, default="none")
    parser.add_argument("--contrast", type=str, default="CLAHE_2.0_8")
    parser.add_argument("--edge-filter", type=str, default="none")
    parser.add_argument("--feature-channels", type=str, default="none")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--combo-top-k", type=int, default=4)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the complete N x N Cartesian grid of all activations in Stage 2",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tb = ActivationTestbench(
        dataset_path=args.dataset,
        output_csv=args.output,
        epoch_csv=args.epoch_output,
        stats_csv=args.stats_output,
        work_dir=args.work_dir,
        train_per_class=args.train_per_class,
        val_total=args.val_total,
        all_data=args.all_data,
        input_size=args.input_size,
        aug_copies=args.aug_copies,
        denoise_method=args.denoise,
        intensity_norm=args.intensity_norm,
        contrast_method=args.contrast,
        edge_filter=args.edge_filter,
        feature_channels=args.feature_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        patience=args.patience,
        combo_top_k=args.combo_top_k,
        full_grid=args.full_grid,
        seeds=args.seeds,
        num_workers=args.workers,
        resume=not args.fresh,
        verbose=args.verbose,
    )
    tb.run()


if __name__ == "__main__":
    main()