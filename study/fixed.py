"""A single fit before 2018, with epoch selection on the entire 2018-2022 block."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from study.data import fit_scaler, target_snapshot, transform
from study.validation import masked_samples


def prepare_fixed(features, history, rules):
    train_end = pd.Timestamp(rules["training_end_exclusive"])
    validation_start = pd.Timestamp(rules["validation_start"])
    end = pd.Timestamp(rules["development_end_exclusive"])
    if not (features.index.is_monotonic_increasing and features.index.is_unique):
        raise ValueError("The feature calendar must be ordered and unique")
    if not train_end == validation_start < end == pd.Timestamp(rules["test_start"]):
        raise ValueError("Inconsistent fixed-partition boundaries")
    if rules["test_evaluation_enabled"]:
        raise ValueError("This runner never evaluates the test")
    if (features.index >= end).any() or features.isna().any(axis=None):
        raise ValueError("Inputs include test dates or missing values")
    for column in ["period_start_date", "realtime_start_date"]:
        if (pd.to_datetime(history[column]) >= end).any():
            raise ValueError("Target history includes test dates")
    horizons, window = rules["horizons"], rules["window"]
    fitting_target = target_snapshot(history, train_end, features.index)
    validation_target = target_snapshot(history, end, features.index)
    start = features.index[0]
    fit = masked_samples(features, fitting_target, start, train_end, window, horizons)
    validation = masked_samples(features, validation_target, validation_start, end, window, horizons)
    for name, part in [("fit", fit), ("validation", validation)]:
        if min(np.isfinite(part[1]).sum(axis=0)) < rules["minimum_evaluation_samples"]:
            raise ValueError(f"Too few labels in {name}")
    scaler = fit_scaler(features, train_end)
    values = transform(features, scaler)
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(windows))
    changes = fit[1] - fit[2][:, None]
    target_scale = np.maximum(np.nanstd(changes, axis=0), 1).astype("float32")
    if not np.isfinite(target_scale).all():
        raise ValueError("Invalid training target scale")
    targets = torch.from_numpy((changes / target_scale).astype("float32"))
    plan = {"training_end_exclusive": str(train_end.date()),
            "validation_start": str(validation_start.date()),
            "validation_end_exclusive": str(end.date()),
            "first_available_input": str(start.date()),
            "first_training_origin": str(features.index[fit[0][0]].date()),
            "fit_samples": len(fit[0]), "validation_samples": len(validation[0]),
            "fit_labels_by_horizon": np.isfinite(fit[1]).sum(axis=0).tolist(),
            "validation_labels_by_horizon": np.isfinite(validation[1]).sum(axis=0).tolist(),
            "test_evaluated": False, "annual_weight_refit": False}
    # The existing trainer names its epoch-selection subset 'inner'. It only
    # receives fit and validation here: there is no outer/test subset to access.
    return {"inputs": inputs, "targets": targets, "target_scale": target_scale,
            "scaler": scaler, "fit": fit, "inner": validation, "window": window, "plan": plan}
