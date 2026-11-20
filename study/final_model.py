"""Final training and inference with an explicit, immutable development boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from study.data import target_snapshot, transform
from study.regularized import _prepared
from study.validation import masked_samples


def training_parts(features, history, settings, horizons, boundary="2023-01-01"):
    stop = pd.Timestamp(boundary)
    if (not features.index.is_unique or not features.index.is_monotonic_increasing
            or (features.index >= stop).any() or features.isna().any(axis=None)):
        raise ValueError("Final training inputs must be ordered and strictly pre-test")
    for column in ["period_start_date", "realtime_start_date"]:
        if (pd.to_datetime(history[column]) >= stop).any():
            raise ValueError("Final training target history reaches test")
    inner = stop - pd.DateOffset(years=2)
    start, window = features.index[0], settings["window"]
    fit = masked_samples(features, target_snapshot(history, inner, features.index),
                         start, inner, window, horizons)
    validation = masked_samples(features, target_snapshot(history, stop, features.index),
                                inner, stop, window, horizons)
    full = masked_samples(features, target_snapshot(history, stop, features.index),
                          start, stop, window, horizons)
    for part in [fit, validation, full]:
        if np.isfinite(part[1]).sum(axis=0).min() < 20:
            raise ValueError("Insufficient training/internal labels")
    return (_prepared(features, {"fit": fit, "inner": validation}, inner, settings, horizons),
            _prepared(features, {"fit": full}, stop, settings, horizons))


def blend(neural, naive, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Neural weight must be in [0,1]")
    return np.asarray(naive) + weight * (np.asarray(neural) - np.asarray(naive))


def choose_weight(annual_scores, tolerance):
    means = annual_scores.groupby("weight").apply(
        lambda group: group.groupby("horizon").relative_rmse.mean().mean(),
        include_groups=False)
    return float(means.index[means <= means.min() + tolerance].min())


def origin_volatility(features):
    return np.log(features.brent_conocido).diff().rolling(21).std()


@torch.no_grad()
def infer_network(model, features, scaler, scale, window, positions, target_mode="change"):
    """No target prices accepted; a window must end at each forecast origin."""
    values = transform(features, scaler)
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0 or (positions < window - 1).any() or (positions >= len(values)).any():
        raise ValueError("Invalid inference origins")
    views = np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(views))
    model.eval()
    indices = positions - window + 1
    raw = np.concatenate([model(inputs[indices[i:i + 256]]).numpy()
                          for i in range(0, len(indices), 256)]) * np.asarray(scale, dtype="float32")
    base = features.brent_conocido.iloc[positions].to_numpy()[:, None]
    result = base * np.exp(raw) if target_mode == "log_return" else base + raw
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite final inference")
    return result
