"""Internal epoch choice followed by a fresh full-history fit and frozen evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.data import fit_scaler, target_snapshot, transform
from study.models import PriceNetwork
from study.validation import masked_samples, relative_score


def _prepared(features, subsets, cutoff, settings, horizons):
    scaler = fit_scaler(features, cutoff)
    values = transform(features, scaler)
    window = settings["window"]
    views = np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(views))
    positions, labels, base = subsets["fit"]
    if settings["target_mode"] == "log_return":
        raw_targets = np.log(labels / base[:, None])
    elif settings["target_mode"] == "change":
        raw_targets = labels - base[:, None]
    else:
        raise ValueError("Unknown target mode")
    scale = np.nanstd(raw_targets, axis=0)
    if (scale <= 0).any() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite or constant training targets")
    return {"inputs": inputs, "subsets": subsets, "scale": scale.astype("float32"),
            "targets": torch.from_numpy((raw_targets / scale).astype("float32")),
            "scaler": scaler, "window": window, "target_mode": settings["target_mode"],
            "horizons": horizons, "fit_before": str(pd.Timestamp(cutoff).date())}


def prepare(features, history, rules, settings, train_end, evaluation_end, horizons):
    train_end, evaluation_end = pd.Timestamp(train_end), pd.Timestamp(evaluation_end)
    test_start = pd.Timestamp(rules["test_start"])
    if rules["test_evaluation_enabled"] or not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("Invalid calendar or test-enabled rules")
    if not train_end < evaluation_end <= test_start or (features.index >= test_start).any():
        raise ValueError("Invalid boundaries or test inputs")
    if features.isna().any(axis=None) or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Missing or nonfinite inputs")
    for name in ["period_start_date", "realtime_start_date"]:
        if (pd.to_datetime(history[name]) >= test_start).any():
            raise ValueError("Target history reaches test")
    inner_start = train_end - pd.DateOffset(years=rules["inner_years"])
    start, window = features.index[0], settings["window"]
    fit = masked_samples(features, target_snapshot(history, inner_start, features.index),
                         start, inner_start, window, horizons)
    inner = masked_samples(features, target_snapshot(history, train_end, features.index),
                           inner_start, train_end, window, horizons)
    full = masked_samples(features, target_snapshot(history, train_end, features.index),
                          start, train_end, window, horizons)
    evaluation = masked_samples(features, target_snapshot(history, evaluation_end, features.index),
                                train_end, evaluation_end, window, horizons)
    for part in [fit, inner, full, evaluation]:
        if min(np.isfinite(part[1]).sum(axis=0)) < rules["minimum_evaluation_samples"]:
            raise ValueError("Too few labels for an internal or evaluation horizon")
    selection = _prepared(features, {"fit": fit, "inner": inner}, inner_start, settings, horizons)
    # Evaluation labels are deliberately absent from both objects given to trainers.
    refitting = _prepared(features, {"fit": full}, train_end, settings, horizons)
    return selection, refitting, evaluation


@torch.no_grad()
def predict(model, prepared, subset):
    positions, _, base = subset
    indices = positions - prepared["window"] + 1
    model.eval()
    raw = np.concatenate([model(prepared["inputs"][indices[i:i + 256]]).numpy()
                          for i in range(0, len(indices), 256)]) * prepared["scale"]
    result = base[:, None] * np.exp(raw) if prepared["target_mode"] == "log_return" else base[:, None] + raw
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite price forecast")
    return result


def normalized_mse(predicted, subset, prepared):
    _, labels, base = subset
    if prepared["target_mode"] == "log_return":
        errors = np.log(predicted / labels) / prepared["scale"]
    else:
        errors = (predicted - labels) / prepared["scale"]
    return float(np.nanmean(errors ** 2))


def fit_network(architecture, prepared, settings, rules, seed, fixed_epochs=None, learning_rates=None):
    """Fixed-epoch refit has no path to early-stopping/evaluation labels."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = PriceNetwork(architecture, prepared["inputs"].shape[-1], settings, len(prepared["horizons"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
                 if settings["scheduler"] and fixed_epochs is None else None)
    ids = torch.as_tensor(prepared["subsets"]["fit"][0] - prepared["window"] + 1)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, stopping_epoch = float("inf"), None, 0, 0
    significant_score = float("inf")
    curves = []
    for epoch in range(1, (fixed_epochs or rules["maximum_epochs"]) + 1):
        if learning_rates is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch - 1]
        model.train()
        for permutation in torch.randperm(len(ids), generator=generator).split(settings["batch_size"]):
            labels = prepared["targets"][permutation]
            output = model(prepared["inputs"][ids[permutation]])
            # Equal horizon weights despite differing numbers of valid labels.
            losses = []
            for col in range(labels.shape[1]):
                valid = torch.isfinite(labels[:, col])
                if valid.any():
                    loss = (nn.functional.huber_loss(output[valid, col], labels[valid, col], delta=1.)
                            if settings["loss"] == "huber" else
                            nn.functional.mse_loss(output[valid, col], labels[valid, col]))
                    losses.append(loss)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        fitting = predict(model, prepared, prepared["subsets"]["fit"])
        row = {"epoch": epoch, "fit_mse": normalized_mse(fitting, prepared["subsets"]["fit"], prepared),
               "learning_rate": optimizer.param_groups[0]["lr"]}
        if fixed_epochs is None:
            inner = prepared["subsets"]["inner"]
            predicted = predict(model, prepared, inner)
            score = relative_score(predicted, inner[1], inner[2])
            row.update(inner_mse=normalized_mse(predicted, inner, prepared), inner_ratio=score)
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            if score < significant_score - rules["minimum_improvement"]:
                significant_score, stopping_epoch = score, epoch
            if scheduler is not None:
                scheduler.step(score)
        curves.append(row)
        if fixed_epochs is None and epoch - stopping_epoch >= rules["patience"]:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
    return model, best_epoch, curves


def ridge_forecast(selection, refitting, evaluation, alphas):
    """Last-day Ridge; alpha chosen only on the internal block, then refitted."""
    def fitted(prepared, alpha):
        ids = prepared["subsets"]["fit"][0] - prepared["window"] + 1
        x = prepared["inputs"][ids, -1].numpy().astype(float)
        y = prepared["targets"].numpy().astype(float)
        coefficients = []
        for col in range(y.shape[1]):
            valid = np.isfinite(y[:, col])
            xv, yv = x[valid], y[valid, col]
            center, intercept = xv.mean(axis=0), yv.mean()
            xc = xv - center
            coefficient = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (yv - intercept))
            coefficients.append((center, coefficient, intercept))
        return coefficients
    def infer(prepared, coefficients, subset):
        x = prepared["inputs"][subset[0] - prepared["window"] + 1, -1].numpy().astype(float)
        raw = np.column_stack([(x - center) @ coef + intercept for center, coef, intercept in coefficients]) * prepared["scale"]
        return subset[2][:, None] * np.exp(raw) if prepared["target_mode"] == "log_return" else subset[2][:, None] + raw
    inner = selection["subsets"]["inner"]
    scores = [(relative_score(infer(selection, fitted(selection, alpha), inner), inner[1], inner[2]), alpha)
              for alpha in alphas]
    alpha = min(scores)[1]
    coefficients = fitted(refitting, alpha)
    return infer(refitting, coefficients, evaluation), alpha, coefficients
