"""Selección interna purgada; la evaluación exterior no interviene al entrenar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.data import samples, target_snapshot, fit_scaler, transform
from study.models import PriceNetwork


def masked_samples(features, target, start, end, window, horizons):
    """Conservar cada etiqueta válida sin exigir que existan las de otros horizontes."""
    parts = [samples(features, target, start, end, window, [h]) for h in horizons]
    positions = np.unique(np.concatenate([part[0] for part in parts]))
    labels = np.full((len(positions), len(horizons)), np.nan)
    for column, (ids, values, _) in enumerate(parts):
        labels[np.searchsorted(positions, ids), column] = values[:, 0]
    base = features.brent_conocido.iloc[positions].to_numpy(dtype=float)
    return positions, labels, base


def relative_score(prediction, labels, base):
    scores = []
    for column in range(labels.shape[1]):
        mask = np.isfinite(labels[:, column])
        if not mask.any():
            continue
        naive = np.sqrt(np.mean((labels[mask, column] - base[mask]) ** 2))
        if naive <= 0:
            raise ValueError("El RMSE ingenuo debe ser positivo")
        scores.append(np.sqrt(np.mean((prediction[mask, column] - labels[mask, column]) ** 2)) / naive)
    if not scores:
        raise ValueError("No hay etiquetas para medir la selección interna")
    return float(np.mean(scores))


def prepare_partition(features, history, protocol, rules, year):
    boundary = pd.Timestamp(year=year, month=1, day=1)
    stop = pd.Timestamp(rules["development_end_exclusive"])
    if stop != pd.Timestamp(protocol["development_end_exclusive"]) or boundary >= stop:
        raise ValueError("La validación debe permanecer dentro de desarrollo")
    past_dates = features.index[features.index < boundary]
    inner_days = max(rules["inner_min_days"], int(np.ceil(len(past_dates) * rules["inner_fraction"])))
    if inner_days >= len(past_dates):
        raise ValueError("Histórico insuficiente para separar selección interna")
    inner_start = past_dates[-inner_days]
    horizons, window = protocol["horizons"], protocol["window"]
    fitting_target = target_snapshot(history, inner_start, features.index)
    inner_target = target_snapshot(history, boundary, features.index)
    outer_target = target_snapshot(history, stop, features.index)
    fit = masked_samples(features, fitting_target, protocol["start"], inner_start, window, horizons)
    inner = masked_samples(features, inner_target, inner_start, boundary, window, horizons)
    # Los orígenes son anuales; sus destinos pueden caer en el año siguiente,
    # siempre dentro de desarrollo y con etiqueta publicada antes de su fin.
    outer_end = min(pd.Timestamp(year=year + 1, month=1, day=1), stop)
    outer = masked_samples(features, outer_target, boundary, stop, window, horizons)
    mask = features.index[outer[0]] < outer_end
    outer = tuple(values[mask] for values in outer)
    if len(fit[0]) < 50 or min(np.isfinite(inner[1]).sum(axis=0)) < rules["minimum_evaluation_samples"]:
        raise ValueError("Muestras insuficientes para ajuste o selección interna por horizonte")
    scaler = fit_scaler(features, inner_start)
    values = transform(features, scaler)
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(windows))
    changes = fit[1] - fit[2][:, None]
    target_scale = np.maximum(np.nanstd(changes, axis=0), 1).astype("float32")
    if not np.isfinite(target_scale).all():
        raise ValueError("Escala objetivo no finita")
    targets = torch.from_numpy((changes / target_scale).astype("float32"))
    plan = {"year": year, "fit_before": str(inner_start.date()), "inner_before": str(boundary.date()),
            "outer_before": str(outer_end.date()), "labels_before": str(stop.date()),
            "fit_samples": len(fit[0]), "inner_samples": len(inner[0]), "outer_samples": len(outer[0]),
            "fit_labels_by_horizon": np.isfinite(fit[1]).sum(axis=0).tolist(),
            "inner_labels_by_horizon": np.isfinite(inner[1]).sum(axis=0).tolist(),
            "outer_labels_by_horizon": np.isfinite(outer[1]).sum(axis=0).tolist()}
    return {"inputs": inputs, "targets": targets, "target_scale": target_scale, "scaler": scaler,
            "fit": fit, "inner": inner, "outer": outer, "window": window, "plan": plan}


@torch.no_grad()
def predict_prices(model, prepared, subset, batch_size=256):
    model.eval()
    positions, _, base = prepared[subset]
    indexes = positions - prepared["window"] + 1
    if len(indexes) == 0:
        return np.empty((0, len(prepared["target_scale"])))
    predicted = np.concatenate([model(prepared["inputs"][indexes[k:k + batch_size]]).numpy()
                                for k in range(0, len(indexes), batch_size)])
    return predicted * prepared["target_scale"] + base[:, None]


def train_with_inner_selection(architecture, prepared, settings, rules, epoch_callback=None):
    """No accede a prepared['outer']; cambiarlo no puede cambiar pesos o época."""
    torch.manual_seed(settings["seed"])
    torch.use_deterministic_algorithms(True)
    model = PriceNetwork(architecture, prepared["inputs"].shape[-1], settings, len(prepared["target_scale"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings.get("weight_decay", 0.))
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4)
                 if settings.get("scheduler", False) else None)
    indexes = torch.as_tensor(prepared["fit"][0] - prepared["window"] + 1)
    best_score, best_epoch, best_state, curve = float("inf"), 0, None, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for offset in range(0, len(indexes), settings["batch_size"]):
            batch = indexes[offset:offset + settings["batch_size"]]
            labels = prepared["targets"][offset:offset + len(batch)]
            valid = torch.isfinite(labels)
            optimizer.zero_grad()
            output = model(prepared["inputs"][batch])
            if settings.get("loss", "mse") == "huber":
                loss = nn.functional.huber_loss(output[valid], labels[valid], delta=1.)
            else:
                loss = nn.functional.mse_loss(output[valid], labels[valid])
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
        inner_pred = predict_prices(model, prepared, "inner")
        score = relative_score(inner_pred, prepared["inner"][1], prepared["inner"][2])
        fit_pred = predict_prices(model, prepared, "fit")
        # Ambas curvas se miden después de actualizar pesos y sin dropout.
        fit_error = (fit_pred - prepared["fit"][1]) / prepared["target_scale"]
        inner_error = (inner_pred - prepared["inner"][1]) / prepared["target_scale"]
        curve.append({"epoch": epoch, "fit_mse_scaled": float(np.nanmean(fit_error ** 2)),
                      "inner_mse_scaled": float(np.nanmean(inner_error ** 2)), "inner_relative_rmse": score})
        if scheduler is not None:
            scheduler.step(score)
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch_callback is not None:
            epoch_callback(epoch, score, curve)
        if not settings.get("disable_early_stopping", False) and epoch - best_epoch >= rules["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, curve
