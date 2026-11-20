"""Evaluación reutilizable del estudio: pesos seleccionados solo con el interno."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from study.metrics import price_metrics
from study.validation import train_with_inner_selection, predict_prices


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        temporary = handle.name
    os.replace(temporary, path)


def objective_score(metrics):
    valid = metrics.loc[metrics.eligible_for_summary]
    if valid.empty:
        raise ValueError("Sin grupos evaluables")
    return float(valid.groupby("horizon").relative_rmse.mean().mean())


def curve_diagnostics(curve, epoch):
    table = pd.DataFrame(curve)
    chosen = table.loc[table.epoch == epoch].iloc[0]
    last = table.iloc[-1]
    deterioration = 100 * (last.inner_relative_rmse / chosen.inner_relative_rmse - 1)
    return {"selected_epoch": int(epoch), "epochs_run": len(table),
            "fit_mse_at_selected": float(chosen.fit_mse_scaled),
            "inner_mse_at_selected": float(chosen.inner_mse_scaled),
            "fit_mse_last": float(last.fit_mse_scaled), "inner_mse_last": float(last.inner_mse_scaled),
            "inner_score_selected": float(chosen.inner_relative_rmse),
            "inner_score_last": float(last.inner_relative_rmse),
            "inner_deterioration_pct": float(deterioration),
            "overfit_signal": bool(last.fit_mse_scaled < chosen.fit_mse_scaled and deterioration > 2)}


def evaluate_fold(architecture, part, settings, rules, features, horizons, directory, callback=None):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        if json.loads((directory / "settings.json").read_text()) != settings:
            raise ValueError("El ajuste guardado pertenece a otra configuración")
        return pd.read_csv(directory / "metrics.csv"), pd.read_csv(directory / "predictions.csv")
    write_json(directory / "settings.json", settings)
    before = time.perf_counter()
    model, epoch, curve = train_with_inner_selection(architecture, part, settings, rules, callback)
    elapsed = time.perf_counter() - before
    pd.DataFrame(curve).to_csv(directory / "curve.csv", index=False)
    torch.save(model.state_dict(), directory / "model.pt")
    write_json(directory / "diagnostics.json", {**curve_diagnostics(curve, epoch), "fit_seconds": elapsed,
        "parameters": sum(p.numel() for p in model.parameters()), "selection_source": "inner_only"})
    write_json(directory / "scaler.json", {**part["scaler"], "target_scale": part["target_scale"].tolist()})
    predicted = predict_prices(model, part, "outer")
    positions, labels, base = part["outer"]
    rolling = features.brent_conocido.rolling(252, min_periods=252).mean()
    metrics, predictions = [], []
    for column, horizon in enumerate(horizons):
        mask = np.isfinite(labels[:, column])
        ids, actual, forecast, naive = positions[mask], labels[mask, column], predicted[mask, column], base[mask]
        if not len(actual):
            continue
        measured, reference = price_metrics(actual, forecast), price_metrics(actual, naive)
        moving = rolling.iloc[ids].to_numpy()
        moving_metric = price_metrics(actual, moving)
        eligible = len(ids) >= rules["minimum_evaluation_samples"]
        metrics.append({"architecture": architecture, "fold": part["plan"]["year"], "horizon": horizon,
            **measured, "naive_rmse": reference["rmse"], "naive_mae": reference["mae"],
            "relative_rmse": measured["rmse"] / reference["rmse"],
            "relative_mae": measured["mae"] / reference["mae"],
            "relative_rmse_vs_rolling": measured["rmse"] / moving_metric["rmse"],
            "eligible_for_summary": eligible, "selected_epoch": epoch, "epochs_run": len(curve),
            "fit_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters())})
        for k, origin in enumerate(ids):
            predictions.append({"architecture": architecture, "fold": part["plan"]["year"], "horizon": horizon,
                "origin": str(features.index[origin].date()), "target_date": str(features.index[origin + horizon].date()),
                "actual": actual[k], "prediction": forecast[k], "naive": naive[k],
                "rolling_mean_252": moving[k], "eligible_for_summary": eligible})
    metrics, predictions = pd.DataFrame(metrics), pd.DataFrame(predictions)
    metrics.to_csv(directory / "metrics.csv", index=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    write_json(directory / "complete.json", {"fit_seconds": elapsed, "selected_epoch": epoch,
        "selection_source": "inner_only", "test_evaluated": False, "calibration_evaluated": False})
    return metrics, predictions
