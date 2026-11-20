"""Train each candidate once, using a fixed pre-2018 / 2018-2022 split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.extended import load_extended
from study.fixed import prepare_fixed
from study.metrics import price_metrics
from study.models import PriceNetwork
from study.search import curve_diagnostics, write_json
from study.validation import predict_prices, train_with_inner_selection


CONFIG = ROOT / "config/fixed_split_2017_2022.json"
OUTPUT = ROOT / "experiments/fixed_split_2017_2022"


def measure(architecture, variant, seed, model, part, features, rules, future_reference):
    rows, forecasts = [], []
    for subset, label in [("fit", "train"), ("inner", "validation")]:
        predicted = predict_prices(model, part, subset)
        positions, labels, baseline = part[subset]
        for col, horizon in enumerate(rules["horizons"]):
            mask = np.isfinite(labels[:, col])
            ids, actual, prediction, naive = positions[mask], labels[mask, col], predicted[mask, col], baseline[mask]
            # Baselines come from the price panel even when model inputs are signals.
            future = future_reference.reindex(features.index[ids]).to_numpy()
            scores, spot_scores = price_metrics(actual, prediction), price_metrics(actual, naive)
            future_scores = price_metrics(actual, future)
            common = {"architecture": architecture, "variant": variant, "seed": seed,
                      "subset": label, "horizon": horizon}
            rows.append({**common, **scores, "naive_rmse": spot_scores["rmse"],
                         "naive_r2": spot_scores["r2"], "future_rmse": future_scores["rmse"],
                         "relative_rmse": scores["rmse"] / spot_scores["rmse"],
                         "relative_rmse_vs_future": scores["rmse"] / future_scores["rmse"]})
            forecasts.append(pd.DataFrame({**common,
                "origin": features.index[ids].strftime("%Y-%m-%d"),
                "target_date": features.index[ids + horizon].strftime("%Y-%m-%d"),
                "actual": actual, "prediction": prediction, "naive": naive, "future_naive": future}))
    return pd.DataFrame(rows), pd.concat(forecasts, ignore_index=True)


def verify_saved(directory, architecture, variant, seed, part, features, rules, future_reference):
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    model = PriceNetwork(architecture, features.shape[1], settings, len(rules["horizons"]))
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"))
    expected_metrics, expected_predictions = measure(architecture, variant, seed, model, part, features, rules, future_reference)
    saved = pd.read_csv(directory / "predictions.csv")
    saved_metrics = pd.read_csv(directory / "metrics.csv")
    pd.testing.assert_frame_equal(saved, expected_predictions, check_exact=False, rtol=1e-10, atol=1e-10)
    pd.testing.assert_frame_equal(saved_metrics, expected_metrics, check_exact=False, rtol=1e-10, atol=1e-10)
    difference = float(np.max(np.abs(saved.prediction - expected_predictions.prediction)))
    curve = pd.read_csv(directory / "curve.csv")
    selected = int(curve.loc[curve.inner_relative_rmse.idxmin(), "epoch"])
    diagnostic = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostic["selected_epoch"] == selected
    assert json.loads((directory / "scaler.json").read_text(encoding="utf-8")) == {
        **part["scaler"], "target_scale": part["target_scale"].tolist()}
    for subset, cutoff in [("train", rules["training_end_exclusive"]),
                           ("validation", rules["development_end_exclusive"])]:
        group = saved.loc[saved.subset == subset]
        assert (group.target_date < cutoff).all() and (group.origin < cutoff).all()
        if subset == "validation":
            assert (group.origin >= rules["validation_start"]).all()
            assert (group.target_date >= rules["validation_start"]).all()
    return saved_metrics, saved, difference


def run(plan_only=False):
    rules = json.loads(CONFIG.read_text(encoding="utf-8"))
    torch.set_num_threads(rules["cpu_threads"])
    source = ROOT / rules["settings_source"]
    candidates = json.loads(source.read_text(encoding="utf-8"))["candidates"]
    settings = {a: json.loads((source.parents[1] / a / f"trial_{trial:04d}/settings.json").read_text(encoding="utf-8"))
                for a, trial in candidates.items()}
    files = ["study/fixed.py", "scripts/run_fixed_split.py", "study/validation.py", "study/data.py",
             "study/models.py", "study/extended.py", "study/search.py", "study/metrics.py"]
    data = {v: load_extended(v) for v in rules["variants"]}
    parts = {v: prepare_fixed(values[0], values[1], rules) for v, values in data.items()}
    future_reference = data["levels"][0]["brent_fut_Close_conocido"]
    frozen = {"rules": rules, "candidate_settings": settings, "candidate_trials": candidates,
              "data_manifest": data[rules["variants"][0]][4],
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
              "partitions": {v: p["plan"] for v, p in parts.items()},
              "selection_data_is_reported_validation": True, "test_evaluated": False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("The saved plan differs; use a new experiment directory")
    write_json(plan_path, frozen)
    print(json.dumps(frozen["partitions"], indent=2), flush=True)
    if plan_only:
        return
    measured, predictions, differences = [], [], []
    started = time.perf_counter()
    for architecture in rules["architectures"]:
        for variant, part in parts.items():
            features = data[variant][0]
            for seed in rules["seeds"]:
                directory = OUTPUT / architecture / variant / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                current = {**settings[architecture], "epochs": rules["epochs"], "seed": seed}
                if not (directory / "complete.json").exists():
                    write_json(directory / "settings.json", current)
                    before = time.perf_counter()
                    model, epoch, curve = train_with_inner_selection(architecture, part, current, rules)
                    elapsed = time.perf_counter() - before
                    metrics, forecast = measure(architecture, variant, seed, model, part, features, rules, future_reference)
                    torch.save(model.state_dict(), directory / "model.pt")
                    metrics.to_csv(directory / "metrics.csv", index=False)
                    forecast.to_csv(directory / "predictions.csv", index=False)
                    pd.DataFrame(curve).to_csv(directory / "curve.csv", index=False)
                    write_json(directory / "scaler.json", {**part["scaler"], "target_scale": part["target_scale"].tolist()})
                    write_json(directory / "diagnostics.json", {**curve_diagnostics(curve, epoch),
                        "fit_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
                        "selection_source": "validation_2018_2022", "test_evaluated": False})
                    write_json(directory / "complete.json", {"selected_epoch": epoch,
                        "fit_seconds": elapsed, "selection_source": "validation_2018_2022", "test_evaluated": False})
                elif json.loads((directory / "settings.json").read_text(encoding="utf-8")) != current:
                    raise ValueError("Saved settings differ")
                metrics, forecast, difference = verify_saved(directory, architecture, variant, seed, part, features, rules, future_reference)
                measured.append(metrics)
                predictions.append(forecast)
                differences.append(difference)
                done = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
                print(f"{architecture} {variant} seed={seed}: epoch={done['selected_epoch']}, "
                      f"{done['fit_seconds']:.1f}s; checkpoint verified", flush=True)
    pd.concat(measured, ignore_index=True).to_csv(OUTPUT / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / "predictions.csv", index=False)
    diagnostics = [json.loads(p.read_text(encoding="utf-8")) for p in OUTPUT.glob("*/*/seed_*/diagnostics.json")]
    write_json(OUTPUT / "completion.json", {"models_fitted": len(measured),
        "checkpoints_verified": len(differences), "maximum_prediction_difference": max(differences),
        "test_evaluated": False, "annual_weight_refit": False,
        "sum_fit_seconds": sum(d["fit_seconds"] for d in diagnostics),
        "overfit_signals": sum(d["overfit_signal"] for d in diagnostics),
        "invocation_seconds": time.perf_counter() - started,
        "validation_used_for_epoch_selection": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args().plan_only)
