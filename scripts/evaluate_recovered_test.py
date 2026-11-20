"""Test the six exact seed42 checkpoints in the user's fixed-split screenshot."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.data import target_snapshot
from study.extended import load_extended, read_panel_before, causal_signals
from study.final_model import infer_network, origin_volatility
from study.metrics import price_metrics
from study.models import PriceNetwork
from study.search import write_json
from study.validation import masked_samples

SOURCE = ROOT / "experiments/fixed_split_2017_2022"
CONFIG = ROOT / "config/recovered_fixed_test_20261120.json"
OUTPUT = ROOT / "experiments/recovered_fixed_test_20261120"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def model_items(config):
    items = []
    for architecture in config["architectures"]:
        for variant in config["variants"]:
            folder = SOURCE / architecture / variant / f"seed_{config['seed']}"
            scaler = read_json(folder / "scaler.json")
            if pd.Timestamp(scaler["last_fitted_date"]) >= pd.Timestamp(config["training_end_exclusive"]):
                raise ValueError("Source scaler was not fixed before 2018")
            saved = pd.read_csv(folder / "predictions.csv")
            validation = saved.loc[saved.subset.eq("validation")]
            if not (validation.target_date.lt(config["test_start"]).all()
                    and validation.origin.ge(config["validation_start"]).all()):
                raise ValueError("Stored validation overlaps test")
            widths = {str(h): float(np.quantile(np.abs(g.actual - g.prediction), .8, method="higher"))
                      for h, g in validation.groupby("horizon")}
            items.append({"architecture": architecture, "variant": variant, "seed": config["seed"],
                "checkpoint": relative(folder / "model.pt"), "scaler": relative(folder / "scaler.json"),
                "settings": read_json(folder / "settings.json"),
                "selected_epoch": read_json(folder / "complete.json")["selected_epoch"],
                "interval_half_widths": widths})
    return items


def predict_checkpoint(item, features, positions):
    """Inference accepts no labels and does not fit a scaler or initialize training."""
    scaler = read_json(ROOT / item["scaler"])
    if list(features.columns) != scaler["columns"]:
        raise ValueError("Checkpoint input columns differ from saved scaler")
    model = PriceNetwork(item["architecture"], features.shape[1], item["settings"], 3)
    model.load_state_dict(torch.load(ROOT / item["checkpoint"], map_location="cpu", weights_only=True))
    return infer_network(model, features, scaler, scaler["target_scale"], 60, positions, "change")


def replay_checkpoint(item):
    features = load_extended(item["variant"])[0]
    saved = pd.read_csv((ROOT / item["checkpoint"]).parent / "predictions.csv")
    saved = saved.loc[saved.subset.eq("validation")]
    origins = sorted(saved.origin.unique())
    positions = features.index.get_indexer(pd.to_datetime(origins))
    if (positions < 0).any():
        raise ValueError("Validation origins not present in frozen development inputs")
    result = predict_checkpoint(item, features, positions)
    differences = []
    for column, horizon in enumerate([5, 63, 252]):
        part = saved.loc[saved.horizon.eq(horizon)].sort_values("origin")
        ids = np.searchsorted(origins, part.origin)
        differences.append(float(np.max(np.abs(part.prediction.to_numpy() - result[ids, column]))))
    maximum = max(differences)
    if maximum > 1e-8:
        raise ValueError(f"Frozen checkpoint does not reproduce saved validation: {item}")
    return {"architecture": item["architecture"], "variant": item["variant"], "seed": item["seed"],
            "maximum_prediction_difference": maximum, "rows_verified": len(saved)}


def freeze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "plan.json"
    if path.exists():
        plan = read_json(path)
        if plan["config_sha256"] != digest(CONFIG):
            raise ValueError("The six-model evaluation is frozen under another configuration")
        for name, expected in plan["files"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        replay = read_json(OUTPUT / "replay.json")
        if plan["replay_sha256"] != digest(OUTPUT / "replay.json"):
            raise ValueError("Source replay verification changed")
        return plan
    config = read_json(CONFIG)
    torch.set_num_threads(1)
    items = model_items(config)
    replay = [replay_checkpoint(item) for item in items]
    write_json(OUTPUT / "replay.json", replay)
    levels, *_ = load_extended("levels")
    threshold = float(origin_volatility(levels).loc[levels.index < config["training_end_exclusive"]].quantile(.9))
    source_plan = read_json(SOURCE / "plan.json")
    for name, expected in source_plan["code_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Original campaign code changed: {name}")
    validation = []
    files = [CONFIG, SOURCE / "plan.json", SOURCE / "report/01_comparacion_completa.png",
             ROOT / "data/development_2022/manifest.json"]
    for item in items:
        folder = (ROOT / item["checkpoint"]).parent
        files.extend(folder / name for name in ["model.pt", "settings.json", "scaler.json", "complete.json",
                                                "predictions.csv", "metrics.csv", "curve.csv"])
        frame = pd.read_csv(folder / "metrics.csv")
        validation.append(frame.loc[frame.subset.eq("validation")])
    pd.concat(validation, ignore_index=True).to_csv(OUTPUT / "validation_metrics.csv", index=False)
    files.extend(ROOT / name for name in ["scripts/evaluate_recovered_test.py", "study/final_model.py",
        "study/data.py", "study/extended.py", "study/models.py", "study/validation.py", "study/metrics.py"])
    plan = {**config, "frozen_utc": now(), "config_sha256": digest(CONFIG), "models": items,
        "origin_volatility_threshold": threshold, "replay_sha256": digest(OUTPUT / "replay.json"),
        "data_manifest": source_plan["data_manifest"], "models_retrained": 0,
        "files": {relative(p): digest(p) for p in files},
        "previous_test_complete_sha256": digest(ROOT / "experiments/final_model_20261119/test/complete.json")}
    write_json(path, plan)
    print("Six original checkpoints and scalers frozen; all six reproduce saved validation.", flush=True)
    return plan


def load_panels(plan):
    # Signals require their entire causal warm-up BEFORE cropping the common calendar.
    panel_path = ROOT / "data/processed/informacion_disponible.csv"
    history_path = ROOT / "data/tables/alfred_DCOILBRENTEU_realtime.csv"
    for name, expected in plan["data_manifest"]["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source data changed since the saved models were developed")
    panel = read_panel_before(panel_path, plan["test_end_exclusive"])
    first = plan["data_manifest"]["first_input_date"]
    levels_columns = load_extended("levels")[0].columns
    frames = {"levels": panel[list(levels_columns)].loc[first:], "signals": causal_signals(panel).loc[first:]}
    expected_dates = pd.bdate_range(first, pd.Timestamp(plan["test_end_exclusive"]) - pd.Timedelta(days=1), name="Date")
    for variant, frame in frames.items():
        if (not frame.index.equals(expected_dates) or frame.isna().any(axis=None)
                or not np.isfinite(frame.to_numpy()).all()):
            raise ValueError("Test inputs have missing observations or a changed weekday calendar")
        old = load_extended(variant)[0]
        pd.testing.assert_frame_equal(frame.loc[old.index], old, check_freq=False, rtol=1e-11, atol=1e-12)
    history_rows = []
    with history_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["period_start_date"] < plan["test_end_exclusive"] and row["realtime_start_date"] < plan["test_end_exclusive"]:
                history_rows.append(row)
    history = pd.DataFrame(history_rows)
    history["DCOILBRENTEU"] = pd.to_numeric(history.DCOILBRENTEU.replace({"": np.nan, ".": np.nan}), errors="raise")
    return frames, history


def metric_summary(frame):
    rows = []
    keys = ["architecture", "variant", "seed", "horizon"]
    for identity, base in frame.groupby(keys, sort=False):
        groups = [("all", base)]
        groups += [(f"target_year_{year}", part) for year, part in base.groupby(pd.to_datetime(base.target_date).dt.year)]
        groups += [(f"origin_{name}", part) for name, part in base.groupby("regime")]
        for name, part in groups:
            measured = price_metrics(part.actual, part.prediction)
            spot = price_metrics(part.actual, part.naive)
            future = price_metrics(part.actual, part.future_naive)
            row = {**dict(zip(keys, identity)), "group": name, **measured,
                "relative_rmse": measured["rmse"] / spot["rmse"],
                "relative_rmse_vs_future": measured["rmse"] / future["rmse"],
                "naive_rmse": spot["rmse"], "future_rmse": future["rmse"],
                "first_origin": part.origin.min(), "last_origin": part.origin.max(),
                "first_target": part.target_date.min(), "last_target": part.target_date.max()}
            if "lower" in part and "upper" in part:
                row.update(interval_coverage=float(((part.actual >= part.lower) & (part.actual <= part.upper)).mean()),
                           interval_mean_width=float((part.upper - part.lower).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(plan):
    path = OUTPUT / "complete.json"
    if path.exists():
        completed = read_json(path)
        if completed["plan_sha256"] != digest(OUTPUT / "plan.json"):
            raise ValueError("An evaluation already exists for another plan")
        for name, expected in completed["files"].items():
            if digest(OUTPUT / name) != expected:
                raise ValueError(f"Saved evaluation changed: {name}")
        print("Six-model evaluation already saved; verified without recomputing predictions.")
        return
    write_json(OUTPUT / "opened.json", {"opened_utc": now(), "plan_sha256": digest(OUTPUT / "plan.json")})
    frames, history = load_panels(plan)
    levels = frames["levels"]
    target = target_snapshot(history, plan["test_end_exclusive"], levels.index)
    subsets = {variant: masked_samples(frame, target, plan["test_start"], plan["test_end_exclusive"],
                                      plan["window"], plan["horizons"]) for variant, frame in frames.items()}
    for left, right in zip(subsets["levels"], subsets["signals"]):
        np.testing.assert_array_equal(left, right)
    volatility = origin_volatility(levels)
    forecasts = []
    for item in plan["models"]:
        subset = subsets[item["variant"]]
        predicted = predict_checkpoint(item, frames[item["variant"]], subset[0])
        for col, horizon in enumerate(plan["horizons"]):
            valid = np.isfinite(subset[1][:, col])
            ids = subset[0][valid]
            width = item["interval_half_widths"][str(horizon)]
            vol = volatility.iloc[ids].to_numpy()
            forecasts.append(pd.DataFrame({"architecture": item["architecture"], "variant": item["variant"],
                "seed": item["seed"], "horizon": horizon, "origin": levels.index[ids].strftime("%Y-%m-%d"),
                "target_date": levels.index[ids + horizon].strftime("%Y-%m-%d"),
                "actual": subset[1][valid, col], "prediction": predicted[valid, col], "naive": subset[2][valid],
                "future_naive": levels.brent_fut_Close_conocido.iloc[ids].to_numpy(),
                "origin_volatility": vol,
                "regime": np.where(vol <= plan["origin_volatility_threshold"], "usual", "high_volatility"),
                "lower": predicted[valid, col] - width, "upper": predicted[valid, col] + width}))
        print(f"Evaluated {item['architecture']} / {item['variant']} / seed {item['seed']}.", flush=True)
    frame = pd.concat(forecasts, ignore_index=True)
    if not (frame.origin.ge(plan["test_start"]) & frame.target_date.lt(plan["test_end_exclusive"])).all():
        raise ValueError("Test boundary violation")
    for horizon, group in frame.groupby("horizon"):
        reference = None
        for identity, part in group.groupby(["architecture", "variant"]):
            same = part[["origin", "target_date", "actual", "naive", "future_naive"]].reset_index(drop=True)
            if reference is not None:
                pd.testing.assert_frame_equal(same, reference, check_exact=True)
            reference = same
    frame.to_csv(OUTPUT / "predictions.csv", index=False)
    metrics = metric_summary(frame)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    counts = []
    origins = np.flatnonzero(levels.index >= plan["test_start"])
    for col, horizon in enumerate(plan["horizons"]):
        inside = origins[origins + horizon < len(levels)]
        n = int(np.isfinite(subsets["levels"][1][:, col]).sum())
        counts.append({"horizon": horizon, "test_origin_days": len(origins),
            "targets_inside_test_calendar": len(inside), "n": n,
            "outside_target_calendar": len(origins) - len(inside), "missing_or_unpublished_target": len(inside) - n})
    pd.DataFrame(counts).to_csv(OUTPUT / "sample_counts.csv", index=False)
    # Verify every model artefact remains byte-for-byte identical after inference.
    for name, expected in plan["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source artefact changed during inference: {name}")
    write_json(path, {"completed_utc": now(), "plan_sha256": digest(OUTPUT / "plan.json"),
        "models_evaluated": len(plan["models"]), "models_retrained": 0, "seed": plan["seed"],
        "test_evaluated": True, "original_checkpoints_unchanged": True,
        "previous_period_evaluation": plan["prior_period_evaluation"],
        "files": {name: digest(OUTPUT / name) for name in ["opened.json", "predictions.csv", "metrics.csv",
                                                           "sample_counts.csv", "validation_metrics.csv", "replay.json"]}})
    print(metrics.loc[metrics.group.eq("all"), ["architecture", "variant", "horizon", "n", "rmse", "r2", "relative_rmse"]].to_string(index=False))


def main():
    torch.set_num_threads(1)
    evaluate(freeze())


if __name__ == "__main__":
    main()
