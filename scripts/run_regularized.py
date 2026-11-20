"""Controlled ablations, seed confirmation, and retrospective walk-forward."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.extended import load_extended
from study.metrics import price_metrics
from study.models import PriceNetwork
from study.regularized import prepare, fit_network, predict, ridge_forecast
from study.search import write_json


CONFIG = ROOT / "config/regularization_20261118.json"
OUTPUT = ROOT / "experiments/regularization_20261118"
_DATA = {}


def source(variant):
    if variant not in _DATA:
        _DATA[variant] = load_extended(variant)
    return _DATA[variant]


def scores(forecasts, minimum):
    rows = []
    for horizon, group in forecasts.groupby("horizon"):
        for year, annual in [("all", group), *list(group.groupby(pd.to_datetime(group.target_date).dt.year))]:
            measured = price_metrics(annual.actual, annual.prediction)
            spot = price_metrics(annual.actual, annual.naive)
            future = price_metrics(annual.actual, annual.future_naive)
            rows.append({"horizon": horizon, "target_year": str(year), **measured,
                         "relative_rmse": measured["rmse"] / spot["rmse"],
                         "relative_rmse_vs_future": measured["rmse"] / future["rmse"],
                         "eligible": len(annual) >= minimum})
    result = pd.DataFrame(rows)
    annual = result.loc[result.target_year.ne("all") & result.eligible]
    score = float(annual.groupby("horizon").relative_rmse.mean().mean())
    return result, score


def do_job(job):
    rules = job["rules"]
    torch.set_num_threads(rules["cpu_threads_per_worker"])
    folder = OUTPUT / job["stage"] / job["fold"] / job["variant"] / job["trial"] / job["architecture"] / f"seed_{job['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    settings = {**rules["base"], **rules["trials"][job["trial"]]}
    identity = {k: v for k, v in job.items() if k != "rules"}
    frozen = {"job": identity, "settings": settings}
    if (folder / "complete.json").exists():
        if json.loads((folder / "job.json").read_text(encoding="utf-8")) != frozen:
            raise ValueError("Inconsistent cached job")
        return {**identity, **json.loads((folder / "complete.json").read_text(encoding="utf-8")), "cached": True}
    write_json(folder / "job.json", frozen)
    features, history, *_ = source(job["variant"])
    future_series = source("levels")[0].brent_fut_Close_conocido
    groups = [[h] for h in rules["horizons"]] if settings["separate_horizons"] else [rules["horizons"]]
    forecasts, diagnostics = [], []
    before = time.perf_counter()
    for horizons in groups:
        selection, refitting, evaluation = prepare(features, history, rules, settings,
            job["train_end"], job["evaluation_end"], horizons)
        branch = folder / (f"h{horizons[0]}" if len(horizons) == 1 else "joint")
        branch.mkdir(exist_ok=True)
        if job["architecture"] == "Ridge":
            forecast, alpha, coefficients = ridge_forecast(selection, refitting, evaluation, rules["ridge_alphas"])
            write_json(branch / "ridge.json", {"alpha": alpha, "coefficients": [
                {"center": c.tolist(), "coefficient": w.tolist(), "intercept": float(b)} for c, w, b in coefficients]})
            parameters = sum(len(w) + 1 for _, w, _ in coefficients)
            diagnostics.append({"horizons": horizons, "selected_epoch": 0, "epochs_run": 0,
                "overfit_signal": False, "alpha": alpha, "parameters": parameters,
                "fit_before": refitting["fit_before"], "selection_before": selection["fit_before"],
                "maximum_prediction_difference": 0.0})
        else:
            selector, epoch, curve = fit_network(job["architecture"], selection, settings, rules, job["seed"])
            rates = [r["learning_rate"] for r in curve[:epoch]]
            model, _, full_curve = fit_network(job["architecture"], refitting, settings, rules,
                                              job["seed"], fixed_epochs=epoch, learning_rates=rates)
            forecast = predict(model, refitting, evaluation)
            torch.save(selector.state_dict(), branch / "selector.pt")
            torch.save(model.state_dict(), branch / "model.pt")
            pd.DataFrame(curve).to_csv(branch / "selection_curve.csv", index=False)
            pd.DataFrame(full_curve).to_csv(branch / "refit_curve.csv", index=False)
            selected, last = curve[epoch - 1], curve[-1]
            deterioration = 100 * (last["inner_ratio"] / selected["inner_ratio"] - 1)
            parameters = sum(p.numel() for p in model.parameters())
            replay = PriceNetwork(job["architecture"], features.shape[1], settings, len(horizons))
            replay.load_state_dict(torch.load(branch / "model.pt", weights_only=True, map_location="cpu"))
            difference = float(np.max(np.abs(forecast - predict(replay, refitting, evaluation))))
            if difference > 1e-8:
                raise ValueError("Checkpoint replay does not match")
            diagnostics.append({"horizons": horizons, "selected_epoch": epoch, "epochs_run": len(curve),
                "overfit_signal": bool(last["fit_mse"] < selected["fit_mse"] and deterioration > 2),
                "inner_deterioration_pct": deterioration, "parameters": parameters,
                "inner_score_selected": selected["inner_ratio"],
                "fit_mse_selected": selected["fit_mse"], "inner_mse_selected": selected["inner_mse"],
                "fit_before": refitting["fit_before"], "selection_before": selection["fit_before"],
                "maximum_prediction_difference": difference})
        write_json(branch / "scalers.json", {"selection": selection["scaler"],
            "selection_target_scale": selection["scale"].tolist(), "refit": refitting["scaler"],
            "refit_target_scale": refitting["scale"].tolist(), "target_mode": settings["target_mode"],
            "horizons": horizons})
        for col, horizon in enumerate(horizons):
            valid = np.isfinite(evaluation[1][:, col])
            ids = evaluation[0][valid]
            forecasts.append(pd.DataFrame({"horizon": horizon,
                "origin": features.index[ids].strftime("%Y-%m-%d"),
                "target_date": features.index[ids + horizon].strftime("%Y-%m-%d"),
                "actual": evaluation[1][valid, col], "prediction": forecast[valid, col],
                "naive": evaluation[2][valid], "future_naive": future_series.reindex(features.index[ids]).to_numpy()}))
    frame = pd.concat(forecasts, ignore_index=True)
    assert (frame.target_date < job["evaluation_end"]).all()
    assert (frame.origin >= job["train_end"]).all()
    assert (frame.target_date < rules["test_start"]).all()
    metrics, score = scores(frame, rules["minimum_evaluation_samples"])
    frame.to_csv(folder / "predictions.csv", index=False)
    metrics.to_csv(folder / "metrics.csv", index=False)
    write_json(folder / "diagnostics.json", diagnostics)
    result = {"score": score, "fit_seconds": time.perf_counter() - before, "branches": len(groups),
              "physical_neural_fits": 0 if job["architecture"] == "Ridge" else 2 * len(groups),
              "parameters": sum(d["parameters"] for d in diagnostics),
              "selected_epochs": [d["selected_epoch"] for d in diagnostics],
              "overfit_signals": sum(d["overfit_signal"] for d in diagnostics),
              "maximum_prediction_difference": max(d["maximum_prediction_difference"] for d in diagnostics),
              "test_evaluated": False}
    write_json(folder / "complete.json", result)
    return {**identity, **result, "cached": False}


def execute(jobs, rules):
    results = []
    with ProcessPoolExecutor(max_workers=rules["workers"]) as pool:
        pending = {pool.submit(do_job, j): j for j in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"{result['stage']} {len(results)}/{len(jobs)}: {result['architecture']} "
                  f"{result['variant']}/{result['trial']} seed={result['seed']} "
                  f"score={result['score']:.4f}, epochs={result['selected_epochs']}", flush=True)
    return results


def job(rules, variant, trial, architecture, seed, stage="search", fold="fixed", train_end=None, evaluation_end=None):
    return {"rules": rules, "variant": variant, "trial": trial, "architecture": architecture,
            "seed": seed, "stage": stage, "fold": fold,
            "train_end": train_end or rules["training_end_exclusive"],
            "evaluation_end": evaluation_end or rules["development_end_exclusive"]}


def rank_conditions(results, rules):
    table = pd.DataFrame(results)
    grouped = table.groupby(["variant", "trial"]).agg(score=("score", "mean"),
        score_std=("score", "std"), branches=("branches", "first"), parameters=("parameters", "mean")).reset_index()
    order = list(rules["trials"])
    remaining, ordered = grouped.copy(), []
    while len(remaining):
        close = remaining.loc[remaining.score <= remaining.score.min() + .005].copy()
        close["trial_order"] = close.trial.map(order.index)
        chosen = close.sort_values(["branches", "parameters", "trial_order", "score", "variant"]).iloc[0]
        ordered.append(chosen.drop(labels="trial_order").to_dict())
        remaining = remaining.drop(index=chosen.name)
    return ordered


def run(plan_only=False):
    rules = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUTPUT.mkdir(exist_ok=True)
    code = ["study/regularized.py", "scripts/run_regularized.py", "study/models.py", "study/data.py",
            "study/validation.py", "study/extended.py", "study/metrics.py"]
    frozen = {"rules": rules, "data_manifest": source("levels")[4],
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code},
              "validation_used_for_hyperparameter_comparison": True, "test_evaluated": False}
    plan = OUTPUT / "plan.json"
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != frozen:
        raise ValueError("The frozen experiment plan has changed")
    write_json(plan, frozen)
    if plan_only:
        print(json.dumps({"search_jobs": 60, "rules": rules}, indent=2))
        return
    before = time.perf_counter()
    first = [job(rules, v, t, a, rules["search_seed"]) for v in rules["variants"]
             for t in rules["trials"] for a in rules["architectures"]]
    search = execute(first, rules)
    write_json(OUTPUT / "search_results.json", search)
    ranked = rank_conditions(search, rules)
    selected = [(r["variant"], r["trial"]) for r in ranked[:2]]
    if ("signals", "control") not in selected:
        selected.append(("signals", "control"))
    write_json(OUTPUT / "confirmation_plan.json", {"conditions": selected, "search_ranking": ranked})
    confirmation = execute([job(rules, v, t, a, s, stage="confirmation") for v, t in selected
                            for a in rules["architectures"] for s in rules["confirmation_seeds"]], rules)
    common = [r for r in search if (r["variant"], r["trial"]) in selected] + confirmation
    final_ranking = rank_conditions(common, rules)
    winner = final_ranking[0]
    write_json(OUTPUT / "selected_condition.json", {"condition": winner, "ranking": final_ranking,
        "scope": "development candidate; chosen using validation, not independent test"})
    checks = [(winner["variant"], winner["trial"])]
    if ("signals", "control") not in checks:
        checks.append(("signals", "control"))
    walk = execute([job(rules, v, t, a, rules["walk_forward_seed"], stage="walk_forward", fold=f["name"],
                      train_end=f["train_end"], evaluation_end=f["validation_end"])
                    for v, t in checks for a in rules["architectures"] for f in rules["walk_forward"]], rules)
    baselines = execute([job(rules, winner["variant"], winner["trial"], "Ridge", 42, stage="baselines",
                      fold=f["name"], train_end=f["train_end"], evaluation_end=f["validation_end"])
                    for f in [{"name": "fixed", "train_end": rules["training_end_exclusive"],
                               "validation_end": rules["development_end_exclusive"]}, *rules["walk_forward"]]], rules)
    results = search + confirmation + walk + baselines
    write_json(OUTPUT / "all_results.json", results)
    write_json(OUTPUT / "completion.json", {"jobs": len(results),
        "physical_neural_fits": sum(r["physical_neural_fits"] for r in results),
        "branches_verified": sum(r["branches"] for r in results if r["architecture"] != "Ridge"),
        "maximum_prediction_difference": max(r["maximum_prediction_difference"] for r in results),
        "sum_fit_seconds": sum(r["fit_seconds"] for r in results),
        "invocation_seconds": time.perf_counter() - before,
        "test_evaluated": False, "selected_condition": winner})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args().plan_only)
