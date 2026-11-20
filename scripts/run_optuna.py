"""Búsqueda CPU reanudable por arquitectura; desarrollo únicamente."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import optuna
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.data import load_development
from study.validation import prepare_partition
from study.search import evaluate_fold, objective_score, write_json


def suggest(trial, plan):
    space = plan["space"]
    settings = {key: trial.suggest_categorical(key, space[key]) for key in
        ["hidden_size", "num_layers", "head_size", "pooling", "batch_size", "weight_decay", "loss", "scheduler"]}
    settings["dropout"] = trial.suggest_float("dropout", *space["dropout"])
    settings["learning_rate"] = trial.suggest_float("learning_rate", *space["learning_rate"], log=True)
    settings.update(epochs=plan["epochs"], seed=plan["seed"], gradient_clip_norm=1.,
                    cpu_threads=plan["cpu_threads_per_worker"])
    return settings


def anchors():
    baseline = {"hidden_size": 32, "num_layers": 1, "head_size": 0, "pooling": "last", "batch_size": 128,
                "weight_decay": 0., "loss": "mse", "scheduler": False, "dropout": 0., "learning_rate": .001}
    complex_model = {**baseline, "num_layers": 2, "hidden_size": 64, "head_size": 32,
                     "dropout": .2, "weight_decay": .0001, "learning_rate": .0003}
    regularized = {**baseline, "num_layers": 2, "hidden_size": 32, "head_size": 16, "pooling": "last_mean",
                   "dropout": .3, "weight_decay": .001, "loss": "huber", "scheduler": True}
    return [baseline, complex_model, regularized]


def run(architecture, directory):
    plan = json.loads((ROOT / "config/optuna_cpu.json").read_text())
    rules = json.loads((ROOT / "config/validation_v2.json").read_text())
    torch.set_num_threads(plan["cpu_threads_per_worker"])
    features, history, protocol, manifest = load_development()
    partitions = [prepare_partition(features, history, protocol, rules, year) for year in rules["origin_years"]]
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in [("plan.json", plan), ("rules.json", rules), ("source_protocol.json", protocol),
                             ("development_manifest.json", manifest)]:
        path = directory / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"No se puede reanudar con datos o reglas diferentes: {filename}")
        write_json(path, value)
    code = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ["scripts/run_optuna.py", "study/search.py", "study/validation.py", "study/models.py", "study/data.py"]}
    path = directory / "code_hashes.json"
    if path.exists() and json.loads(path.read_text()) != code:
        raise ValueError("El código cambió; usar otro directorio de estudio")
    write_json(path, code)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=architecture, direction="minimize",
        storage="sqlite:///" + (directory / f"{architecture}.sqlite3").as_posix(), load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=plan["seed"], n_startup_trials=plan["startup_trials"], multivariate=False),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=plan["startup_trials"], n_warmup_steps=4, interval_steps=2))
    if len(study.trials) == 0:
        for settings in anchors():
            study.enqueue_trial(settings)
    started = time.perf_counter()
    def objective(trial):
        trial_started = time.perf_counter()
        settings = suggest(trial, plan)
        root = directory / architecture / f"trial_{trial.number:04d}"
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "settings.json", settings)
        all_metrics, all_predictions = [], []
        # Los tres controles se ejecutan completos, sin poda.
        def callback(epoch, score, curve):
            trial.report(min(row["inner_relative_rmse"] for row in curve), epoch - 1)
            if trial.number >= 3 and trial.should_prune():
                pd.DataFrame(curve).to_csv(root / "pruned_inner_curve.csv", index=False)
                write_json(root / "pruned_training.json", {"epochs_run": epoch,
                    "fit_seconds": time.perf_counter() - trial_started, "fold": rules["origin_years"][0],
                    "models_partially_fitted": 1, "selection_source": "inner_only"})
                raise optuna.TrialPruned("Poda por selección interna del primer corte")
        for index, part in enumerate(partitions):
            measured, predicted = evaluate_fold(architecture, part, settings, rules, features, protocol["horizons"],
                root / str(part["plan"]["year"]), callback if index == 0 else None)
            all_metrics.append(measured)
            all_predictions.append(predicted)
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.to_csv(root / "metrics.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(root / "predictions.csv", index=False)
        score = objective_score(metrics)
        trial.set_user_attr("models_fitted", len(partitions))
        trial.set_user_attr("seconds", float(metrics.groupby("fold").fit_seconds.first().sum()))
        return score
    remaining = max(0, plan["trials_per_architecture"] - len([t for t in study.trials if t.state.is_finished()]))
    def progress(study, trial):
        study.trials_dataframe().to_csv(directory / f"{architecture}_trials.csv", index=False)
        finished = sum(t.state.is_finished() for t in study.trials)
        write_json(directory / f"{architecture}_status.json", {"architecture": architecture,
            "finished_trials": finished, "best_score": study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None,
            "seconds_session": time.perf_counter() - started, "test_evaluated": False, "calibration_evaluated": False})
        best = study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None
        print(f"{architecture}: intento {trial.number}, {trial.state.name}, valor {trial.value}, mejor {best}", flush=True)
    study.optimize(objective, n_trials=remaining, timeout=plan["timeout_seconds_per_architecture"],
                   callbacks=[progress], catch=(ValueError, FloatingPointError))
    write_json(directory / f"{architecture}_finished.json", {"trials": len(study.trials),
        "complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_trial": study.best_trial.number, "best_score": study.best_value,
        "seconds_session": time.perf_counter() - started, "test_evaluated": False, "calibration_evaluated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.architecture, args.output.resolve())
