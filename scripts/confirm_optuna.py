"""Congela finalistas y comprueba estabilidad de inicialización, solo en desarrollo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import pandas as pd
import torch

from scripts.download_data import ROOT
from study.data import load_development
from study.validation import prepare_partition
from study.search import evaluate_fold, write_json


def check_code(directory):
    for name, digest in json.loads((directory / "code_hashes.json").read_text(encoding="utf-8")).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Código distinto del estudio: {name}")


def select(directory, only_architecture=None):
    check_code(directory)
    output = directory / "confirmation"
    output.mkdir(exist_ok=True)
    policy = json.loads((ROOT / "config/optuna_confirmation.json").read_text(encoding="utf-8"))
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    candidates = {}
    for architecture in ([only_architecture] if only_architecture else plan["architectures"]):
        if not (directory / f"{architecture}_finished.json").exists():
            raise ValueError(f"La búsqueda {architecture} sigue pendiente")
        trials = pd.read_csv(directory / f"{architecture}_trials.csv")
        complete = trials.loc[trials.state == "COMPLETE"].sort_values(["value", "number"])
        chosen = complete.head(policy["best_search_candidates_per_architecture"]).number.tolist()
        if policy["include_baseline"] and 0 not in chosen:
            chosen.append(0)
        if 0 not in complete.number.tolist():
            raise ValueError("El control sencillo no terminó")
        candidates[architecture] = [{"trial": int(number), "settings": json.loads(
            (directory / architecture / f"trial_{int(number):04d}" / "settings.json").read_text(encoding="utf-8"))}
            for number in chosen]
    frozen = {"policy": policy, "candidates": candidates, "test_evaluated": False, "calibration_evaluated": False}
    path = output / (f"{only_architecture}_selection.json" if only_architecture else "selection.json")
    if not only_architecture:
        for architecture, chosen in candidates.items():
            prior = output / f"{architecture}_selection.json"
            if prior.exists():
                previous = json.loads(prior.read_text(encoding="utf-8"))
                if previous["candidates"][architecture] != chosen or previous["policy"] != policy:
                    raise ValueError("La selección por arquitectura ya estaba congelada")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("La selección ya está congelada; no se puede reemplazar")
    write_json(path, frozen)
    print(json.dumps({a: [c["trial"] for c in v] for a, v in candidates.items()}), flush=True)


def run(directory, architecture, baseline_only=False):
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    check_code(directory)
    (directory / "confirmation").mkdir(exist_ok=True)
    if baseline_only:
        frozen = {"policy": json.loads((ROOT / "config/optuna_confirmation.json").read_text(encoding="utf-8")),
                  "candidates": {architecture: [{"trial": 0, "settings": json.loads(
                      (directory / architecture / "trial_0000/settings.json").read_text(encoding="utf-8"))}]}}
    else:
        path = directory / "confirmation/selection.json"
        if not path.exists():
            path = directory / "confirmation" / f"{architecture}_selection.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    rules = json.loads((directory / "rules.json").read_text(encoding="utf-8"))
    torch.set_num_threads(plan["cpu_threads_per_worker"])
    features, history, protocol, manifest = load_development()
    if manifest != json.loads((directory / "development_manifest.json").read_text(encoding="utf-8")):
        raise ValueError("Datos de desarrollo distintos")
    partitions = [prepare_partition(features, history, protocol, rules, year) for year in rules["origin_years"]]
    metrics, predictions = [], []
    for candidate in frozen["candidates"][architecture]:
        number = candidate["trial"]
        for seed in frozen["policy"]["seeds"]:
            settings = {**candidate["settings"], "seed": seed}
            for part in partitions:
                year = part["plan"]["year"]
                if seed == plan["seed"]:
                    output = directory / architecture / f"trial_{number:04d}" / str(year)
                    if not (output / "complete.json").exists():
                        raise ValueError("Falta el ajuste de búsqueda que se debe reutilizar")
                else:
                    output = directory / "confirmation" / architecture / f"trial_{number:04d}" / f"seed_{seed}" / str(year)
                measured, predicted = evaluate_fold(architecture, part, settings, rules, features, protocol["horizons"], output)
                metrics.append(measured.assign(trial=number, seed=seed))
                predictions.append(predicted.assign(trial=number, seed=seed))
                print(f"{architecture}: finalista {number}, semilla {seed}, año {year}", flush=True)
    prefix = architecture + ("_baseline" if baseline_only else "")
    pd.concat(metrics, ignore_index=True).to_csv(directory / "confirmation" / f"{prefix}_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "confirmation" / f"{prefix}_predictions.csv", index=False)
    write_json(directory / "confirmation" / f"{prefix}_finished.json", {
        "candidates": len(frozen["candidates"][architecture]), "seeds": frozen["policy"]["seeds"],
        "new_models_fitted": len(frozen["candidates"][architecture]) * (len(frozen["policy"]["seeds"]) - 1) * len(partitions),
        "seconds_session": time.perf_counter() - started, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "reused_seed": plan["seed"], "test_evaluated": False, "calibration_evaluated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--architecture", choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--baseline-only", action="store_true", help="Adelantar las semillas del control prefijado")
    args = parser.parse_args()
    if args.select_only:
        select(args.experiment.resolve(), args.architecture)
    elif args.architecture:
        run(args.experiment.resolve(), args.architecture, args.baseline_only)
    else:
        parser.error("Indicar --select-only o --architecture")
