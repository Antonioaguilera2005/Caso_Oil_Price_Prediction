"""Reproduce the selected RNN in a separate directory, using local frozen data.

The delivered model, application and historical experiments remain unchanged.
Only pre-2018 training and 2018–2022 validation are used. No 2023+ test is run,
no data is downloaded, and reproduction never replaces the delivered weights.

Examples (from the repository root):
  python -m scripts.reproduce_delivery --plan-only
  python -m scripts.reproduce_delivery
  python -m scripts.reproduce_delivery --output runs/reproduction/my_run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

# Also support ``python scripts/reproduce_delivery.py`` from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_fixed_split import measure, verify_saved
from study.extended import load_extended
from study.fixed import prepare_fixed
from study.search import curve_diagnostics
from study.validation import train_with_inner_selection


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within_workspace(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("--output must be a new directory inside this repository.")
    # Keep generated output away from source, data, published assets and experiments.
    protected_names = {".git", ".agents", ".codex", ".aws", ".venv", ".tools", "data", "config",
                       "experiments", "study", "scripts", "tests", "app_site", "model_hub", "docs"}
    relative = output.relative_to(ROOT)
    if relative.parts[0].casefold() in {name.casefold() for name in protected_names}:
        raise ValueError("--output cannot be inside a source/data/experiment directory; use runs/reproduction/<name>.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output already exists and is populated. Choose a new directory; nothing was overwritten.")
    return output


def protected_files(delivery: dict, fixed_rules: dict) -> list[Path]:
    # Protect both the original validation experiment and the historical test,
    # although the latter is only hashed and is never evaluated or used to train.
    files = set()
    model_directory = (ROOT / delivery["checkpoint"]).parent
    experiment = model_directory.parents[2]
    for directory in (experiment, ROOT / delivery["evaluation"]):
        files.update(path for path in directory.rglob("*") if path.is_file())
    files.update(path for path in (ROOT / "study").glob("*.py"))
    files.update({ROOT / "scripts/run_fixed_split.py", ROOT / "scripts/reproduce_delivery.py",
                  ROOT / "config/delivery_model.json", ROOT / "config/fixed_split_2017_2022.json",
                  ROOT / "config/validation_2021_2022.json", ROOT / fixed_rules["settings_source"]})
    development = ROOT / "data/development_2022"
    manifest = read_json(development / "manifest.json")
    files.add(development / "manifest.json")
    files.update(development / name for name in manifest["files"])
    # Hash full raw/current panels for integrity only; training reads the frozen
    # development exports ending in 2022, not these later observations.
    files.update(ROOT / name.replace("\\", "/") for name in manifest.get("source_hashes", {}))
    files.update({ROOT / delivery["settings"], ROOT / delivery["scaler"]})
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local files; this command does not download data: " + ", ".join(missing))
    return sorted(files)


def snapshot(files: list[Path]) -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def digest_snapshot(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_unchanged(before: dict[str, str], files: list[Path]) -> dict:
    after = snapshot(files)
    if before != after:
        changed = [name for name in set(before) | set(after) if before.get(name) != after.get(name)]
        raise RuntimeError("Protected source files changed during reproduction: " + ", ".join(sorted(changed)))
    return {"verified": True, "file_count": len(before), "aggregate_sha256": digest_snapshot(before)}


def compare_original(output: Path, original: Path) -> dict:
    # This comparison is performed after training. It does not select epochs or
    # modify new weights, and it uses the original train/validation files only.
    old_metrics, new_metrics = pd.read_csv(original / "metrics.csv"), pd.read_csv(output / "metrics.csv")
    columns = ["mae", "mse", "rmse", "r2", "relative_rmse", "relative_rmse_vs_future"]
    joined = new_metrics.merge(old_metrics, on=["architecture", "variant", "seed", "subset", "horizon"],
                               suffixes=("_new", "_original"), validate="one_to_one")
    deltas = [{"subset": row["subset"], "horizon": int(row["horizon"]),
               **{name + "_delta": float(row[name + "_new"] - row[name + "_original"]) for name in columns}}
              for row in joined.to_dict(orient="records")]
    keys = ["architecture", "variant", "seed", "subset", "horizon", "origin", "target_date"]
    old_predictions, new_predictions = pd.read_csv(original / "predictions.csv"), pd.read_csv(output / "predictions.csv")
    forecasts = new_predictions.merge(old_predictions, on=keys, suffixes=("_new", "_original"), validate="one_to_one")
    if not len(joined) == len(new_metrics) == len(old_metrics):
        raise ValueError("Training/validation metric partitions differ from the saved reference.")
    if not len(forecasts) == len(new_predictions) == len(old_predictions):
        raise ValueError("Training/validation forecast rows differ from the saved reference.")
    old_weights = torch.load(original / "model.pt", weights_only=True, map_location="cpu")
    new_weights = torch.load(output / "model.pt", weights_only=True, map_location="cpu")
    if old_weights.keys() != new_weights.keys():
        raise ValueError("Reproduced architecture has different weight names.")
    maximum_weight_difference = max(float(torch.max(torch.abs(new_weights[name] - old_weights[name]))) for name in old_weights)
    return {"original_selected_epoch": read_json(original / "diagnostics.json")["selected_epoch"],
            "reproduced_selected_epoch": read_json(output / "diagnostics.json")["selected_epoch"],
            "maximum_absolute_prediction_difference_usd": float(np.max(np.abs(forecasts.prediction_new - forecasts.prediction_original))),
            "maximum_absolute_weight_difference": maximum_weight_difference,
            "weights_bitwise_equal": all(torch.equal(new_weights[name], old_weights[name]) for name in old_weights),
            "metric_deltas": deltas,
            "note": "Numerical differences across PyTorch versions, processors or operating systems are reported, not hidden; matching the exact checkpoint bytes is not required."}


def run(args: argparse.Namespace) -> dict:
    delivery = read_json(ROOT / "config/delivery_model.json")
    rules = read_json(ROOT / "config/fixed_split_2017_2022.json")
    settings = read_json(ROOT / delivery["settings"])
    if (delivery["architecture"], delivery["variant"], delivery["seed"]) != ("RNN", "signals", 42):
        raise ValueError("This reproduction command targets the frozen RNN/signals/seed42 delivery.")
    if (rules["training_end_exclusive"], rules["validation_start"], rules["development_end_exclusive"]) != ("2018-01-01", "2018-01-01", "2023-01-01"):
        raise ValueError("The frozen pre-2018 / 2018–2022 partition has changed.")
    if rules["test_evaluation_enabled"] or rules["horizons"] != [5, 63, 252] or rules["window"] != 60:
        raise ValueError("The frozen no-test partition/window/horizons have changed.")
    if (settings["seed"], settings["learning_rate"], settings["batch_size"], settings["epochs"], rules["patience"]) != (42, .001, 128, 40, 10):
        raise ValueError("The frozen delivery settings have changed.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = within_workspace(args.output or f"runs/reproduction/{timestamp}")
    files = protected_files(delivery, rules)
    before = snapshot(files)
    torch.set_num_threads(settings.get("cpu_threads", rules["cpu_threads"]))
    features, history, protocol, _, data_manifest = load_extended("signals")
    levels, _, _, _, _ = load_extended("levels")
    prepared = prepare_fixed(features, history, rules)
    original = (ROOT / delivery["checkpoint"]).parent
    source_keys = ["config/delivery_model.json", "config/fixed_split_2017_2022.json",
                   "config/validation_2021_2022.json", "scripts/run_fixed_split.py",
                   "scripts/reproduce_delivery.py", "study/fixed.py", "study/validation.py",
                   "study/extended.py", "study/models.py", "data/development_2022/manifest.json",
                   "data/development_2022/signals.csv", "data/development_2022/levels.csv",
                   "data/development_2022/target_vintages.csv", delivery["checkpoint"], delivery["settings"], delivery["scaler"]]
    plan = {"operation": "plan_only" if args.plan_only else "isolated_delivery_reproduction",
            "model_id": delivery["id"], "architecture": "RNN", "variant": "signals", "seed": 42,
            "output": output.relative_to(ROOT).as_posix(), "downloads": False,
            "test_evaluated": False, "delivery_model_replaced": False,
            "settings": settings, "patience": rules["patience"], "window": rules["window"],
            "horizons": rules["horizons"], "partition": prepared["plan"],
            "feature_count": int(features.shape[1]), "feature_columns": list(features.columns),
            "input_rows": int(len(features)), "last_input_date": features.index.max().strftime("%Y-%m-%d"),
            "source_sha256": {name: before[name] for name in source_keys},
            "protected_files": {"file_count": len(before), "aggregate_sha256": digest_snapshot(before)},
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "pytorch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu"},
            "method": "Fit scaler and weights using pre-2018 observations; choose epoch using the complete 2018–2022 validation block. Validation is used for selection and is not an independent test.",
            "numerical_reproducibility": "Seed 42 and deterministic PyTorch algorithms are used; cross-platform numerical differences are reported rather than suppressed."}
    if args.plan_only:
        plan["original_files_unchanged"] = verify_unchanged(before, files)
        return plan
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("The output directory became populated. Nothing was overwritten.")
    write_json(output / "plan.json", plan)
    write_json(output / "settings.json", settings)
    try:
        started = time.perf_counter()
        model, epoch, curve = train_with_inner_selection("RNN", prepared, settings, rules)
        fit_seconds = time.perf_counter() - started
        future_reference = levels["brent_fut_Close_conocido"]
        metrics, predictions = measure("RNN", "signals", 42, model, prepared, features, rules, future_reference)
        torch.save(model.state_dict(), output / "model.pt")
        metrics.to_csv(output / "metrics.csv", index=False)
        predictions.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(curve).to_csv(output / "curve.csv", index=False)
        write_json(output / "scaler.json", {**prepared["scaler"], "target_scale": prepared["target_scale"].tolist()})
        diagnostics = {**curve_diagnostics(curve, epoch), "fit_seconds": fit_seconds,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "selection_source": "validation_2018_2022", "test_evaluated": False}
        write_json(output / "diagnostics.json", diagnostics)
        _, _, reload_difference = verify_saved(output, "RNN", "signals", 42, prepared, features, rules, future_reference)
        reference = compare_original(output, original)
        integrity = verify_unchanged(before, files)
        manifest = {**plan, "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_epoch": epoch, "epochs_run": len(curve), "fit_seconds": fit_seconds,
                    "checkpoint_reload_maximum_prediction_difference_usd": reload_difference,
                    "original_files_unchanged": integrity, "original_file_sha256": before,
                    "data_manifest": data_manifest, "reference_comparison": reference,
                    "outputs_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
        write_json(output / "manifest.json", manifest)
        return {"status": "completed", "output": output.relative_to(ROOT).as_posix(),
                "manifest": (output / "manifest.json").relative_to(ROOT).as_posix(),
                "selected_epoch": epoch, "epochs_run": len(curve), "fit_seconds": fit_seconds,
                "metrics": metrics.to_dict(orient="records"), "reference_comparison": reference,
                "original_files_unchanged": integrity, "test_evaluated": False, "delivery_model_replaced": False}
    finally:
        # Run the check even when fitting/exporting fails; never silently leave
        # existing checkpoint or source changes unnoticed.
        verify_unchanged(before, files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true", help="Print partitions, settings and source hashes as JSON; write nothing and do not train.")
    parser.add_argument("--output", help="New/empty output directory inside this repository (default: runs/reproduction/<UTC timestamp>). Existing populated directories are refused.")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
