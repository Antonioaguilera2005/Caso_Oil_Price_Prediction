"""Read-only, standard-library verification of the selected model and saved release.

This checks stored evidence; it does not train, infer, download or rewrite files.
The 2023--2024 evaluation is a historical comparison of a previously examined period.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {5: 499, 63: 441, 252: 258}
NUMERIC_METRICS = ("mae", "mse", "rmse", "r2", "relative_rmse",
                   "relative_rmse_vs_future", "naive_rmse", "future_rmse")


def calculate_metrics(rows):
    """Compute price metrics directly from saved observations, without ML packages."""
    if not rows:
        raise ValueError("A saved metric group has no prediction observations")
    actual = [float(row["actual"]) for row in rows]
    predicted = [float(row["prediction"]) for row in rows]
    naive = [float(row["naive"]) for row in rows]
    future = [float(row["future_naive"]) for row in rows]
    if not all(math.isfinite(value) for column in (actual, predicted, naive, future) for value in column):
        raise ValueError("Saved predictions contain missing or nonfinite prices")
    n = len(rows)
    mean = math.fsum(actual) / n
    squared = math.fsum((left - right) ** 2 for left, right in zip(actual, predicted))
    naive_squared = math.fsum((left - right) ** 2 for left, right in zip(actual, naive))
    future_squared = math.fsum((left - right) ** 2 for left, right in zip(actual, future))
    total = math.fsum((value - mean) ** 2 for value in actual)
    return {"n": n, "mae": math.fsum(abs(left - right) for left, right in zip(actual, predicted)) / n,
            "mse": squared / n, "rmse": math.sqrt(squared / n),
            "r2": 1 - squared / total if total else None,
            "relative_rmse": math.sqrt(squared / naive_squared) if naive_squared else None,
            "relative_rmse_vs_future": math.sqrt(squared / future_squared) if future_squared else None,
            "naive_rmse": math.sqrt(naive_squared / n), "future_rmse": math.sqrt(future_squared / n)}


def _model_key(row):
    return row["architecture"], row["variant"], int(row["seed"]), int(row["horizon"])


def _prediction_key(row):
    return int(row["horizon"]), row["origin"], row["target_date"]


class ReleaseAudit:
    def __init__(self, root, require_build=False):
        self.root = Path(root).resolve()
        self.require_build = require_build
        self.errors = []
        self.warnings = []
        self.checks = 0
        self.digests = {}
        self.summary = {"operation": "read_only_verification", "models_retrained": 0,
                        "predictions_recomputed": False,
                        "evaluation": "Historical 2023-2024 comparison; this period was previously examined."}

    def path(self, relative, base=None):
        candidate = ((base or self.root) / str(relative).replace("\\", "/")).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Manifest path leaves the repository: {relative}")
        return candidate

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def require(self, condition, message):
        if not condition:
            raise ValueError(message)
        self.checks += 1

    def read_json(self, relative, base=None):
        with self.path(relative, base).open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_csv(self, relative):
        with self.path(relative).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def digest(self, path):
        if path not in self.digests:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            self.digests[path] = hasher.hexdigest()
        return self.digests[path]

    def verify_hashes(self, files, base=None):
        self.require(isinstance(files, dict) and bool(files), "A required hash manifest is empty")
        for relative, expected in files.items():
            path = self.path(relative, base)
            if not path.is_file():
                self.check(False, f"Missing file recorded in a frozen manifest: {path.relative_to(self.root)}")
                continue
            self.check(self.digest(path) == expected, f"SHA-256 mismatch: {path.relative_to(self.root)}")

    def compare_metrics(self, computed, saved, label):
        self.check(int(saved["n"]) == computed["n"], f"Observation count mismatch: {label}")
        for name in NUMERIC_METRICS:
            value = computed[name]
            stored = saved.get(name)
            if value is None:
                valid = stored in (None, "", "nan", "NaN")
            else:
                valid = stored is not None and math.isclose(value, float(stored), rel_tol=1e-10, abs_tol=1e-9)
            self.check(valid, f"Saved {name} differs from predictions: {label}")

    def verify(self):
        config = self.read_json("config/delivery_model.json")
        self.require(config["architecture"] == "RNN" and config["variant"] == "signals" and config["seed"] == 42,
                     "Delivery configuration does not select the evaluated RNN/signals/seed42")
        self.require(config["window"] == 60 and config["horizons"] == [5, 63, 252],
                     "The selected input window or horizons changed")
        evaluation = self.path(config["evaluation"])
        plan = self.read_json("plan.json", evaluation)
        complete = self.read_json("complete.json", evaluation)
        self.verify_hashes({"plan.json": complete["plan_sha256"]}, evaluation)
        self.verify_hashes(complete["files"], evaluation)
        self.verify_hashes(plan["files"])
        self.check(complete["models_retrained"] == 0 and plan["models_retrained"] == 0,
                   "The recovered-model evaluation records training")
        self.require(bool(plan.get("prior_period_evaluation")), "The historical evaluation context is missing")
        selected = [entry for entry in plan["models"] if (entry["architecture"], entry["variant"], entry["seed"]) ==
                    (config["architecture"], config["variant"], config["seed"])]
        self.require(len(selected) == 1, "The chosen model is absent or duplicated in the frozen evaluation plan")
        chosen = selected[0]
        self.check(chosen["checkpoint"] == config["checkpoint"] and chosen["scaler"] == config["scaler"],
                   "Delivery weights/scaler differ from the frozen evaluated model")
        settings, scaler = self.read_json(config["settings"]), self.read_json(config["scaler"])
        self.check(settings == chosen["settings"], "Delivery settings differ from the frozen evaluation plan")
        self.require(all(path in plan["files"] for path in (config["checkpoint"], config["scaler"], config["settings"])),
                     "Selected checkpoint/scaler/settings lack frozen hashes")

        relative_predictions = str(evaluation.relative_to(self.root) / "predictions.csv")
        predictions = self.read_csv(relative_predictions)
        saved_metrics = self.read_csv(str(evaluation.relative_to(self.root) / "metrics.csv"))
        groups = defaultdict(list)
        seen = set()
        for row in predictions:
            key = _model_key(row)
            pair = (*key, row["origin"], row["target_date"])
            self.require(pair not in seen, f"Duplicate saved forecast: {pair}")
            seen.add(pair)
            origin, target = date.fromisoformat(row["origin"]), date.fromisoformat(row["target_date"])
            self.require(date(2023, 1, 1) <= origin < target < date(2025, 1, 1),
                         f"Forecast leaves the historical evaluation interval: {pair}")
            remaining, expected_target = key[3], origin
            while remaining:
                expected_target += timedelta(days=1)
                if expected_target.weekday() < 5:
                    remaining -= 1
            self.require(target == expected_target, f"Forecast horizon/date mismatch: {pair}")
            groups[key].append(row)
        self.require(len(groups) == 18, "The six-model/three-horizon comparison is incomplete")
        metric_groups = set()
        for metric in saved_metrics:
            key, group = _model_key(metric), metric["group"]
            if group == "all":
                subset = groups[key]
            elif group.startswith("target_year_"):
                year = group.removeprefix("target_year_")
                subset = [row for row in groups[key] if row["target_date"].startswith(year + "-")]
            elif group in ("origin_usual", "origin_high_volatility"):
                subset = [row for row in groups[key] if row["regime"] == group.removeprefix("origin_")]
            else:
                raise ValueError(f"Unknown saved metric group: {group}")
            self.require((*key, group) not in metric_groups, f"Duplicate metric group: {key}/{group}")
            metric_groups.add((*key, group))
            self.compare_metrics(calculate_metrics(subset), metric, f"{key}/{group}")
        expected_groups = {"all", "target_year_2023", "target_year_2024", "origin_usual", "origin_high_volatility"}
        self.check(len(metric_groups) == len(groups) * len(expected_groups), "Annual/regime metric groups are incomplete")
        selected_rows = [row for row in predictions if _model_key(row)[:3] == ("RNN", "signals", 42)]
        counts = dict(Counter(int(row["horizon"]) for row in selected_rows))
        self.check(counts == EXPECTED_COUNTS, f"Historical test counts changed: {counts}")
        long_2023 = sum(int(row["horizon"]) == 252 and row["target_date"].startswith("2023-") for row in selected_rows)
        self.check(long_2023 == 6, "The six 2023 target observations at 252 days must remain in the complete evaluation")

        export = self.read_json("app_site/data/export-manifest.json")
        self.verify_hashes(export["files"], self.path("app_site/data"))
        self.check(export["selectedModel"] == config and export["modelRetrained"] is False,
                   "The application export does not match the delivery configuration")
        checkpoint_digest = self.digest(self.path(config["checkpoint"]))
        self.check(export["checkpointSha256"] == checkpoint_digest, "The app records different checkpoint weights")
        self.check(export["evaluationCompleteSha256"] == self.digest(evaluation / "complete.json"),
                   "The app records a different historical evaluation")
        model = self.read_json("app_site/data/model.json")
        self.check(model["id"] == config["id"] and model["architecture"] == "RNN" and
                   model["window"] == config["window"] and model["horizons"] == config["horizons"],
                   "Application model identity, window or horizons changed")
        self.check(model["metadata"]["checkpointSha256"] == checkpoint_digest,
                   "Application model metadata records a different checkpoint")
        self.check(model["featureColumns"] == scaler["columns"] and
                   model["scaler"] == {"median": scaler["median"], "scale": scaler["scale"], "targetScale": scaler["target_scale"]},
                   "Application signals/scaling differ from the original training scaler")
        self.check(model["intervalHalfWidths"] == chosen["interval_half_widths"], "Application interval widths changed")
        app_evaluation = self.read_json("app_site/data/evaluation.json")
        app_rows = app_evaluation["predictions"]
        self.check(len(app_rows) == len(selected_rows), "Application evaluation loses saved forecast observations")
        canonical_rows = {_prediction_key(row): row for row in selected_rows}
        self.check(len({_prediction_key(row) for row in app_rows}) == len(app_rows), "Application evaluation contains duplicate observations")
        for row in app_rows:
            key = _prediction_key(row)
            self.require(key in canonical_rows, f"Application forecast absent from the canonical saved CSV: {key}")
            original = canonical_rows[key]
            for column in ("actual", "prediction", "naive", "future_naive", "lower", "upper"):
                self.check(math.isclose(float(row[column]), float(original[column]), rel_tol=0, abs_tol=1e-10),
                           f"Application {column} differs from the canonical forecast: {key}")
        for metric in app_evaluation["selected"]:
            horizon = int(metric["horizon"])
            self.compare_metrics(calculate_metrics(groups[("RNN", "signals", 42, horizon)]), metric, f"app/{horizon}")
        self.check(len(app_evaluation["selected"]) == 3, "Application must contain all three selected-model metric rows")
        comparison_keys = set()
        for metric in app_evaluation["comparison"]:
            key = _model_key(metric)
            self.require(key in groups and key not in comparison_keys,
                         f"Application comparison model is absent or duplicated: {key}")
            comparison_keys.add(key)
            self.compare_metrics(calculate_metrics(groups[key]), metric, f"app comparison/{key}")
        self.check(comparison_keys == set(groups), "Application comparison does not include all six models and three horizons")

        hub = self.path("model_hub/rnn-brent-signals")
        hub_manifest = self.read_json("manifest.json", hub)
        self.verify_hashes(hub_manifest["source_files"])
        self.verify_hashes(hub_manifest["files"], hub)
        self.check(hub_manifest["model_id"] == config["id"] and
                   hub_manifest["checkpoint_sha256"] == checkpoint_digest and hub_manifest["model_retrained"] is False,
                   "Portable model manifest identifies different weights or training")
        self.check(self.read_json("settings.json", hub) == settings and self.read_json("scaler.json", hub) == scaler,
                   "Portable model settings/scaler differ from the original model")
        hub_model = self.read_json("model.json", hub)
        for key in ("id", "architecture", "window", "horizons", "featureColumns", "scaler", "weights", "intervalHalfWidths"):
            self.check(hub_model[key] == model[key], f"Portable model {key} differs from the application model")
        hub_metrics = self.read_json("metrics.json", hub)
        for metric in hub_metrics["metrics"]:
            self.compare_metrics(calculate_metrics(groups[("RNN", "signals", 42, int(metric["horizon"]))]), metric,
                                 f"model_hub/{metric['horizon']}")

        for relative in ("README.md", "pyproject.toml", "poetry.lock", "app.py", "scripts/build_app.py",
                         "scripts/export_app.py", "app_site/index.html", "app_site/styles.css", "app_site/app.js", "app_site/engine.js",
                         "app_site/data/app-data.json", "app_site/data/inference-verification.json", "docs/assignment/ENUNCIADO.pdf"):
            self.check(self.path(relative).is_file(), f"Required source/release file is missing: {relative}")
        parity = self.read_json("app_site/data/inference-verification.json")
        self.check(parity["chosen_model"] == config["id"] and parity["models_retrained"] == 0 and
                   parity["maximum_price_difference_usd"] < parity["tolerance_usd"],
                   "Saved browser/Python inference verification is missing or failed")
        self.verify_build()
        self.summary.update({"model_id": config["id"], "checkpoint_sha256": checkpoint_digest,
                             "saved_metric_groups_recalculated": len(saved_metrics), "selected_test_counts": counts,
                             "preserved_252_day_targets_in_2023": long_2023})

    def verify_build(self):
        directory = self.path("app_site/dist")
        if not (directory / "build-manifest.json").is_file():
            message = "Static build is absent. Generate it with: python -m scripts.build_app"
            if self.require_build:
                self.check(False, message)
            else:
                self.warnings.append(message + " (optional for source verification)")
            self.summary["static_build"] = "not_built"
            return
        manifest = self.read_json("build-manifest.json", directory)
        self.verify_hashes(manifest["files"], directory)
        self.check(manifest["model_retrained"] is False, "Static build records model retraining")
        for relative in ("index.html", "styles.css", "app.js", "engine.js", "data/model.json", "data/app-data.json", "data/evaluation.json"):
            built, source = self.path(relative, directory), self.path(relative, self.path("app_site"))
            self.check(built.is_file() and source.is_file() and self.digest(built) == self.digest(source),
                       f"Static build is stale: {relative}. Run python -m scripts.build_app")
        self.summary["static_build"] = "verified"

    def result(self):
        return {"ok": not self.errors, **self.summary, "checks": self.checks,
                "hashed_files": len(self.digests), "warnings": self.warnings, "errors": self.errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository checkout to verify (defaults to this script's checkout)")
    parser.add_argument("--require-build", action="store_true", help="Fail unless the generated static app is present and current")
    arguments = parser.parse_args(argv)
    audit = ReleaseAudit(arguments.root, arguments.require_build)
    try:
        audit.verify()
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        audit.errors.append(f"Verification could not complete: {error}")
    print(json.dumps(audit.result(), ensure_ascii=True, indent=2, allow_nan=False))
    return 1 if audit.errors else 0


if __name__ == "__main__":
    sys.exit(main())
