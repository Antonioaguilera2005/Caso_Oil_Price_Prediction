"""Replay and causal inference checks for the saved fixed-partition models."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from scripts.evaluate_recovered_test import (
    SOURCE,
    metric_summary,
    predict_checkpoint,
    replay_checkpoint,
)
from study.extended import causal_signals, load_extended, read_panel_before


class RecoveredCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features = {}
        cls.history = {}
        cls.items = []
        for variant in ["levels", "signals"]:
            features, history, *_ = load_extended(variant)
            cls.features[variant] = features
            cls.history[variant] = history
            for architecture in ["RNN", "LSTM", "GRU"]:
                directory = SOURCE / architecture / variant / "seed_42"
                cls.items.append({
                    "architecture": architecture,
                    "variant": variant,
                    "seed": 42,
                    "checkpoint": str((directory / "model.pt").relative_to(ROOT)),
                    "settings": json.loads((directory / "settings.json").read_text(encoding="utf-8")),
                    "scaler": str((directory / "scaler.json").relative_to(ROOT)),
                })

    def test_all_six_displayed_checkpoints_replay_saved_validation(self):
        for item in self.items:
            with self.subTest(architecture=item["architecture"], variant=item["variant"]):
                result = replay_checkpoint(item)
                self.assertEqual(result["architecture"], item["architecture"])
                self.assertEqual(result["variant"], item["variant"])
                self.assertEqual(result["seed"], 42)
                self.assertGreater(result["rows_verified"], 1000)
                self.assertLessEqual(result["maximum_prediction_difference"], 1e-8)

    def test_causal_signals_and_inference_ignore_inputs_after_origins(self):
        panel = read_panel_before(ROOT / "data/processed/informacion_disponible.csv", "2023-01-01")
        original_signals = causal_signals(panel)
        signals = self.features["signals"]
        pd.testing.assert_frame_equal(original_signals.loc[signals.index], signals,
                                      check_freq=False, rtol=1e-12, atol=1e-12)
        positions = np.array([1800, 1830, 1860])
        cutoff = signals.index[positions.max()]
        changed_panel = panel.copy()
        changed_panel.loc[changed_panel.index > cutoff] *= 1.5
        changed_signals = causal_signals(changed_panel).loc[signals.index]
        pd.testing.assert_frame_equal(changed_signals.loc[:cutoff], original_signals.loc[signals.index].loc[:cutoff])
        for item in self.items:
            with self.subTest(architecture=item["architecture"], variant=item["variant"]):
                features = self.features[item["variant"]]
                changed = (changed_signals if item["variant"] == "signals"
                           else changed_panel.loc[features.index, features.columns])
                expected = predict_checkpoint(item, features, positions)
                actual = predict_checkpoint(item, changed, positions)
                self.assertEqual(actual.shape, (len(positions), 3))
                np.testing.assert_array_equal(actual, expected)

    def test_metric_ratios_use_each_models_correct_baselines(self):
        rows = []
        actual = np.array([10., 12., 14., 16.])
        for architecture, error in [("RNN", 1.), ("LSTM", 3.)]:
            for seed in [42, 43]:
                for horizon in [5, 63, 252]:
                    for index, value in enumerate(actual):
                        rows.append({
                            "architecture": architecture, "variant": "signals", "seed": seed,
                            "horizon": horizon, "origin": f"2021-01-{index + 4:02d}",
                            "target_date": f"2022-01-{index + 4:02d}",
                            "actual": value, "prediction": value - error - (seed - 42),
                            "naive": value - 2., "future_naive": value - 1., "regime": "usual",
                        })
        summary = metric_summary(pd.DataFrame(rows))
        overall = summary.loc[summary.group.eq("all")]
        if "model" in overall.columns:
            overall = overall.loc[~overall.model.isin(["naive_spot", "naive_future"])]
        for architecture, error in [("RNN", 1.), ("LSTM", 3.)]:
            for seed in [42, 43]:
                selected = overall.loc[overall.architecture.eq(architecture) & overall.seed.eq(seed)]
                self.assertEqual(len(selected), 3)
                np.testing.assert_allclose(selected.rmse, error + seed - 42)
                np.testing.assert_allclose(selected.relative_rmse, (error + seed - 42) / 2.)
                np.testing.assert_allclose(selected.relative_rmse_vs_future, error + seed - 42)
                np.testing.assert_array_equal(selected.n.to_numpy(), np.full(3, 4))

    def test_checkpoint_inference_never_creates_optimizer(self):
        positions = np.array([1800, 1830])
        with patch("torch.optim.Adam", side_effect=AssertionError("Inference must not train")):
            for item in self.items:
                result = predict_checkpoint(item, self.features[item["variant"]], positions)
                self.assertTrue(np.isfinite(result).all())

    def test_replay_uses_only_development_and_pre2018_scalers(self):
        for variant, features in self.features.items():
            self.assertLess(features.index.max(), pd.Timestamp("2023-01-01"))
            for column in ["period_start_date", "realtime_start_date"]:
                self.assertTrue((pd.to_datetime(self.history[variant][column]) < pd.Timestamp("2023-01-01")).all())
        with patch("scripts.evaluate_recovered_test.load_test_inputs", create=True,
                   side_effect=AssertionError("Replay must not open test inputs")):
            item = self.items[0]
            result = replay_checkpoint(item)
            self.assertGreater(result["rows_verified"], 1000)
        for item in self.items:
            scaler = json.loads((ROOT / item["scaler"]).read_text(encoding="utf-8"))
            self.assertLess(pd.Timestamp(scaler["last_fitted_date"]), pd.Timestamp("2018-01-01"))
            saved = pd.read_csv((ROOT / item["checkpoint"]).parent / "predictions.csv")
            saved = saved.loc[saved.subset.eq("validation")]
            for column in ["origin", "target_date"]:
                self.assertTrue(saved[column].ge("2018-01-01").all())
                self.assertTrue(saved[column].lt("2023-01-01").all())


if __name__ == "__main__":
    unittest.main()
