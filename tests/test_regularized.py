"""Isolation and reproducibility of internal selection and full-history refitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.extended import load_extended
from study.models import PriceNetwork
from study.regularized import prepare, fit_network, predict, ridge_forecast


class RegularizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features, cls.history, *_ = load_extended("signals")
        cls.rules = json.loads((ROOT / "config/regularization_20261118.json").read_text(encoding="utf-8"))
        cls.rules.update(maximum_epochs=3, patience=2)
        cls.settings = {**cls.rules["base"], "hidden_size": 4, "window": 20}

    def parts(self, features=None, history=None, settings=None):
        return prepare(self.features if features is None else features,
                       self.history if history is None else history, self.rules,
                       settings or self.settings, "2018-01-01", "2023-01-01", [5, 63, 252])

    def test_boundaries_and_trainer_inputs_exclude_evaluation_labels(self):
        selection, refit, evaluation = self.parts()
        self.assertEqual(set(selection["subsets"]), {"fit", "inner"})
        self.assertEqual(set(refit["subsets"]), {"fit"})
        for part, cutoff in [(selection["subsets"]["fit"], "2016-01-01"),
                             (selection["subsets"]["inner"], "2018-01-01"),
                             (refit["subsets"]["fit"], "2018-01-01"), (evaluation, "2023-01-01")]:
            for col, h in enumerate([5, 63, 252]):
                ids = part[0][np.isfinite(part[1][:, col])]
                self.assertLess(self.features.index[ids + h].max(), pd.Timestamp(cutoff))
        self.assertLess(pd.Timestamp(selection["scaler"]["last_fitted_date"]), pd.Timestamp("2016-01-01"))
        self.assertLess(pd.Timestamp(refit["scaler"]["last_fitted_date"]), pd.Timestamp("2018-01-01"))

    def test_scored_validation_cannot_change_epoch_or_refitted_weights(self):
        first, full, _ = self.parts()
        features, history = self.features.copy(), self.history.copy()
        features.loc[features.index >= "2018-01-01"] += 500
        history.loc[pd.to_datetime(history.period_start_date) >= "2018-01-01", "DCOILBRENTEU"] += 500
        changed, full_changed, _ = self.parts(features, history)
        model, epoch, curve = fit_network("RNN", first, self.settings, self.rules, 42)
        other, other_epoch, other_curve = fit_network("RNN", changed, self.settings, self.rules, 42)
        self.assertEqual(epoch, other_epoch)
        self.assertEqual(curve, other_curve)
        for a, b in zip(model.state_dict().values(), other.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        a, _, _ = fit_network("RNN", full, self.settings, self.rules, 42, fixed_epochs=epoch)
        b, _, _ = fit_network("RNN", full_changed, self.settings, self.rules, 42, fixed_epochs=epoch)
        for x, y in zip(a.state_dict().values(), b.state_dict().values()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_zero_log_return_output_is_exactly_known_spot(self):
        settings = {**self.settings, "target_mode": "log_return"}
        _, full, evaluation = self.parts(settings=settings)
        model = PriceNetwork("GRU", self.features.shape[1], settings, 3)
        forecast = predict(model, full, evaluation)
        np.testing.assert_array_equal(forecast, np.repeat(evaluation[2][:, None], 3, axis=1))

    def test_refit_replays_learning_rates_without_inner_access(self):
        _, full, _ = self.parts()
        _, epoch, curve = fit_network("RNN", full, self.settings, self.rules, 42,
                                      fixed_epochs=2, learning_rates=[.001, .0002])
        self.assertEqual(epoch, 2)
        self.assertEqual([r["learning_rate"] for r in curve], [.001, .0002])
        self.assertTrue(all("inner_mse" not in r for r in curve))

    def test_ridge_does_not_use_evaluation_targets(self):
        selection, full, evaluation = self.parts()
        forecast, alpha, _ = ridge_forecast(selection, full, evaluation, [100., 1000.])
        altered = (evaluation[0], evaluation[1] * 100, evaluation[2])
        other, second_alpha, _ = ridge_forecast(selection, full, altered, [100., 1000.])
        self.assertEqual(alpha, second_alpha)
        np.testing.assert_array_equal(forecast, other)

    def test_future_data_are_rejected(self):
        row = self.features.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex(["2023-01-02"])
        with self.assertRaises(ValueError):
            self.parts(pd.concat([self.features, row]))

    def test_separate_horizon_job_saves_and_replays_all_branches(self):
        from scripts import run_regularized as runner
        rules = copy.deepcopy(self.rules)
        rules["base"].update(hidden_size=4, window=20)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner, "OUTPUT", Path(temporary)):
                item = runner.job(rules, "signals", "separate_returns", "RNN", 42)
                first = runner.do_job(item)
                second = runner.do_job(item)
                self.assertEqual(first["branches"], 3)
                self.assertEqual(first["physical_neural_fits"], 6)
                self.assertEqual(first["maximum_prediction_difference"], 0.)
                self.assertTrue(second["cached"])
                self.assertEqual(first["score"], second["score"])


if __name__ == "__main__":
    unittest.main()
