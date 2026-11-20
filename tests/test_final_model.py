"""Boundary, causal inference and selection checks for the locked final pipeline."""
import json
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.extended import load_extended
from study.final_model import blend, choose_weight, infer_network, origin_volatility, training_parts
from study.models import PriceNetwork
from study.regularized import fit_network, predict


class FinalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features, cls.history, *_ = load_extended("levels")
        cls.settings = json.loads((ROOT / "experiments/regularization_20261118/report/candidate.json").read_text(encoding="utf-8"))["settings"]

    def test_training_purges_targets_at_both_boundaries(self):
        selection, refit = training_parts(self.features, self.history, self.settings, [5, 63, 252])
        for part, stop in [(selection["subsets"]["fit"], "2021-01-01"),
                           (selection["subsets"]["inner"], "2023-01-01"),
                           (refit["subsets"]["fit"], "2023-01-01")]:
            for col, horizon in enumerate([5, 63, 252]):
                positions = part[0][np.isfinite(part[1][:, col])]
                self.assertLess(self.features.index[positions + horizon].max(), pd.Timestamp(stop))
        self.assertEqual(set(refit["subsets"]), {"fit"})
        self.assertLess(pd.Timestamp(refit["scaler"]["last_fitted_date"]), pd.Timestamp("2023-01-01"))

    def test_training_rejects_test_inputs_and_labels(self):
        row = self.features.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex(["2023-01-02"])
        with self.assertRaises(ValueError):
            training_parts(pd.concat([self.features, row]), self.history, self.settings, [5, 63, 252])
        history = self.history.copy()
        history.loc[0, "realtime_start_date"] = "2023-01-02"
        with self.assertRaises(ValueError):
            training_parts(self.features, history, self.settings, [5, 63, 252])

    def test_inference_matches_trainer_and_ignores_future_inputs(self):
        _, refit = training_parts(self.features, self.history, self.settings, [5, 63, 252])
        model, _, _ = fit_network("RNN", refit, self.settings, {}, 42, fixed_epochs=1)
        subset = tuple(a[:25] for a in refit["subsets"]["fit"])
        expected = predict(model, refit, subset)
        actual = infer_network(model, self.features, refit["scaler"], refit["scale"], 60, subset[0])
        np.testing.assert_array_equal(actual, expected)
        changed = self.features.copy()
        changed.iloc[subset[0].max() + 1:] += 500
        other = infer_network(model, changed, refit["scaler"], refit["scale"].tolist(), 60, subset[0])
        np.testing.assert_array_equal(other, actual)
        np.testing.assert_array_equal(origin_volatility(changed).iloc[:subset[0].max() + 1],
                                      origin_volatility(self.features).iloc[:subset[0].max() + 1])

    def test_weight_selection_uses_scores_and_prefers_smaller_weight_within_tie(self):
        rows = pd.DataFrame({"weight": [0., .25, .5, 1.], "horizon": [5] * 4,
                             "relative_rmse": [1., .98, .977, .96]})
        self.assertEqual(choose_weight(rows, .005), 1.)
        self.assertEqual(choose_weight(rows.iloc[:3], .005), .25)
        np.testing.assert_array_equal(blend([80., 100.], [90., 90.], 0), [90., 90.])


if __name__ == "__main__":
    unittest.main()
