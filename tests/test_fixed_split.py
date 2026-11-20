"""Temporal isolation for the single fixed train/validation partition."""
import json
import unittest

import numpy as np
import pandas as pd

from scripts.download_data import ROOT
from study.extended import load_extended
from study.fixed import prepare_fixed
from study.models import PriceNetwork
from scripts.run_fixed_split import measure


class FixedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features, cls.history, *_ = load_extended("levels")
        cls.rules = json.loads((ROOT / "config/fixed_split_2017_2022.json").read_text(encoding="utf-8"))

    def test_origins_and_labels_obey_fixed_boundaries(self):
        part = prepare_fixed(self.features, self.history, self.rules)
        for name, start, end in [("fit", self.features.index[0], "2018-01-01"),
                                  ("inner", "2018-01-01", "2023-01-01")]:
            positions, labels, _ = part[name]
            for col, horizon in enumerate(self.rules["horizons"]):
                ids = positions[np.isfinite(labels[:, col])]
                self.assertTrue((self.features.index[ids] >= pd.Timestamp(start)).all())
                self.assertTrue((self.features.index[ids + horizon] < pd.Timestamp(end)).all())
        self.assertNotIn("outer", part)
        self.assertFalse(part["plan"]["annual_weight_refit"])

    def test_validation_changes_do_not_change_training_or_scaling(self):
        original = prepare_fixed(self.features, self.history, self.rules)
        features, history = self.features.copy(), self.history.copy()
        features.loc[features.index >= "2018-01-01"] *= 10
        future = pd.to_datetime(history.period_start_date) >= pd.Timestamp("2018-01-01")
        history.loc[future, "DCOILBRENTEU"] *= 10
        changed = prepare_fixed(features, history, self.rules)
        self.assertEqual(original["scaler"], changed["scaler"])
        np.testing.assert_array_equal(original["target_scale"], changed["target_scale"])
        np.testing.assert_array_equal(original["targets"], changed["targets"])
        ids = original["fit"][0] - self.rules["window"] + 1
        np.testing.assert_array_equal(original["inputs"][ids], changed["inputs"][ids])

    def test_test_inputs_and_target_history_are_rejected(self):
        row = self.features.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex(["2023-01-02"])
        with self.assertRaises(ValueError):
            prepare_fixed(pd.concat([self.features, row]), self.history, self.rules)
        history = self.history.copy()
        history.loc[history.index[-1], "realtime_start_date"] = "2023-01-02"
        with self.assertRaises(ValueError):
            prepare_fixed(self.features, history, self.rules)

    def test_signals_use_the_same_price_baseline(self):
        import torch
        torch.set_num_threads(1)
        signals, history, *_ = load_extended("signals")
        part = prepare_fixed(signals, history, self.rules)
        settings = {"pooling": "last", "hidden_size": 4, "num_layers": 1, "dropout": 0.}
        model = PriceNetwork("GRU", signals.shape[1], settings, len(self.rules["horizons"]))
        future = self.features["brent_fut_Close_conocido"]
        measured, forecast = measure("GRU", "signals", 42, model, part, signals, self.rules, future)
        np.testing.assert_allclose(measured.relative_rmse, 1., atol=1e-7)
        np.testing.assert_array_equal(forecast.future_naive, future.reindex(pd.to_datetime(forecast.origin)))


if __name__ == "__main__":
    unittest.main()
