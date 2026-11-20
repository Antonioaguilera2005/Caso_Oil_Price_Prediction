"""Regresiones: etiquetas por horizonte y evaluación exterior fuera de selección."""
import copy
import json
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.data import load_development
from study.validation import masked_samples, prepare_partition, train_with_inner_selection


class ValidationV2Tests(unittest.TestCase):
    def test_missing_long_target_does_not_remove_short_target(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        features = pd.DataFrame({"brent_conocido": np.arange(30.) + 10}, index=dates)
        target = pd.Series(np.arange(30.) + 20, index=dates)
        target.iloc[18] = np.nan
        positions, labels, _ = masked_samples(features, target, dates[10], dates[25], 5, [1, 3])
        row = np.flatnonzero(positions == 15)[0]
        self.assertEqual(labels[row, 0], target.iloc[16])
        self.assertTrue(np.isnan(labels[row, 1]))
        self.assertIn(23, positions)
        self.assertTrue(np.isnan(labels[np.flatnonzero(positions == 23)[0], 1]))

    def test_actual_partitions_purge_both_boundaries(self):
        features, history, protocol, _ = load_development()
        rules = json.loads((ROOT / "config/validation_v2.json").read_text())
        part = prepare_partition(features, history, protocol, rules, 2018)
        inner_start = pd.Timestamp(part["plan"]["fit_before"])
        self.assertLess(pd.Timestamp(part["scaler"]["last_fitted_date"]), inner_start)
        for subset, boundary in [("fit", inner_start), ("inner", pd.Timestamp("2018-01-01")),
                                 ("outer", pd.Timestamp("2022-01-01"))]:
            positions, labels, _ = part[subset]
            for column, horizon in enumerate(protocol["horizons"]):
                ids = positions[np.isfinite(labels[:, column])]
                self.assertLess(features.index[ids + horizon].max(), boundary)
        self.assertGreaterEqual(min(part["plan"]["inner_labels_by_horizon"]), 20)
        self.assertGreater(part["plan"]["outer_labels_by_horizon"][0], 241)

    def test_outer_labels_and_windows_cannot_change_weights_or_epoch(self):
        generator = np.random.default_rng(17)
        def subset(ids):
            return np.array(ids), generator.normal(10, 1, (len(ids), 2)), np.full(len(ids), 10.)
        fit, inner, outer = subset(range(3, 19)), subset(range(20, 28)), subset(range(31, 40))
        fit[1][0, 1] = np.nan
        part = {"inputs": torch.tensor(generator.normal(size=(40, 4, 2)), dtype=torch.float32),
                "targets": torch.tensor(fit[1] - fit[2][:, None], dtype=torch.float32),
                "target_scale": np.ones(2, dtype="float32"), "window": 4,
                "fit": fit, "inner": inner, "outer": outer}
        changed = copy.deepcopy(part)
        changed["outer"][1][:] = 1e10
        changed["inputs"][28:] = 1e8
        settings = {"seed": 7, "hidden_size": 4, "num_layers": 1, "dropout": .2,
                    "pooling": "last", "learning_rate": .001, "epochs": 3,
                    "batch_size": 8, "gradient_clip_norm": 1.}
        for options in [settings, {**settings, "num_layers": 2, "head_size": 4, "pooling": "last_mean",
                                    "loss": "huber", "weight_decay": .001, "scheduler": True}]:
            first, epoch, curve = train_with_inner_selection("GRU", part, options, {"patience": 2})
            second, other_epoch, other_curve = train_with_inner_selection("GRU", changed, options, {"patience": 2})
            self.assertEqual(epoch, other_epoch)
            self.assertEqual(curve, other_curve)
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
