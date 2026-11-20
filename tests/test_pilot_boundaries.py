"""Separación temporal del piloto y versiones de sus etiquetas."""
import unittest

import numpy as np
import pandas as pd

from study.data import load_development, target_snapshot, samples, fit_scaler


class PilotBoundaryTests(unittest.TestCase):
    def test_target_revision_only_used_after_publication(self):
        history = pd.DataFrame({"period_start_date": ["2017-12-20", "2017-12-20", "2017-12-29"],
                                "DCOILBRENTEU": [50., 90., 60.],
                                "realtime_start_date": ["2017-12-22", "2018-01-05", "2018-01-03"],
                                "realtime_end_date": ["2018-01-04", None, None]})
        dates = pd.bdate_range("2017-12-20", "2017-12-29")
        training = target_snapshot(history, "2018-01-01", dates)
        self.assertEqual(training.loc["2017-12-20"], 50.)
        self.assertTrue(pd.isna(training.loc["2017-12-29"]))
        later = target_snapshot(history, "2018-01-06", dates)
        self.assertEqual(later.loc["2017-12-20"], 90.)

    def test_windows_and_destinations_stay_inside_block(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        features = pd.DataFrame({"brent_conocido": np.arange(30.) + 10}, index=dates)
        target = pd.Series(np.arange(30.) + 20, index=dates)
        target.iloc[18] = np.nan
        positions, labels, base = samples(features, target, dates[10], dates[25], 5, [1, 3])
        self.assertTrue((positions >= 10).all())
        self.assertTrue((positions + 3 < 25).all())
        self.assertNotIn(17, positions)
        self.assertNotIn(15, positions)
        self.assertTrue(np.isfinite(labels).all())
        np.testing.assert_array_equal(base, features.brent_conocido.iloc[positions])

    def test_future_features_do_not_affect_scaler(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        original = pd.DataFrame({"brent_conocido": np.arange(30.) + 10}, index=dates)
        changed = original.copy()
        changed.iloc[20:] = 1e12
        self.assertEqual(fit_scaler(original, dates[20]), fit_scaler(changed, dates[20]))

    def test_pilot_files_and_samples_exclude_calibration_and_test(self):
        features, history, protocol, manifest = load_development()
        cutoff = pd.Timestamp(protocol["development_end_exclusive"])
        self.assertLess(features.index.max(), cutoff)
        self.assertLess(pd.to_datetime(history.realtime_start_date).max(), cutoff)
        self.assertFalse(manifest["calibration_exported"])
        self.assertFalse(manifest["test_exported"])
        for fold in protocol["validation_folds"]:
            training_target = target_snapshot(history, fold["train_end_exclusive"], features.index)
            valid_target = target_snapshot(history, fold["validation_end_exclusive"], features.index)
            training, _, _ = samples(features, training_target, protocol["start"], fold["train_end_exclusive"],
                                      protocol["window"], protocol["horizons"])
            validation, _, _ = samples(features, valid_target, fold["train_end_exclusive"], fold["validation_end_exclusive"],
                                        protocol["window"], protocol["horizons"])
            self.assertLess(features.index[training + max(protocol["horizons"])].max(), features.index[validation].min())
            self.assertLess(features.index[validation + max(protocol["horizons"])].max(), cutoff)


if __name__ == "__main__":
    unittest.main()
