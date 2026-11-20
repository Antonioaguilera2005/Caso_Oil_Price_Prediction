"""La ampliación y las señales no incorporan el test ni información futura."""
import unittest

import numpy as np
import pandas as pd

from study.extended import causal_signals, load_extended
from study.validation import prepare_partition


class ExtendedTests(unittest.TestCase):
    def test_signals_are_causal_and_preserve_negative_wti(self):
        dates = pd.bdate_range("2010-01-01", periods=600)
        frame = pd.DataFrame({name: 30. + np.sin(np.arange(600) / 13.) for name in
            ["brent_conocido", "wti_conocido", "oro_conocido", "cobre_conocido", "indice_dolar_conocido",
             "vix_conocido", "brent_fut_Close_conocido"]}, index=dates)
        frame.loc[dates[300], "wti_conocido"] = -37.
        original = causal_signals(frame)
        altered = frame.copy()
        altered.iloc[400:] = 999999.
        pd.testing.assert_frame_equal(original.iloc[:400], causal_signals(altered).iloc[:400])
        self.assertTrue(np.isfinite(original.loc[dates[300], "wti_asinh_change"]))

    def test_exported_variants_share_calendar_and_exclude_test(self):
        levels, history, protocol, rules, manifest = load_extended("levels")
        signals, *_ = load_extended("signals")
        pd.testing.assert_index_equal(levels.index, signals.index)
        self.assertEqual(levels.shape, signals.shape)
        self.assertLess(levels.index.max(), pd.Timestamp("2023-01-01"))
        self.assertLess(pd.to_datetime(history.realtime_start_date).max(), pd.Timestamp("2023-01-01"))
        self.assertFalse(manifest["test_exported"])

    def test_all_label_destinations_stay_before_their_boundaries(self):
        features, history, protocol, rules, _ = load_extended("levels")
        for year in [2021, 2022]:
            part = prepare_partition(features, history, protocol, rules, year)
            for subset, boundary in [("fit", part["plan"]["fit_before"]), ("inner", f"{year}-01-01"),
                                     ("outer", "2023-01-01")]:
                positions, labels, _ = part[subset]
                for column, horizon in enumerate(protocol["horizons"]):
                    ids = positions[np.isfinite(labels[:, column])]
                    self.assertLess(features.index[ids + horizon].max(), pd.Timestamp(boundary))
            self.assertLess(pd.Timestamp(part["scaler"]["last_fitted_date"]), pd.Timestamp(part["plan"]["fit_before"]))


if __name__ == "__main__":
    unittest.main()
