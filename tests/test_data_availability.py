"""Comprobar publicación, revisiones y separación del objetivo observado."""
import unittest

import numpy as np
import pandas as pd

from scripts.build_data_tables import ROOT, replay_history, validate_history, yahoo_available


def history(rows):
    frame = pd.DataFrame(rows, columns=["period_start_date", "PRICE", "realtime_start_date", "realtime_end_date"])
    return validate_history(frame, "PRICE")


class AvailabilityTests(unittest.TestCase):
    def test_publication_and_revision_do_not_arrive_early(self):
        versions = history([
            ("2020-01-02", 10., "2020-01-03", "2020-01-07"),
            ("2020-01-02", 20., "2020-01-08", None),
            ("2020-01-06", 30., "2020-01-10", None),
        ])
        replay = replay_history(versions, pd.bdate_range("2020-01-02", "2020-01-10"))
        self.assertTrue(pd.isna(replay.loc["2020-01-02", "value"]))
        self.assertEqual(replay.loc["2020-01-07", "value"], 10.)
        self.assertEqual(replay.loc["2020-01-08", "value"], 20.)
        self.assertEqual(replay.loc["2020-01-10", "value"], 30.)

    def test_withdrawn_observation_is_not_carried_forward(self):
        versions = history([
            ("2020-01-01", 5., "2020-01-02", None),
            ("2020-01-02", 10., "2020-01-03", "2020-01-07"),
            ("2020-01-02", np.nan, "2020-01-08", None),
        ])
        replay = replay_history(versions, pd.bdate_range("2020-01-03", "2020-01-09"))
        self.assertEqual(replay.loc["2020-01-07", "value"], 10.)
        self.assertEqual(replay.loc["2020-01-08", "value"], 5.)

    def test_overlapping_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "solapada"):
            history([("2020-01-02", 10., "2020-01-03", None),
                     ("2020-01-02", 20., "2020-01-08", None)])

    def test_future_release_cannot_change_earlier_panel(self):
        original = history([("2020-01-02", 10., "2020-01-03", None)])
        changed = history([("2020-01-02", 10., "2020-01-03", "2020-01-09"),
                           ("2020-01-02", 1000., "2020-01-10", None)])
        dates = pd.bdate_range("2020-01-02", "2020-01-09")
        pd.testing.assert_frame_equal(replay_history(original, dates), replay_history(changed, dates))

    def test_yahoo_friday_close_used_from_monday(self):
        source = pd.DataFrame({"close": [10., 20.]}, index=pd.to_datetime(["2020-01-09", "2020-01-10"]))
        result = yahoo_available(source, pd.bdate_range("2020-01-09", "2020-01-13"))
        self.assertTrue(pd.isna(result.loc["2020-01-09", "close"]))
        self.assertEqual(result.loc["2020-01-10", "close"], 10.)
        self.assertEqual(result.loc["2020-01-13", "close"], 20.)

    def test_processed_target_matches_original_including_missing(self):
        path = ROOT / "data/processed/observaciones_diarias.csv"
        if not path.exists():
            self.skipTest("Ejecutar antes scripts.build_data_tables")
        observed = pd.read_csv(path, index_col=0, parse_dates=True)
        original = pd.read_csv(ROOT / "data/raw/fred_DCOILBRENTEU.csv", index_col=0,
                               parse_dates=True, na_values=".").iloc[:, 0]
        pd.testing.assert_series_equal(observed.brent, original.reindex(observed.index), check_names=False)
        trace = pd.read_csv(ROOT / "data/processed/procedencia_temporal.csv", index_col=0, parse_dates=True)
        for column in trace:
            if column.endswith(("observation_date", "available_from")):
                values = pd.to_datetime(trace[column].dropna())
                self.assertTrue((values <= values.index).all(), column)


if __name__ == "__main__":
    unittest.main()
