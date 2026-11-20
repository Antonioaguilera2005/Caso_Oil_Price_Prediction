import unittest

import numpy as np

from study.diagnostics import block_ratio_interval
from study.metrics import price_metrics


class DiagnosticsTests(unittest.TestCase):
    def test_identical_forecasts_give_ratio_one_in_every_resample(self):
        errors = np.arange(1., 50.)
        measured = block_ratio_interval(errors, errors, 12)
        self.assertEqual(measured["ratio"], 1.)
        self.assertEqual(measured["ratio_low"], 1.)
        self.assertEqual(measured["ratio_high"], 1.)

    def test_uniform_error_reduction_preserves_ratio_under_resampling(self):
        errors = np.arange(1., 50.)
        measured = block_ratio_interval(errors * .81, errors, 12)
        self.assertAlmostEqual(measured["ratio_low"], .9)
        self.assertAlmostEqual(measured["ratio_high"], .9)

    def test_ratio_and_r2_identity_share_same_observations(self):
        actual = np.array([12., 15., 20., 17.])
        naive = np.array([10., 12., 18., 14.])
        predicted = np.array([11., 14., 18., 16.])
        model, base = price_metrics(actual, predicted), price_metrics(actual, naive)
        ratio = model["rmse"] / base["rmse"]
        self.assertAlmostEqual(model["r2"], 1 - ratio ** 2 * (1 - base["r2"]))

    def test_one_effective_block_does_not_produce_spurious_precision(self):
        result = block_ratio_interval(np.ones(253), np.arange(1., 254.), 252)
        self.assertIsNone(result["ratio_low"])
        self.assertIsNone(result["ratio_high"])
        self.assertEqual(result["valid_repetitions"], 0)


if __name__ == "__main__":
    unittest.main()
