"""Interpretación del R² y de las unidades de los errores."""
import unittest

import numpy as np

from study.metrics import price_metrics


class PriceMetricsTests(unittest.TestCase):
    def test_perfect_and_worse_than_mean_forecasts(self):
        actual = np.array([10., 20., 30.])
        self.assertEqual(price_metrics(actual, actual)["r2"], 1.)
        self.assertEqual(price_metrics(actual, np.full(3, 20.))["r2"], 0.)
        self.assertLess(price_metrics(actual, np.full(3, 100.))["r2"], 0.)

    def test_units_change_errors_but_not_r2(self):
        actual, predicted = np.array([1., 2., 3.]), np.array([1., 2., 4.])
        dollars = price_metrics(actual, predicted)
        cents = price_metrics(actual * 100, predicted * 100)
        self.assertAlmostEqual(dollars["r2"], .5)
        self.assertAlmostEqual(cents["r2"], dollars["r2"])
        self.assertAlmostEqual(cents["mae"], dollars["mae"] * 100)
        self.assertAlmostEqual(cents["rmse"], dollars["rmse"] * 100)
        self.assertAlmostEqual(cents["mse"], dollars["mse"] * 10000)

    def test_constant_target_has_undefined_r2_and_missing_values_fail(self):
        self.assertTrue(np.isnan(price_metrics([5., 5.], [5., 5.])["r2"]))
        with self.assertRaises(ValueError):
            price_metrics([5., np.nan], [5., 5.])


if __name__ == "__main__":
    unittest.main()
