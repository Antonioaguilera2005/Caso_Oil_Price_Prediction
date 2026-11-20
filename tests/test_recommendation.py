"""La recomendación no debe depender solo de la mejor semilla."""
import json
import unittest

import pandas as pd

from scripts.download_data import ROOT
from scripts.report_optuna import recommend


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "config/optuna_confirmation.json").read_text(encoding="utf-8"))

    def scores(self, challenger):
        return pd.DataFrame({"trial": [0] * 3 + [1] * 3, "seed": [42, 43, 44] * 2,
                             "score": [1., 1., 1.] + challenger})

    def test_one_lucky_seed_does_not_replace_baseline(self):
        self.assertEqual(recommend(self.scores([.7, 1.02, 1.02]), self.policy), 0)

    def test_consistent_gain_can_replace_baseline(self):
        self.assertEqual(recommend(self.scores([.94, .95, .96]), self.policy), 1)

    def test_tiny_gain_keeps_simple_control(self):
        self.assertEqual(recommend(self.scores([.999, .999, .999]), self.policy), 0)

    def test_unstable_challenger_keeps_control(self):
        self.assertEqual(recommend(self.scores([.7, .99, 1.01]), self.policy), 0)


if __name__ == "__main__":
    unittest.main()
