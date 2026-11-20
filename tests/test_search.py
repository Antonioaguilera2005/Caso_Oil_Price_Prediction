"""Controles de las arquitecturas ampliadas y de la selección de hiperparámetros."""
import unittest

import numpy as np
import pandas as pd
import torch

from study.models import PriceNetwork
from study.search import objective_score, curve_diagnostics


class SearchTests(unittest.TestCase):
    def test_complex_models_start_naive_and_can_learn(self):
        torch.set_num_threads(1)
        for architecture in ["RNN", "LSTM", "GRU"]:
            torch.manual_seed(123)
            settings = {"hidden_size": 8, "num_layers": 2, "dropout": .2,
                        "head_size": 4, "pooling": "last_mean"}
            model = PriceNetwork(architecture, 3, settings, 3)
            # Activar la pequeña cabeza de prueba: una ReLU completamente
            # apagada puede dar gradiente cero sin que exista un fallo de red.
            with torch.no_grad():
                model.head[0].bias.fill_(10.)
            values = torch.randn(5, 7, 3)
            torch.testing.assert_close(model(values), torch.zeros(5, 3), rtol=0, atol=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                loss = (model(values) - 1).square().mean()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertGreater(float(model.recurrent.weight_ih_l0.grad.abs().sum()), 0)
            self.assertEqual(model(values).shape, (5, 3))

    def test_legacy_network_keeps_state_shape(self):
        model = PriceNetwork("RNN", 16, {"hidden_size": 32, "num_layers": 1, "dropout": 0., "pooling": "last"}, 3)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1699)
        self.assertFalse(any(key.startswith("head.") for key in model.state_dict()))

    def test_objective_weights_horizons_equally_and_excludes_small_groups(self):
        frame = pd.DataFrame({"horizon": [5, 5, 63, 252, 252], "relative_rmse": [1., 1., 2., 3., .01],
                              "eligible_for_summary": [True, True, True, True, False]})
        self.assertAlmostEqual(objective_score(frame), 2.)
        with self.assertRaises(ValueError):
            objective_score(frame.assign(eligible_for_summary=False))

    def test_overfit_signal_requires_train_improvement_and_inner_deterioration(self):
        curve = [{"epoch": 1, "fit_mse_scaled": 1., "inner_mse_scaled": 1., "inner_relative_rmse": 1.},
                 {"epoch": 2, "fit_mse_scaled": .8, "inner_mse_scaled": 1.4, "inner_relative_rmse": 1.2}]
        self.assertTrue(curve_diagnostics(curve, 1)["overfit_signal"])
        curve[-1]["fit_mse_scaled"] = 1.1
        self.assertFalse(curve_diagnostics(curve, 1)["overfit_signal"])


if __name__ == "__main__":
    unittest.main()
