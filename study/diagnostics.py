"""Diagnósticos de error; nunca ajustan el modelo ni consultan el test."""
import numpy as np


def block_ratio_interval(model_squared_error, naive_squared_error, block, repetitions=1000, seed=42):
    model, naive = np.asarray(model_squared_error, dtype=float), np.asarray(naive_squared_error, dtype=float)
    if model.ndim != 1 or model.shape != naive.shape or not len(model):
        raise ValueError("Vectores de error incompatibles")
    if not np.isfinite(model).all() or not np.isfinite(naive).all() or min(model.min(), naive.min()) < 0 or naive.mean() <= 0:
        raise ValueError("Errores cuadrados inválidos")
    n = len(model)
    block = min(n, max(1, int(block)))
    if n / block < 2:
        return {"ratio": float(np.sqrt(model.mean() / naive.mean())), "ratio_low": None, "ratio_high": None,
                "block": block, "effective_blocks_approx": n / block,
                "bootstrap_repetitions": 0, "valid_repetitions": 0}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block))))
    ids = ((starts[:, :, None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    denominator = naive[ids].mean(axis=1)
    ratios = np.sqrt(model[ids].mean(axis=1)[denominator > 0] / denominator[denominator > 0])
    low, high = np.quantile(ratios, [.025, .975])
    return {"ratio": float(np.sqrt(model.mean() / naive.mean())), "ratio_low": float(low),
            "ratio_high": float(high), "block": block, "effective_blocks_approx": n / block,
            "bootstrap_repetitions": repetitions, "valid_repetitions": len(ratios)}
