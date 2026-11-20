"""Métricas de precios en unidades originales y R² con referencia a su media."""
import numpy as np


def price_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError("Se requieren vectores no vacíos del mismo tamaño")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Las métricas requieren observaciones y predicciones finitas")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / total_variation) if total_variation > 0 else np.nan
    return {"n": len(actual), "mae": float(np.abs(error).mean()), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": r2}
