"""Preparación causal de ventanas y etiquetas publicadas por fold."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from scripts.download_data import ROOT


def load_development():
    protocol_path = ROOT / "config/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    directory = ROOT / "data/pilot"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("El protocolo cambió: vuelve a preparar los archivos de desarrollo")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"El archivo del piloto ha cambiado: {name}")
    features = pd.read_csv(directory / "development_inputs.csv", index_col="Date", parse_dates=True)
    history = pd.read_csv(directory / "development_target_vintages.csv", na_values=".")
    cutoff = pd.Timestamp(protocol["development_end_exclusive"])
    if (features.index >= cutoff).any() or (pd.to_datetime(history.realtime_start_date) >= cutoff).any():
        raise ValueError("El piloto no admite datos posteriores al periodo de desarrollo")
    if list(features.columns) != protocol["feature_columns"] or features.isna().any(axis=None):
        raise ValueError("Variables o cobertura incompatibles con el protocolo")
    return features, history, protocol, manifest


def target_snapshot(history, cutoff_exclusive, dates):
    cutoff = pd.Timestamp(cutoff_exclusive)
    start = pd.to_datetime(history.realtime_start_date)
    end = pd.to_datetime(history.realtime_end_date, errors="coerce")
    # La última versión vigente el día anterior al corte exclusivo.
    selected = history.loc[(start < cutoff) & (end.isna() | (end >= cutoff - pd.Timedelta(days=1)))].copy()
    selected["period_start_date"] = pd.to_datetime(selected.period_start_date)
    if selected.period_start_date.duplicated().any():
        raise ValueError("Más de una versión vigente para una fecha objetivo")
    values = selected.set_index("period_start_date").DCOILBRENTEU
    return values.reindex(dates)


def samples(features, target, start, end, window, horizons):
    """Purga por fecha de cada destino y descarta etiquetas ausentes."""
    dates = features.index
    positions = np.arange(window - 1, len(features) - max(horizons), dtype=int)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    prices = target.to_numpy(dtype=float)
    base = features.brent_conocido.to_numpy(dtype=float)
    valid = (dates[positions] >= start) & (dates[positions] < end)
    for horizon in horizons:
        destinations = positions + horizon
        valid &= (dates[destinations] >= start) & (dates[destinations] < end)
        valid &= np.isfinite(prices[destinations])
    valid &= np.isfinite(base[positions])
    positions = positions[valid]
    labels = np.column_stack([prices[positions + horizon] for horizon in horizons])
    return positions, labels, base[positions]


def fit_scaler(features, cutoff_exclusive):
    frame = features.loc[features.index < pd.Timestamp(cutoff_exclusive)].copy()
    if frame.empty:
        raise ValueError("No hay entradas de entrenamiento")
    for column in frame:
        if "Volume" in column:
            if (frame[column] < 0).any():
                raise ValueError("Volumen negativo")
            frame[column] = np.log1p(frame[column])
    median = frame.median()
    scale = (frame.quantile(.75) - frame.quantile(.25)).replace(0, 1)
    return {"columns": list(frame.columns), "median": median.to_list(), "scale": scale.to_list(),
            "last_fitted_date": str(frame.index[-1].date()), "rows_fitted": len(frame)}


def transform(features, scaler):
    frame = features[scaler["columns"]].copy()
    for column in frame:
        if "Volume" in column:
            if (frame[column] < 0).any():
                raise ValueError("Volumen negativo")
            frame[column] = np.log1p(frame[column])
    values = (frame.to_numpy() - np.asarray(scaler["median"])) / np.asarray(scaler["scale"])
    if not np.isfinite(values).all():
        raise ValueError("Entradas no finitas; no se imputan automáticamente en el piloto")
    return values.astype("float32")
