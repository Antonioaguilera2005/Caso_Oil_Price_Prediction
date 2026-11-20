"""Ampliación hasta 2022 y señales causales; no cambia los experimentos congelados."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pandas as pd

from scripts.download_data import ROOT


DIRECTORY = ROOT / "data/development_2022"


def read_panel_before(path, stop):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Date"] >= stop:
                break
            rows.append(row)
    frame = pd.DataFrame(rows).set_index("Date")
    frame.index = pd.to_datetime(frame.index)
    return frame.apply(pd.to_numeric, errors="raise")


def causal_signals(frame):
    """Todas las ventanas acaban en el origen. WTI admite precios negativos."""
    brent = frame.brent_conocido
    if (brent.dropna() <= 0).any():
        raise ValueError("Brent debe ser positivo para sus logaritmos")
    log_brent = np.log(brent)
    returns = log_brent.diff()
    out = pd.DataFrame({"brent_conocido": brent}, index=frame.index)
    for days in [1, 5, 21, 63]:
        out[f"brent_change_{days}"] = log_brent.diff(days)
    for days in [21, 63]:
        out[f"brent_volatility_{days}"] = returns.rolling(days).std()
    for days in [63, 252]:
        out[f"brent_ma_distance_{days}"] = brent / brent.rolling(days).mean() - 1
    out["brent_drawdown_252"] = brent / brent.rolling(252).max() - 1
    out["wti_asinh_change"] = np.arcsinh(frame.wti_conocido / 20.).diff()
    for name, column in [("gold", "oro_conocido"), ("copper", "cobre_conocido"), ("usd", "indice_dolar_conocido")]:
        if (frame[column].dropna() <= 0).any():
            raise ValueError(f"Precio no positivo: {column}")
        out[f"{name}_change"] = np.log(frame[column]).diff()
    out["log_vix"] = np.log(frame.vix_conocido)
    out["brent_basis"] = frame.brent_fut_Close_conocido / brent - 1
    return out.replace([np.inf, -np.inf], np.nan)


def load_extended(variant):
    rules = json.loads((ROOT / "config/validation_2021_2022.json").read_text(encoding="utf-8"))
    manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((ROOT / "config/validation_2021_2022.json").read_bytes()).hexdigest() != manifest["rules_sha256"]:
        raise ValueError("Cambió el plan; preparar otro conjunto antes de entrenar")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cambió el archivo {name}")
    frame = pd.read_csv(DIRECTORY / f"{variant}.csv", index_col="Date", parse_dates=True)
    history = pd.read_csv(DIRECTORY / "target_vintages.csv")
    stop = pd.Timestamp(rules["development_end_exclusive"])
    if (frame.index >= stop).any() or (pd.to_datetime(history.realtime_start_date) >= stop).any():
        raise ValueError("Los archivos exportados alcanzan el test")
    if (pd.to_datetime(history.period_start_date) >= stop).any() or frame.isna().any(axis=None):
        raise ValueError("Etiquetas fuera de desarrollo o entradas incompletas")
    protocol = {**rules, "start": str(frame.index[0].date()), "feature_columns": list(frame.columns)}
    return frame, history, protocol, rules, manifest
