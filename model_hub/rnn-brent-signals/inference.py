"""Load safe JSON weights and predict with the original PyTorch RNN.

No pickle, remote code execution, fitting, network calls or model updates.
Input rows must contain observations already known at each Date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

RAW_COLUMNS = ["brent_conocido", "wti_conocido", "vix_conocido", "oro_conocido",
               "cobre_conocido", "indice_dolar_conocido", "brent_fut_Close_conocido"]
WEIGHT_NAMES = {"input": "recurrent.weight_ih_l0", "recurrent": "recurrent.weight_hh_l0",
                "inputBias": "recurrent.bias_ih_l0", "recurrentBias": "recurrent.bias_hh_l0",
                "output": "output.weight", "outputBias": "output.bias"}


class BrentRNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.recurrent = nn.RNN(len(model["featureColumns"]), model["metadata"]["hiddenSize"],
                                num_layers=1, batch_first=True, nonlinearity="tanh")
        self.output = nn.Linear(model["metadata"]["hiddenSize"], len(model["horizons"]))
        weights = {WEIGHT_NAMES[name]: torch.tensor(value, dtype=torch.float32)
                   for name, value in model["weights"].items()}
        self.load_state_dict(weights, strict=True)
        self.eval()

    def forward(self, values):
        sequence, _ = self.recurrent(values)
        return self.output(sequence[:, -1])


def load_model(directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parent
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if (model["architecture"] != "RNN" or model["window"] != 60 or
            model["horizons"] != [5, 63, 252] or model["metadata"]["parameters"] != 1699):
        raise ValueError("Unexpected model configuration")
    return BrentRNN(model), model


def prepare_signals(frame):
    frame = frame.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
        frame = frame.set_index("Date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("A Date column or DatetimeIndex is required")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if frame.empty or not frame.index.equals(pd.bdate_range(frame.index[0], frame.index[-1], name=frame.index.name)):
        raise ValueError("Rows must include each Monday-Friday date; use already-known inputs on publication holidays")
    frame = frame[RAW_COLUMNS].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Infinite raw inputs are not allowed")
    for column in set(RAW_COLUMNS) - {"wti_conocido"}:
        if (frame[column].dropna() <= 0).any():
            raise ValueError(f"{column} must be positive")
    brent = frame.brent_conocido
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
        out[f"{name}_change"] = np.log(frame[column]).diff()
    out["log_vix"] = np.log(frame.vix_conocido)
    out["brent_basis"] = frame.brent_fut_Close_conocido / brent - 1
    return out.replace([np.inf, -np.inf], np.nan), frame


def predict_frame(frame, origin=None, directory=None):
    network, model = load_model(directory)
    signals, raw = prepare_signals(frame)
    origin = pd.Timestamp(origin) if origin else signals.index[-1]
    if origin not in signals.index:
        raise ValueError("Origin is not in the input data")
    window = signals.loc[:origin, model["featureColumns"]].tail(model["window"])
    if len(window) != model["window"] or window.isna().any(axis=None):
        raise ValueError("At least 60 complete signal rows plus 252-day raw history are required")
    scaled = (window.to_numpy() - np.array(model["scaler"]["median"])) / np.array(model["scaler"]["scale"])
    with torch.inference_mode():
        output = network(torch.tensor(scaled, dtype=torch.float32).unsqueeze(0))[0].numpy()
    base = float(raw.loc[origin, "brent_conocido"])
    changes = output.astype(np.float64) * np.array(model["scaler"]["targetScale"])
    result = []
    for index, horizon in enumerate(model["horizons"]):
        price = base + float(changes[index])
        width = model["intervalHalfWidths"][str(horizon)]
        result.append({"horizon": horizon, "target_date": str((origin + pd.offsets.BDay(horizon)).date()),
                       "price": price, "change": float(changes[index]), "lower_80": price - width,
                       "upper_80": price + width})
    return {"model_id": model["id"], "origin": str(origin.date()), "known_spot": base, "predictions": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="UTF-8 CSV with Date and the seven known-input columns")
    parser.add_argument("--origin", default=None, help="YYYY-MM-DD; defaults to the final input row")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, sep=None, engine="python", encoding="utf-8-sig")
    print(json.dumps(predict_frame(frame, args.origin), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
