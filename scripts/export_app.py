"""Export the chosen frozen RNN, public input snapshot and saved evaluation for HTML."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.download_data import ROOT
from study.data import target_snapshot

OUTPUT = ROOT / "app_site/data"
RAW_COLUMNS = ["brent_conocido", "wti_conocido", "vix_conocido", "oro_conocido",
               "cobre_conocido", "indice_dolar_conocido", "brent_fut_Close_conocido"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = read_json(ROOT / "config/delivery_model.json")
    folder = (ROOT / selected["checkpoint"]).parent
    scaler = read_json(ROOT / selected["scaler"])
    settings = read_json(ROOT / selected["settings"])
    state = torch.load(ROOT / selected["checkpoint"], map_location="cpu", weights_only=True)
    if selected["architecture"] != "RNN" or settings["num_layers"] != 1 or settings.get("head_size", 0) or settings["pooling"] != "last":
        raise ValueError("Browser export supports this chosen one-layer, last-state RNN only")
    source = ROOT / selected["evaluation"]
    plan = read_json(source / "plan.json")
    frozen = next(m for m in plan["models"] if m["architecture"] == "RNN" and m["variant"] == "signals")
    if frozen["checkpoint"] != selected["checkpoint"] or sha(ROOT / selected["checkpoint"]) != plan["files"][selected["checkpoint"]]:
        raise ValueError("Delivery weights differ from the evaluated checkpoint")
    parameters = sum(t.numel() for t in state.values())
    model = {"id": selected["id"], "architecture": "RNN", "window": selected["window"],
        "horizons": selected["horizons"], "featureColumns": scaler["columns"],
        "scaler": {"median": scaler["median"], "scale": scaler["scale"], "targetScale": scaler["target_scale"]},
        "weights": {name: state[key].tolist() for name, key in {
            "input": "recurrent.weight_ih_l0", "recurrent": "recurrent.weight_hh_l0",
            "inputBias": "recurrent.bias_ih_l0", "recurrentBias": "recurrent.bias_hh_l0",
            "output": "output.weight", "outputBias": "output.bias"}.items()},
        "intervalHalfWidths": frozen["interval_half_widths"],
        "metadata": {"name": "RNN con señales", "trainingEnd": scaler["last_fitted_date"],
            "validationStart": "2018-01-01", "validationEnd": "2022-12-30", "testStart": "2023-01-01", "testEnd": "2024-12-31",
            "selectedEpoch": frozen["selected_epoch"], "parameters": parameters, "hiddenSize": settings["hidden_size"],
            "seed": 42, "dropout": settings["dropout"], "weightDecay": settings["weight_decay"],
            "learningRate": settings["learning_rate"], "source": selected["checkpoint"], "checkpointSha256": sha(ROOT / selected["checkpoint"]),
            "limitations": ["Mejora al ingenuo spot a 5 días; a 63 y 252 días prácticamente empata.",
                "No supera al futuro conocido en los tres horizontes del bloque evaluado.",
                "Los R² de 63 y 252 días son negativos; los errores largos están solapados.",
                "Las bandas son empíricas y se calibraron con validación usada para seleccionar la época; no garantizan cobertura futura.",
                "Las entradas completas del modelo empiezan en 2012; el histórico de precios mostrado comienza en 2000."]}}
    save(OUTPUT / "model.json", model)
    panel = pd.read_csv(ROOT / "data/processed/informacion_disponible.csv")
    raw_rows = [[str(row[0]), *[None if pd.isna(x) else float(x) for x in row[1:]]]
                for row in panel[["Date", *RAW_COLUMNS]].itertuples(index=False, name=None)]
    dates = pd.DatetimeIndex(pd.to_datetime(panel.Date), name="Date")
    end = str((dates[-1] + pd.Timedelta(days=1)).date())
    history = pd.read_csv(ROOT / "data/tables/alfred_DCOILBRENTEU_realtime.csv", na_values=".")
    target = target_snapshot(history, end, dates).dropna()
    download = read_json(ROOT / "data/download_report.json")
    data = {"columns": ["Date", *RAW_COLUMNS], "rows": raw_rows,
        "observed": [{"date": str(date.date()), "price": float(price)} for date, price in target.items()],
        "events": read_json(ROOT / "config/app_events.json"),
        "snapshot": {"generatedAt": pd.Timestamp.now(tz="UTC").isoformat(), "downloadedAt": download["downloaded_at_utc"],
            "lastInputDate": str(dates[-1].date()), "lastObservedDate": str(target.index[-1].date()),
            "referenceDate": "2026-11-20", "source": "Copia local de FRED/ALFRED y Yahoo Finance; actualización manual mediante CSV"},
        "sources": [
            {"name": "EIA / FRED", "url": "https://fred.stlouisfed.org/series/DCOILBRENTEU", "description": "Brent spot diario publicado; puede llevar retraso respecto a la fecha de observación."},
            {"name": "ALFRED", "url": "https://alfred.stlouisfed.org/series?seid=DCOILBRENTEU", "description": "Versiones y fechas de disponibilidad histórica utilizadas en las entradas."},
            {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/quote/BZ=F/history/", "description": "Futuros del Brent, oro, cobre e índice del dólar como datos complementarios."}]}
    save(OUTPUT / "app-data.json", data)
    metrics = pd.read_csv(source / "metrics.csv")
    predictions = pd.read_csv(source / "predictions.csv")
    mask = metrics.architecture.eq("RNN") & metrics.variant.eq("signals")
    chosen = metrics.loc[mask & metrics.group.eq("all")].copy()
    prediction_mask = predictions.architecture.eq("RNN") & predictions.variant.eq("signals")
    validation = pd.read_csv(source / "validation_metrics.csv")
    evaluation = {"selected": chosen.to_dict(orient="records"),
        "predictions": predictions.loc[prediction_mask].to_dict(orient="records"),
        "comparison": metrics.loc[metrics.group.eq("all")].to_dict(orient="records"),
        "validation": validation.loc[validation.architecture.eq("RNN") & validation.variant.eq("signals")].to_dict(orient="records"),
        "sampleCounts": pd.read_csv(source / "sample_counts.csv").to_dict(orient="records")}
    save(OUTPUT / "evaluation.json", evaluation)
    # The chosen checkpoint and original evaluation remain untouched.
    save(OUTPUT / "export-manifest.json", {"selectedModel": selected, "checkpointSha256": sha(ROOT / selected["checkpoint"]),
        "evaluationCompleteSha256": sha(source / "complete.json"), "modelRetrained": False,
        "files": {name: sha(OUTPUT / name) for name in ["model.json", "app-data.json", "evaluation.json"]}})
    print(json.dumps({"model": selected["id"], "parameters": parameters, "input_rows": len(panel),
        "last_input": data["snapshot"]["lastInputDate"], "last_observed": data["snapshot"]["lastObservedDate"],
        "files": {p.name: p.stat().st_size for p in OUTPUT.glob("*.json")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
