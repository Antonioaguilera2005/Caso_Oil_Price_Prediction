"""Cross-check browser signals and price forecasts against the original PyTorch model."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from scripts.download_data import ROOT
from scripts.evaluate_recovered_test import predict_checkpoint
from study.extended import causal_signals


def main():
    selected = json.loads((ROOT / "config/delivery_model.json").read_text(encoding="utf-8"))
    runtime = ROOT / ".cache/node-runtime.json"
    executable = shutil.which("node") or (json.loads(runtime.read_text())["node"] if runtime.exists() else None)
    if not executable:
        raise RuntimeError("Node.js is required only for this browser/PyTorch conformance check")
    panel = pd.read_csv(ROOT / "data/processed/informacion_disponible.csv", index_col="Date", parse_dates=True)
    features = causal_signals(panel).dropna()
    dates = ["2016-06-15", "2020-04-28", "2023-01-03", "2024-12-27", str(features.index[-1].date())]
    item = {**selected, "settings": json.loads((ROOT / selected["settings"]).read_text(encoding="utf-8"))}
    predicted = predict_checkpoint(item, features, features.index.get_indexer(pd.to_datetime(dates)))
    reference = [{"date": date, "signals": features.loc[date].to_numpy().tolist(), "prediction": predicted[i].tolist()}
                 for i, date in enumerate(dates)]
    module = (ROOT / "app_site/engine.js").as_uri()
    data_path = str(ROOT / "app_site/data/app-data.json")
    model_path = str(ROOT / "app_site/data/model.json")
    script = """import fs from 'node:fs';
import {prepareSeries,predictAt} from MODULE;
const reference = JSON.parse(fs.readFileSync(0,'utf8'));
const data = JSON.parse(fs.readFileSync(DATA_PATH));
const model = JSON.parse(fs.readFileSync(MODEL_PATH));
const prepared = prepareSeries(data);
let signal=0,price=0;
for(const row of reference){
const actual=prepared.signals[prepared.dates.indexOf(row.date)];
signal=Math.max(signal,...actual.map((v,i)=>Math.abs(v-row.signals[i])));
const output=predictAt(model,prepared,row.date).predictions;
price=Math.max(price,...output.map((v,i)=>Math.abs(v.price-row.prediction[i])));
}
console.log(JSON.stringify({origins:reference.length,maximum_signal_difference:signal,maximum_price_difference_usd:price}));
if(signal>1e-9||price>0.0001)process.exitCode=1;
""".replace("MODULE", json.dumps(module)).replace("DATA_PATH", json.dumps(data_path)).replace("MODEL_PATH", json.dumps(model_path))
    result = subprocess.run([executable, "--input-type=module", "-e", script], input=json.dumps(reference),
                            capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT)
    verified = {**json.loads(result.stdout), "chosen_model": selected["id"], "models_retrained": 0,
                "tolerance_usd": 0.0001, "dates": dates}
    (ROOT / "app_site/data/inference-verification.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(json.dumps(verified, ensure_ascii=False))


if __name__ == "__main__":
    main()
