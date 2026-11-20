"""Crear archivos de desarrollo separados antes de entrenar el piloto."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from scripts.download_data import ROOT


def prepare():
    protocol_path = ROOT / "config/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    first = pd.Timestamp(protocol["start"])
    stop = pd.Timestamp(protocol["development_end_exclusive"])
    output = ROOT / "data/pilot"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/informacion_disponible.csv"
    features = pd.read_csv(source, index_col="Date", parse_dates=True)
    features = features.loc[(features.index >= first) & (features.index < stop), protocol["feature_columns"]].copy()
    expected = pd.bdate_range(first, stop - pd.Timedelta(days=1), name="Date")
    if not features.index.equals(expected) or features.isna().any(axis=None):
        raise ValueError("El desarrollo no tiene la cobertura continua declarada")
    features.to_csv(output / "development_inputs.csv")

    history_path = ROOT / "data/tables/alfred_DCOILBRENTEU_realtime.csv"
    history = pd.read_csv(history_path, na_values=".")
    observed = pd.to_datetime(history.period_start_date)
    available = pd.to_datetime(history.realtime_start_date)
    history = history.loc[(observed >= first) & (observed < stop) & (available < stop)].copy()
    # No exportar al piloto fechas de futuras revisiones que aún no habían ocurrido.
    end = pd.to_datetime(history.realtime_end_date, errors="coerce")
    history.loc[end >= stop, "realtime_end_date"] = None
    history.to_csv(output / "development_target_vintages.csv", index=False)
    metadata = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "input_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target_source_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "first_input_date": str(features.index[0].date()),
        "last_input_date": str(features.index[-1].date()),
        "input_rows": len(features), "features": list(features.columns),
        "calibration_exported": False, "test_exported": False,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in ["development_inputs.csv", "development_target_vintages.csv"]},
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Desarrollo preparado: {len(features)} filas, {len(features.columns)} variables, "
          f"{metadata['first_input_date']} a {metadata['last_input_date']}")
    print("Calibración y test no se han exportado a los archivos del piloto.")
    return metadata


if __name__ == "__main__":
    prepare()
