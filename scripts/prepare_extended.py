"""Exporta exclusivamente datos anteriores a 2023 para la nueva validación."""
import csv
import hashlib
import json

import pandas as pd

from scripts.download_data import ROOT
from study.extended import DIRECTORY, read_panel_before, causal_signals
from study.search import write_json


def prepare():
    rule_path = ROOT / "config/validation_2021_2022.json"
    rules = json.loads(rule_path.read_text(encoding="utf-8"))
    stop = rules["development_end_exclusive"]
    if stop != rules["test_start"] or stop > "2023-01-01":
        raise ValueError("El test 2023–2024 no puede incorporarse a desarrollo")
    source = ROOT / "data/processed/informacion_disponible.csv"
    panel = read_panel_before(source, stop)
    old = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))
    levels = panel[old["feature_columns"]]
    signals = causal_signals(panel)
    valid = levels.notna().all(axis=1) & signals.notna().all(axis=1) & (panel.index >= rules["start"])
    first = panel.index[valid][0]
    # Ambas variantes conservan exactamente el mismo calendario e histórico.
    levels, signals = levels.loc[first:], signals.loc[first:]
    expected = pd.bdate_range(first, pd.Timestamp(stop) - pd.Timedelta(days=1), name="Date")
    for frame in [levels, signals]:
        if not frame.index.equals(expected) or frame.isna().any(axis=None):
            raise ValueError("Revisar cobertura antes de entrenar")
    history_path = ROOT / "data/tables/alfred_DCOILBRENTEU_realtime.csv"
    rows = []
    with history_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(first.date()) <= row["period_start_date"] < stop and row["realtime_start_date"] < stop:
                if row["realtime_end_date"] >= stop:
                    row["realtime_end_date"] = ""
                rows.append(row)
    history = pd.DataFrame(rows)
    history["DCOILBRENTEU"] = pd.to_numeric(history.DCOILBRENTEU.replace(".", None))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    levels.to_csv(DIRECTORY / "levels.csv")
    signals.to_csv(DIRECTORY / "signals.csv")
    history.to_csv(DIRECTORY / "target_vintages.csv", index=False)
    write_json(DIRECTORY / "manifest.json", {
        "rules_sha256": hashlib.sha256(rule_path.read_bytes()).hexdigest(),
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, history_path]},
        "first_input_date": str(first.date()), "last_input_date": str(expected[-1].date()),
        "rows": len(levels), "test_exported": False, "test_start": stop,
        "files": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() for name in
                  ["levels.csv", "signals.csv", "target_vintages.csv"]}})
    print(f"Exportadas {len(levels)} filas por variante: {first.date()}–{expected[-1].date()}; test no exportado")


if __name__ == "__main__":
    prepare()
