"""Tablas de observaciones y de disponibilidad; no imputa etiquetas ni entrena."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_data import ROOT, SERIES_FRED


def read_source(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True, na_values=".")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"Fechas desordenadas o duplicadas: {path.name}")
    return frame.apply(pd.to_numeric, errors="raise")


def validate_history(history, series):
    frame = history.rename(columns={"period_start_date": "observation_date", series: "value",
                                    "realtime_start_date": "available_from", "realtime_end_date": "valid_until"}).copy()
    frame["observation_date"] = pd.to_datetime(frame.observation_date)
    frame["available_from"] = pd.to_datetime(frame.available_from)
    frame["valid_until"] = pd.to_datetime(frame.valid_until, errors="coerce")
    if frame.duplicated(["observation_date", "available_from"]).any():
        raise ValueError(f"Versiones duplicadas de {series}")
    if (frame.valid_until.notna() & (frame.valid_until < frame.available_from)).any():
        raise ValueError(f"Vigencia negativa en {series}")
    frame = frame.sort_values(["observation_date", "available_from"])
    following = frame.groupby("observation_date").available_from.shift(-1)
    overlaps = following.notna() & (frame.valid_until.isna() | (frame.valid_until >= following))
    if overlaps.any():
        raise ValueError(f"Versiones con vigencia solapada en {series}")
    if np.isinf(frame.value.dropna().to_numpy()).any():
        raise ValueError(f"Valores no finitos en {series}")
    return frame


def replay_history(history, calendar):
    """Última observación vigente en cada fecha, usando solo versiones disponibles.

    Una retirada (revisión a NaN) caduca la versión anterior; no se propaga una
    versión retirada. Se retrocede a la observación vigente más reciente.
    """
    events = []
    records = []
    for row in history.itertuples(index=False):
        if pd.isna(row.value):
            continue
        activation = max(row.observation_date, row.available_from)
        expiry = row.valid_until if pd.notna(row.valid_until) else pd.Timestamp.max
        if activation > expiry:
            continue
        record = (float(row.value), row.observation_date, row.available_from, expiry)
        records.append(record)
        events.append((activation, len(records) - 1))
    events.sort()
    position, heap, rows = 0, [], []
    for date in calendar:
        while position < len(events) and events[position][0] <= date:
            _, index = events[position]
            _, observed, available, _ = records[index]
            heapq.heappush(heap, (-observed.value, -available.value, index))
            position += 1
        while heap and records[heap[0][2]][3] < date:
            heapq.heappop(heap)
        if not heap:
            rows.append((np.nan, pd.NaT, pd.NaT))
        else:
            value, observed, available, _ = records[heap[0][2]]
            rows.append((value, observed, available))
    result = pd.DataFrame(rows, columns=["value", "observation_date", "available_from"], index=calendar)
    result["age_days"] = (pd.Series(calendar, index=calendar) - result.observation_date).dt.days
    return result


def yahoo_available(frame, calendar):
    """Convención conservadora de uso: desde el siguiente día de lunes a viernes.

    Es un supuesto diario explícito, no un archivo de versiones ni de timestamps.
    """
    values = frame.copy()
    values["observation_date"] = values.index
    values["available_from"] = values.index + pd.offsets.BDay(1)
    values = values.sort_values("available_from")
    # Si una fuente excepcionalmente trae fin de semana, varias filas pueden
    # tener la misma disponibilidad. Conservar la observación más reciente.
    values = values.drop_duplicates("available_from", keep="last")
    result = pd.merge_asof(pd.DataFrame({"Date": calendar}), values.reset_index(drop=True),
                           left_on="Date", right_on="available_from", direction="backward").set_index("Date")
    result["age_days"] = (pd.Series(calendar, index=result.index) - result.observation_date).dt.days
    return result


def build(start="2000-01-01", end="2026-11-14", output=None):
    output = Path(output) if output else ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw"
    calendar = pd.bdate_range(start, end, name="Date")
    observations = pd.DataFrame(index=calendar)
    known = pd.DataFrame(index=calendar)
    trace = pd.DataFrame(index=calendar)
    coverage, archive_rows, inputs = [], [], {}
    for series, name in SERIES_FRED.items():
        path = raw / f"fred_{series}.csv"
        original = read_source(path).iloc[:, 0]
        observations[name] = original.reindex(calendar)
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive = ROOT / "data/tables" / f"alfred_{series}_realtime.csv"
        history = validate_history(pd.read_csv(archive, na_values="."), series)
        replay = replay_history(history, calendar)
        known[f"{name}_conocido"] = replay.value
        for column in ["observation_date", "available_from", "age_days"]:
            trace[f"{name}_{column}"] = replay[column]
        metadata = json.loads((raw / "alfred" / f"{series}_realtime.json").read_text(encoding="utf-8"))
        for chunk in metadata["chunks"]:
            zip_path = raw / "alfred" / chunk["file"]
            actual_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
            if actual_hash != chunk["sha256"]:
                raise ValueError(f"El archivo ALFRED ha cambiado: {zip_path.name}")
            inputs[str(zip_path.relative_to(ROOT))] = actual_hash
        inputs[str(archive.relative_to(ROOT))] = hashlib.sha256(archive.read_bytes()).hexdigest()
        current = history.loc[(history.available_from <= pd.Timestamp(end)) &
                              (history.valid_until.isna() | (history.valid_until >= pd.Timestamp(end)))]
        latest = current.set_index("observation_date").value.reindex(calendar)
        value_mismatches = int(((observations[name] - latest).abs() > 1e-8).sum())
        missing_mismatches = int((observations[name].isna() != latest.isna()).sum())
        first_archive = metadata["first_listed_vintage"]
        later = history.loc[history.observation_date >= pd.Timestamp(first_archive)]
        releases = later.loc[later.value.notna()].groupby("observation_date").available_from.min()
        lags = (releases - pd.Series(releases.index, index=releases.index)).dt.days
        archive_rows.append({"variable": name, "series": series, "first_archived_vintage": first_archive,
                             "first_usable_date_in_panel": str(replay.value.first_valid_index().date()),
                             "version_records": len(history), "median_first_release_lag_calendar_days": float(lags.median()),
                             "p95_first_release_lag_calendar_days": float(lags.quantile(.95)),
                             "latest_value_differences_vs_fred": value_mismatches,
                             "latest_missing_differences_vs_fred": missing_mismatches})
    for path in sorted(raw.glob("yahoo_*.csv")):
        original = read_source(path)
        if original.isna().any(axis=None):
            raise ValueError(f"Yahoo contiene filas parciales; revisarlas antes de alinear: {path.name}")
        observations = observations.join(original.reindex(calendar))
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        replay = yahoo_available(original, calendar)
        for column in original:
            known[f"{column}_conocido"] = replay[column]
        name = path.stem.removeprefix("yahoo_")
        for column in ["observation_date", "available_from", "age_days"]:
            trace[f"yahoo_{name}_{column}"] = replay[column]
    for column in observations:
        series = observations[column]
        first, last = series.first_valid_index(), series.last_valid_index()
        coverage.append({"variable": column, "observed_rows": int(series.notna().sum()),
                         "first_observation": str(first.date()) if first is not None else None,
                         "last_observation": str(last.date()) if last is not None else None,
                         "missing_before_series": int(series.loc[series.index < first].isna().sum()) if first is not None else len(series),
                         "missing_inside_series": int(series.loc[first:last].isna().sum()) if first is not None else 0,
                         "missing_after_series": int(series.loc[series.index > last].isna().sum()) if last is not None else 0})
    # Validación exhaustiva de la procedencia temporal de cada entrada disponible.
    for column in trace.columns:
        if column.endswith(("observation_date", "available_from")):
            if (trace[column].dropna() > trace[column].dropna().index).any():
                raise ValueError(f"Dato futuro en el panel: {column}")
    for column in observations:
        if np.isinf(observations[column].dropna().to_numpy()).any():
            raise ValueError(f"Valor infinito en {column}")
    observations.to_csv(output / "observaciones_diarias.csv")
    observations.notna().to_csv(output / "mascara_observaciones.csv")
    known.to_csv(output / "informacion_disponible.csv")
    trace.to_csv(output / "procedencia_temporal.csv", date_format="%Y-%m-%d")
    pd.DataFrame(coverage).to_csv(output / "cobertura.csv", index=False)
    pd.DataFrame(archive_rows).to_csv(output / "cobertura_alfred.csv", index=False)
    complete = known.notna().all(axis=1)
    profile = {"start": str(calendar[0].date()), "end": str(calendar[-1].date()),
               "rows": len(calendar), "observed_variables": len(observations.columns),
               "observed_brent_prices": int(observations.brent.notna().sum()),
               "first_row_with_all_known_features": str(complete[complete].index[0].date()) if complete.any() else None,
               "rows_with_all_known_features": int(complete.sum()),
               "target_imputed": False, "scalers_fitted": False, "models_trained": False,
               "availability_convention": "ALFRED daily vintage; Yahoo previous observed close usable next weekday (assumption)",
               "archives": archive_rows, "source_sha256": inputs,
               "limitations": ["Antes del inicio de cada archivo ALFRED no se reconstruye su publicación histórica.",
                               "Yahoo no aporta aquí un archivo de revisiones ni timestamps históricos de publicación.",
                               "La cesta OPEP sigue pendiente de descarga (HTTP 403).",
                               "El calendario incluye lunes a viernes; no clasifica automáticamente cada hueco como festivo."]}
    for filename in ["observaciones_diarias.csv", "informacion_disponible.csv", "procedencia_temporal.csv"]:
        profile.setdefault("output_sha256", {})[filename] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    (output / "perfil_datos.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Tablas: {output}\n{len(calendar):,} días de calendario, {len(observations.columns)} variables, "
          f"{profile['observed_brent_prices']:,} precios Brent observados.")
    print(pd.DataFrame(archive_rows).to_string(index=False))
    return observations, known, trace, profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2026-11-14")
    args = parser.parse_args()
    build(args.start, args.end)
