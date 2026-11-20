"""Descargar los periodos de vigencia desde el formulario público de ALFRED.

Se conserva el ZIP original. Las fechas anteriores al inicio del archivo no se
presentan como fechas de publicación históricas reconstruidas.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def download(series="DCOILBRENTEU", start="2000-01-01", end="2026-11-14", refresh=False):
    directory = ROOT / "data/raw/alfred"
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
    metadata_path = directory / f"{series}_realtime.json"
    cached = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    cache_matches = (cached is not None and cached["requested_start"] == start
                     and cached["requested_end"] == end
                     and all((directory / part["file"]).exists() for part in cached["chunks"]))
    if not cache_matches or refresh:
        with urlopen(url, timeout=30) as response:
            html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        selector = soup.find("select", attrs={"name": "form[selected_vintage_dates][]"})
        if selector is None:
            raise ValueError("No se encuentra el selector público de versiones de ALFRED")
        dates = [option.get("value") for option in selector.find_all("option") if option.get("value")]
        if not dates:
            raise ValueError("ALFRED no devuelve fechas de versiones")
        last_observation = soup.find("input", attrs={"name": "form[obs_end_date]"}).get("value")
        observation_end = min(end, last_observation)
        print(f"Descargando {series}: {len(dates)} versiones disponibles...", flush=True)
        dates = sorted(date for date in dates if date <= end)
        chunks = []
        # El formulario admite 450 versiones diarias por descarga. Dejar una
        # posición para la fecha de consulta final, incluso si aún no es vintage.
        for offset in range(0, len(dates), 449):
            selected = dates[offset:offset + 449]
            fields = [("form[units]", "lin"), ("form[obs_start_date]", start),
                      ("form[obs_end_date]", observation_end), ("form[entered_vintage_dates]", end),
                      ("form[file_type]", "1"), ("form[file_format]", "csv"),
                      ("form[download_data]", "")]
            fields += [("form[selected_vintage_dates][]", date) for date in selected]
            request = Request(url, data=urlencode(fields).encode(), method="POST")
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if not zipfile.is_zipfile(io.BytesIO(content)):
                response_soup = BeautifulSoup(content, "html.parser")
                errors = response_soup.select("[class*='error'], [class*='invalid'], [class*='alert']")
                detail = " ".join(item.get_text(" ", strip=True) for item in errors)
                if not detail:
                    for node in response_soup.find_all(["script", "style", "option", "header", "footer"]):
                        node.decompose()
                    detail = response_soup.get_text(" ", strip=True)[-1800:]
                raise ValueError(f"El formulario no devolvió un ZIP de datos: {detail[:1800]}")
            path = directory / f"{series}_realtime_part{len(chunks) + 1:03d}.zip"
            path.write_bytes(content)
            chunks.append({"file": path.name, "first_selected_vintage": min(selected),
                           "last_selected_vintage": max(selected),
                           "sha256": hashlib.sha256(content).hexdigest()})
            print(f"  Bloque {len(chunks)} guardado ({len(content):,} bytes)", flush=True)
        metadata = {"series": series, "url": url, "first_listed_vintage": min(dates),
                    "last_listed_vintage": max(dates), "listed_vintages": len(dates),
                    "requested_start": start, "requested_end": end, "observation_end": observation_end,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "chunks": chunks, "output_format": "real_time_periods"}
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tables = []
    for chunk in metadata["chunks"]:
        path = directory / chunk["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
            raise ValueError(f"El ZIP no coincide con su hash: {path.name}")
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist()
                       if "readme" not in name.lower() and name.lower().endswith((".csv", ".txt", ".tsv"))]
            if len(members) != 1:
                raise ValueError(f"Contenido del ZIP inesperado: {archive.namelist()}")
            payload = archive.read(members[0])
        tables.append(pd.read_csv(io.BytesIO(payload), sep=None, engine="python", na_values="."))
    # Solo eliminar registros idénticos que aparecen en ambos bloques.
    table = pd.concat(tables, ignore_index=True).drop_duplicates().reset_index(drop=True)
    required = {"period_start_date", series, "realtime_start_date", "realtime_end_date"}
    if not required.issubset(table.columns):
        raise ValueError(f"El ZIP no tiene el formato de vigencia esperado: {list(table.columns)}")
    table = table.sort_values(["period_start_date", "realtime_start_date"]).reset_index(drop=True)
    destination = ROOT / "data/tables" / f"alfred_{series}_realtime.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    print(f"Guardado: {destination}\nFilas: {len(table)}; columnas: {list(table.columns)}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return table, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="DCOILBRENTEU")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2026-11-14")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    download(args.series, args.start, args.end, args.refresh)
