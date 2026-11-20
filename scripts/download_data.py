"""Descarga FRED (obligatorio), Yahoo y OPEP; conserva los datos originales.

Los huecos del objetivo Brent se mantienen: nunca se inventan etiquetas.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES_FRED = {
    "DCOILBRENTEU": "brent", "DCOILWTICO": "wti", "DHHNGSP": "gas_natural",
    "ECBDFR": "tipo_bce", "DEXUSEU": "usd_por_eur", "DEXCHUS": "cny_por_usd",
    "VIXCLS": "vix",
}
OTROS_YAHOO = {"HG=F": "cobre", "GC=F": "oro", "DX-Y.NYB": "indice_dolar"}
OPEC_URL = "https://www.opec.org/basket/basketDayArchives.xml"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Descarga incompleta")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def download(start="2000-01-01", end=None, offline=False) -> dict:
    end = end or datetime.now(timezone.utc).date().isoformat()
    directory = ROOT / "data"
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    report = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "requested_start": start, "requested_end": end, "offline": offline,
              "sources": {}, "warnings": []}
    frames = []
    for code, name in SERIES_FRED.items():
        path = raw / f"fred_{code}.csv"
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}&coed={end}"
        print(f"FRED: {code}", flush=True)
        if not offline:
            content = fetch(url)
            check = pd.read_csv(io.BytesIO(content), na_values=".")
            if code not in check or check[code].notna().sum() == 0:
                raise ValueError(f"Respuesta FRED vacía o inválida: {code}")
            path.write_bytes(content)
        frame = pd.read_csv(path, index_col=0, parse_dates=True, na_values=".")
        frame = normalize(frame.iloc[:, :1].rename(columns={frame.columns[0]: name})).loc[start:end]
        report["sources"][name] = {
            "url": url, "observations": int(frame[name].notna().sum()),
            "last_observation": str(frame[name].last_valid_index().date()),
        }
        frames.append(frame)

    import yfinance as yf
    tickers = [("BZ=F", "brent_fut", True), ("CL=F", "wti_fut", True)]
    tickers += [(ticker, name, False) for ticker, name in OTROS_YAHOO.items()]
    yf.set_tz_cache_location(str(ROOT / ".cache" / "yfinance"))
    for ticker, name, ohlcv in tickers:
        path = raw / f"yahoo_{name}.csv"
        try:
            print(f"Yahoo: {ticker}", flush=True)
            if offline:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
                frame = yf.download(ticker, start=start, end=stop, auto_adjust=False,
                                    progress=False, threads=False, timeout=30)
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                if frame.empty or frame["Close"].notna().sum() == 0:
                    raise ValueError("Yahoo no devolvió cotizaciones")
                if ohlcv:
                    frame = frame[["Open", "High", "Low", "Close", "Volume"]].add_prefix(f"{name}_")
                else:
                    frame = frame[["Close"]].rename(columns={"Close": name})
                frame = normalize(frame)
                frame.to_csv(path, index_label="Date")
            frame = normalize(frame).loc[start:end]
            frames.append(frame)
            report["sources"][name] = {"ticker": ticker, "observations": len(frame),
                                        "last_observation": str(frame.index.max().date())}
        except Exception as error:
            message = f"{name}: {type(error).__name__}: {error}"
            print(f"Aviso: {message}", flush=True)
            report["warnings"].append(message)

    try:
        path = raw / "opec.xml"
        content = path.read_bytes() if offline else fetch(OPEC_URL)
        tree = ET.fromstring(content)
        rows = []
        for element in tree.iter():
            attrs = {key.lower(): value for key, value in element.attrib.items()}
            if "date" in attrs and "value" in attrs:
                rows.append((attrs["date"], float(attrs["value"])))
        if not rows:
            raise ValueError("El XML no contiene observaciones diarias reconocibles")
        frame = pd.DataFrame(rows, columns=["Date", "cesta_opep"]).set_index("Date")
        frame = normalize(frame).loc[start:end]
        if frame.empty:
            raise ValueError("Sin observaciones OPEP en el periodo solicitado")
        path.write_bytes(content)
        frames.append(frame)
        report["sources"]["cesta_opep"] = {"url": OPEC_URL, "observations": len(frame)}
    except Exception as error:
        message = f"Cesta OPEP no disponible: {type(error).__name__}: {error}"
        print(f"Aviso: {message}", flush=True)
        report["warnings"].append(message)

    original = pd.concat(frames, axis=1, sort=True).sort_index().loc[start:end]
    observed = original["brent"].dropna()
    dataset = original.reindex(pd.bdate_range(observed.index.min(), original.index.max()))
    predictors = dataset.columns.difference(["brent"])
    # Solo propagación hacia delante de indicadores; el objetivo conserva NaN.
    dataset[predictors] = dataset[predictors].ffill()
    dataset.index.name = "Date"
    path = directory / "petroleo_dataset.csv"
    temporary = path.with_suffix(".csv.tmp")
    dataset.to_csv(temporary)
    temporary.replace(path)
    report.update({"rows": len(dataset), "columns": list(dataset.columns),
                   "first_date": str(dataset.index.min().date()),
                   "last_date": str(dataset.index.max().date()),
                   "last_observed_brent": str(observed.index.max().date()),
                   "missing": {key: int(value) for key, value in dataset.isna().sum().items()},
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory / "download_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {path}\n{len(dataset):,} filas, {len(dataset.columns)} variables. "
          f"Último Brent observado: {report['last_observed_brent']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--offline", action="store_true", help="Reconstruir desde data/raw, sin conexión")
    args = parser.parse_args()
    download(args.start, args.end, args.offline)
