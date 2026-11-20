"""Prepare the selected, already evaluated RNN for a Hugging Face model repo.

Only local files are written. No training, account access or upload takes place.
Weights are exported as safe, portable JSON rather than pickle.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "model_hub/rnn-brent-signals"
WEIGHT_NAMES = {"input": "recurrent.weight_ih_l0", "recurrent": "recurrent.weight_hh_l0",
                "inputBias": "recurrent.bias_ih_l0", "recurrentBias": "recurrent.bias_hh_l0",
                "output": "output.weight", "outputBias": "output.bias"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def model_card(model, metrics):
    rows = "\n".join(f"| {row['horizon']} | {row['n']} | {row['mae']:.4f} | {row['mse']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | {row['relative_rmse']:.4f} | {row['relative_rmse_vs_future']:.4f} |"
                     for row in metrics)
    return f'''---
language:
- es
library_name: pytorch
tags:
- time-series
- forecasting
- rnn
- brent
- educational
model-index:
- name: RNN Brent con señales
  results: []
---

# RNN Brent con señales

Modelo académico para estimar el precio Brent spot en USD/barril a 5, 63 y 252 días de lunes a viernes, incluidos festivos. Es el checkpoint original de la RNN con señales, semilla 42, seleccionado al cerrar un proceso con múltiples arquitecturas, configuraciones y experimentos. Publicar este paquete no entrena ni modifica sus pesos.

## Arquitectura y partición

- RNN tanh de una capa con 32 unidades; último estado y salida lineal de 3 componentes.
- 16 señales causales, ventana de 60 observaciones, 1.699 parámetros.
- Aprendizaje 0,001; época seleccionada 17; sin dropout ni penalización L2 en este checkpoint.
- Entrenamiento y ajuste del escalado hasta el 29 de diciembre de 2017.
- Validación de enero de 2018 a diciembre de 2022 para seleccionar época.
- Comparación histórica guardada de 2023–2024, con los pesos y escalas anteriores intactos.

Las entradas multivariables completas empiezan en 2012. El bloque 2023–2024 se había consultado previamente durante el proyecto; sus resultados no se presentan como una primera prueba ciega. `metrics.json` conserva las métricas del checkpoint elegido.

## Uso en Python

Tras descargar esta carpeta e instalar `requirements.txt`:

```python
import pandas as pd
from inference import predict_frame

inputs = pd.read_csv("informacion_disponible.csv")
result = predict_frame(inputs, origin="2024-12-27")
print(result)
```

También se puede ejecutar `python inference.py informacion_disponible.csv --origin 2024-12-27`.

El CSV requiere `Date` y las columnas `brent_conocido`, `wti_conocido`, `vix_conocido`, `oro_conocido`, `cobre_conocido`, `indice_dolar_conocido`, `brent_fut_Close_conocido`, ordenadas por fecha. Debe representar la información ya disponible en cada origen, con una fila por día de lunes a viernes. No se debe rellenar hacia atrás ni introducir precios que se conocieron después. Se necesitan al menos 252 filas para el calentamiento de las señales y una ventana posterior de 60 señales completas. WTI admite valores negativos; las demás entradas deben ser positivas.

`model.json` contiene los pesos exactos y el escalado, sin pickle. `inference.py` reconstruye explícitamente la red; este modelo personalizado no utiliza `AutoModel` ni tiene un proveedor de inferencia alojado configurado. `engine.js` ofrece la misma preparación e inferencia en el navegador y `package.json` habilita su importación como módulo ES.

## Preprocesamiento

Cambios logarítmicos de Brent a 1, 5, 21 y 63 días; volatilidades muestrales de 21 y 63 días; distancias a medias de 63 y 252 días; drawdown de 252 días; cambio de `asinh(WTI/20)`; cambios logarítmicos de oro, cobre y dólar; logaritmo del VIX y base futuro/spot. Todas las ventanas terminan en el origen.

Se resta la mediana y se divide por la escala guardada de cada señal. La RNN devuelve cambios normalizados; se multiplican por las escalas de objetivo guardadas y se añade el último Brent spot conocido. `preprocessing.json` documenta el contrato y `scaler.json` preserva el escalador original.

## Resultados guardados

| Horizonte | N | MAE | MSE | RMSE | R² | RMSE/ingenuo spot | RMSE/futuro conocido |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

Los cocientes inferiores a 1 indican menor RMSE que la referencia correspondiente, no rentabilidad. El modelo mejora al ingenuo spot a 5 días; a 63 y 252 días prácticamente empata. No supera al futuro conocido en ninguno de esos horizontes. Los R² de los dos horizontes largos son negativos y sus errores están solapados.

Las bandas simétricas del 80 % usan un cuantil empírico de los errores absolutos de validación, conjunto que también seleccionó la época. Son rangos descriptivos y no tienen una garantía formal de cobertura futura. La incertidumbre es especialmente amplia a 252 días.

## Procedencia y alcance

Pesos procedentes de `{model['metadata']['source']}`.

SHA-256 del checkpoint original: `{model['metadata']['checkpointSha256']}`.

El paquete contiene un modelo, su documentación y métricas; no redistribuye las tablas completas de mercado. Los datos del proyecto proceden de FRED/ALFRED y Yahoo Finance, con disponibilidad histórica documentada en el repositorio académico. Es un ejercicio educativo, sin estrategia de negociación ni evaluación de comisiones, liquidez o rentabilidad.

No se ha asignado una licencia de redistribución a este paquete. Antes de elegirla, debe revisarse la titularidad del código y las condiciones de las fuentes. La integridad de todos los archivos figura en `manifest.json`.
'''


def main():
    import torch

    selected = read(ROOT / "config/delivery_model.json")
    model = read(ROOT / "app_site/data/model.json")
    source = ROOT / selected["evaluation"]
    plan = read(source / "plan.json")
    for key in ["checkpoint", "settings", "scaler"]:
        relative = selected[key]
        if relative not in plan["files"] or sha(ROOT / relative) != plan["files"][relative]:
            raise ValueError(f"Selected {key} differs from the evaluated artifact")
    if (selected["id"] != model["id"] or selected["architecture"] != "RNN" or
            selected["variant"] != "signals" or selected["seed"] != 42 or
            selected["window"] != model["window"] or selected["horizons"] != model["horizons"] or
            model["metadata"]["checkpointSha256"] != sha(ROOT / selected["checkpoint"])):
        raise ValueError("Browser model is not the selected checkpoint")
    settings = read(ROOT / selected["settings"])
    scaler = read(ROOT / selected["scaler"])
    if (settings["num_layers"] != 1 or settings["hidden_size"] != 32 or
            settings["pooling"] != "last" or settings.get("head_size", 0) or
            model["featureColumns"] != scaler["columns"] or
            model["scaler"] != {"median": scaler["median"], "scale": scaler["scale"], "targetScale": scaler["target_scale"]}):
        raise ValueError("Unexpected architecture or preprocessing")
    weights = torch.load(ROOT / selected["checkpoint"], map_location="cpu", weights_only=True)
    if set(weights) != set(WEIGHT_NAMES.values()):
        raise ValueError("Unexpected checkpoint tensors")
    for short, name in WEIGHT_NAMES.items():
        converted = torch.tensor(model["weights"][short], dtype=weights[name].dtype)
        if not torch.equal(converted, weights[name]):
            raise ValueError(f"JSON weights do not match {name}")
    if sum(value.numel() for value in weights.values()) != 1699:
        raise ValueError("Unexpected parameter count")
    with (source / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        selected_rows = [row for row in csv.DictReader(handle)
                         if row["architecture"] == "RNN" and row["variant"] == "signals" and row["group"] == "all"]
    metrics = [{key: (int(value) if key in {"seed", "horizon", "n"} else float(value))
                if key not in {"architecture", "variant", "group", "first_origin", "last_origin", "first_target", "last_target"}
                else value for key, value in row.items()} for row in selected_rows]
    metrics.sort(key=lambda row: row["horizon"])
    if [row["horizon"] for row in metrics] != [5, 63, 252]:
        raise ValueError("Incomplete selected model metrics")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write("model.json", model)
    write("scaler.json", scaler)
    write("settings.json", settings)
    write("metrics.json", {"evaluation": selected["evaluation"], "metrics": metrics,
                           "prior_period_evaluation": plan["prior_period_evaluation"], "model_retrained": False})
    write("preprocessing.json", {"feature_columns": model["featureColumns"], "window": model["window"],
        "horizons": model["horizons"], "calendar": "Monday-Friday including holidays", "raw_column_order": ["Date",
        "brent_conocido", "wti_conocido", "vix_conocido", "oro_conocido", "cobre_conocido", "indice_dolar_conocido", "brent_fut_Close_conocido"],
        "raw_data_policy": "Already-known inputs only, no backward filling; no target labels are needed for inference",
        "scale": "(signal - saved_median) / saved_scale; never refit on inference data",
        "output": "known_Brent_spot + RNN_output * saved_target_scale",
        "interval": "point forecast +/- validation absolute-residual 80th percentile; no coverage guarantee"})
    shutil.copy2(ROOT / "app_site/engine.js", OUTPUT / "engine.js")
    write("package.json", {"name": "rnn-brent-signals", "private": True, "type": "module"})
    (OUTPUT / "README.md").write_text(model_card(model, metrics), encoding="utf-8")
    package_files = ["README.md", "model.json", "scaler.json", "settings.json", "metrics.json", "preprocessing.json",
                     "engine.js", "package.json", "inference.py", "requirements.txt", ".gitignore"]
    for name in package_files:
        if not (OUTPUT / name).is_file():
            raise FileNotFoundError(OUTPUT / name)
    write("manifest.json", {"model_id": model["id"], "checkpoint_sha256": sha(ROOT / selected["checkpoint"]),
        "model_retrained": False, "weights_format": "JSON arrays containing exact original float32 values",
        "source_files": {selected[key]: sha(ROOT / selected[key]) for key in ["checkpoint", "settings", "scaler"]},
        "files": {name: sha(OUTPUT / name) for name in package_files}})
    print(json.dumps({"package": str(OUTPUT), "model": model["id"], "parameters": 1699,
                      "files": len(package_files) + 1, "uploaded": False, "retrained": False}))


if __name__ == "__main__":
    main()
