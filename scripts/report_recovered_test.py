"""Render the saved test of the six recovered seed-42 networks; never fit models."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from scripts.download_data import ROOT


OUTPUT = ROOT / "experiments/recovered_fixed_test_20261120"
SOURCE = ROOT / "experiments/fixed_split_2017_2022"
KEYS = ["architecture", "variant", "seed", "horizon"]
MEASURES = ["n", "mae", "mse", "rmse", "r2", "relative_rmse",
            "relative_rmse_vs_future", "naive_rmse", "future_rmse",
            "interval_coverage", "interval_mean_width"]
LABELS = {"levels": "Niveles", "signals": "Señales"}
COLORS = {"levels": "#D5541F", "signals": "#7A44A0"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    rendered = frame.copy()
    for name in rendered.select_dtypes(include="number").columns:
        if name not in ["seed", "horizon", "n", "selected_epoch", "epochs_run"]:
            rendered[name] = rendered[name].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "—")
    return "\n".join([
        "| " + " | ".join(map(str, rendered.columns)) + " |",
        "| " + " | ".join(["---"] * len(rendered.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rendered.itertuples(index=False, name=None)],
    ])


def html_table(frame):
    return frame.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.4f}", escape=True)


def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def main(output=OUTPUT):
    output = Path(output)
    required = ["plan.json", "complete.json", "replay.json", "predictions.csv",
                "metrics.csv", "validation_metrics.csv", "sample_counts.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Saved evaluation is incomplete: {missing}")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    if completion.get("models_evaluated") != 6 or completion.get("models_retrained") != 0:
        raise ValueError("Expected a completed evaluation of six networks without training")
    if completion.get("plan_sha256") != digest(output / "plan.json"):
        raise ValueError("Completed evaluation and plan do not match")
    for name, expected in completion.get("files", {}).items():
        if digest(output / name) != expected:
            raise ValueError(f"Saved evaluation file changed: {name}")
    if len(replay) != 6 or any(row["maximum_prediction_difference"] > 1e-8 for row in replay):
        raise ValueError("The six validation replays have not been verified")
    config = plan.get("config", plan.get("rules", plan))
    architectures = config.get("architectures", ["RNN", "LSTM", "GRU"])
    variants = config.get("variants", ["levels", "signals"])
    horizons = list(map(int, config.get("horizons", [5, 63, 252])))
    if architectures != ["RNN", "LSTM", "GRU"] or set(variants) != {"levels", "signals"} or horizons != [5, 63, 252]:
        raise ValueError("This report is only for the six networks and three horizons from the screenshot")
    forecasts = pd.read_csv(output / "predictions.csv", parse_dates=["origin", "target_date"])
    metrics = pd.read_csv(output / "metrics.csv")
    validation = pd.read_csv(output / "validation_metrics.csv")
    counts = pd.read_csv(output / "sample_counts.csv")
    for frame in [forecasts, metrics, validation]:
        if not set(KEYS).issubset(frame.columns) or not frame.seed.eq(42).all():
            raise ValueError("Expected only saved seed-42 outputs")
    test_start = pd.Timestamp(config.get("test_start", "2023-01-01"))
    test_end = pd.Timestamp(config.get("test_end_exclusive", config.get("test_end", "2025-01-01")))
    if not (forecasts.origin.ge(test_start).all() and forecasts.target_date.ge(test_start).all()
            and forecasts.target_date.lt(test_end).all()):
        raise ValueError("Forecast dates fall outside the declared test")
    if forecasts.duplicated(KEYS + ["origin", "target_date"]).any():
        raise ValueError("Duplicate saved forecasts")
    pooled = metrics.loc[metrics.group.eq("all")].copy()
    if "group" in validation.columns:
        validation = validation.loc[validation.group.eq("all")].copy()
    if "subset" in validation.columns:
        validation = validation.loc[validation.subset.eq("validation")].copy()
    if len(pooled) != 18 or pooled.duplicated(KEYS).any() or len(validation) != 18 or validation.duplicated(KEYS).any():
        raise ValueError("Expected exactly one saved metric row for each model and horizon")
    model_order = {(a, v): i for i, (a, v) in enumerate((a, v) for a in architectures for v in variants)}
    pooled["_order"] = [model_order[(a, v)] for a, v in zip(pooled.architecture, pooled.variant)]
    pooled = pooled.sort_values(["_order", "horizon"]).drop(columns="_order").reset_index(drop=True)
    out = output / "report"
    out.mkdir(exist_ok=True)
    comparison = pooled[KEYS + [m for m in MEASURES if m in pooled.columns]].merge(
        validation[KEYS + [c for c in MEASURES if c in validation.columns]].rename(
            columns={name: f"validation_{name}" for name in MEASURES if name in validation.columns}),
        on=KEYS, validate="one_to_one")
    comparison.to_csv(out / "comparison_wide.csv", index=False)
    ratios = comparison.pivot(index=["architecture", "variant", "seed"], columns="horizon",
                              values=["validation_relative_rmse", "relative_rmse"])
    ratios.columns = [f"{'validation' if name.startswith('validation_') else 'test'}_{horizon}" for name, horizon in ratios.columns]
    ratios = ratios.reset_index()
    ratios["_order"] = [model_order[(a, v)] for a, v in zip(ratios.architecture, ratios.variant)]
    ratios = ratios.sort_values("_order").drop(columns="_order")
    ratios.to_csv(out / "ratios_validation_test.csv", index=False)
    years = metrics.loc[metrics.group.str.startswith("target_year_")].copy()
    regimes = metrics.loc[metrics.group.str.startswith("origin_")].copy()
    years.to_csv(out / "metrics_by_target_year.csv", index=False)
    regimes.to_csv(out / "metrics_by_origin_regime.csv", index=False)

    plt.rcParams.update({"figure.dpi": 125, "font.size": 9, "axes.grid": True, "grid.alpha": .16,
                         "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
    for row, architecture in enumerate(architectures):
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            reference = None
            for variant in variants:
                group = forecasts.loc[forecasts.architecture.eq(architecture) & forecasts.variant.eq(variant)
                                      & forecasts.horizon.eq(horizon)].sort_values("target_date")
                if group.empty:
                    raise ValueError(f"Missing saved forecasts: {architecture}/{variant}/{horizon}")
                if reference is None:
                    reference = group
                    ax.plot(group.target_date, group.actual, color="#17365D", lw=1.6, label="Real", zorder=4)
                else:
                    for name in ["origin", "target_date", "actual", "naive"]:
                        pd.testing.assert_series_equal(reference[name].reset_index(drop=True),
                                                       group[name].reset_index(drop=True), check_names=False)
                ax.plot(group.target_date, group.prediction, color=COLORS[variant], lw=1.0, label=variant)
            ax.plot(reference.target_date, reference.naive, color="#738478", ls="--", lw=.85, label="Ingenuo spot")
            ax.set(title=f"{architecture} · {horizon} días · N={len(reference)}", ylabel="USD/barril")
            date_axis(ax)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Mismos pesos hasta 2017 · test 2023–2024 · semilla 42 · fecha objetivo", fontsize=14)
    figures.append(("01_comparacion_completa.png", "Precio real y predicciones de los seis modelos"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained", sharex=True)
    x = np.arange(3)
    for row, architecture in enumerate(architectures):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            rows = comparison.loc[comparison.architecture.eq(architecture) & comparison.variant.eq(variant)].set_index("horizon").loc[horizons]
            ax.bar(x - .18, rows.validation_relative_rmse, width=.36, color="#587D9B", label="Validación 2018–2022")
            ax.bar(x + .18, rows.relative_rmse, width=.36, color=COLORS[variant], label="Test 2023–2024")
            ax.axhline(1, color="#272F35", lw=1, ls="--", label="Ingenuo spot = 1")
            for offset, column in [(-.18, "validation_relative_rmse"), (.18, "relative_rmse")]:
                for pos, value in zip(x + offset, rows[column]):
                    ax.annotate(f"{value:.3f}", (pos, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
            ax.set(title=f"{architecture} · {LABELS[variant]}", ylabel="RMSE / RMSE ingenuo spot", xticks=x,
                   xticklabels=[f"{h} días" for h in horizons])
            ax.set_ylim(0, max(1.45, float(rows[["validation_relative_rmse", "relative_rmse"]].max().max()) * 1.45))
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Validación y test · las mismas seis redes, sin reajuste ni mezcla", fontsize=14)
    figures.append(("02_validacion_vs_test.png", "RMSE relativo en validación y test, con los mismos pesos"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    horizon_colors = {5: "#2D7591", 63: "#D07030", 252: "#79509D"}
    for row, architecture in enumerate(architectures):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            selected = forecasts.loc[forecasts.architecture.eq(architecture) & forecasts.variant.eq(variant)]
            low = min(float(selected.actual.min()), float(selected.prediction.min()))
            high = max(float(selected.actual.max()), float(selected.prediction.max()))
            ax.plot([low, high], [low, high], color="#343A40", ls="--", lw=1, label="Predicción = real")
            for horizon in horizons:
                group = selected.loc[selected.horizon.eq(horizon)]
                ax.scatter(group.actual, group.prediction, s=9, alpha=.32, color=horizon_colors[horizon],
                           edgecolors="none", label=f"{horizon} días · N={len(group)}", rasterized=True)
            ax.set(title=f"{architecture} · {LABELS[variant]}", xlabel="Real · USD/barril", ylabel="Predicción · USD/barril")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Diagnóstico en test: desviación respecto a la diagonal · semilla 42", fontsize=14)
    figures.append(("03_diagnostico.png", "Real frente a predicho por arquitectura, variante y horizonte"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)

    diagnostics = []
    for architecture in architectures:
        for variant in variants:
            source = SOURCE / architecture / variant / "seed_42" / "diagnostics.json"
            if source.exists():
                value = json.loads(source.read_text(encoding="utf-8"))
                diagnostics.append({"architecture": architecture, "variant": variant, "seed": 42,
                                    **{name: value.get(name) for name in ["selected_epoch", "epochs_run", "parameters", "overfit_signal"]}})
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(out / "original_training_diagnostics.csv", index=False)
    scalar_columns = ["architecture", "variant", "horizon", "n", "mae", "mse", "rmse", "r2", "relative_rmse", "relative_rmse_vs_future"]
    interval_columns = ["architecture", "variant", "horizon", "n", "interval_coverage", "interval_mean_width"]
    short_columns = ["architecture", "variant", "horizon", "group", "n", "rmse", "r2", "relative_rmse"]
    long_count = forecasts.loc[forecasts.horizon.eq(252)].groupby(["architecture", "variant"]).size()
    long_n = int(long_count.iloc[0])
    better = pooled.assign(better=pooled.relative_rmse.lt(1)).groupby("horizon").better.sum()
    performance = "; ".join(f"{int(better.loc[h])}/6 redes mejoran el RMSE spot a {h} días" for h in horizons)
    all_future_wins = pooled.groupby(["architecture", "variant"]).relative_rmse_vs_future.apply(lambda values: values.lt(1).all()).sum()
    future_note = ("Ninguna de las seis redes mejora al ingenuo de futuros en los tres horizontes."
                   if all_future_wins == 0 else f"{int(all_future_wins)} de las seis redes mejora al ingenuo de futuros en los tres horizontes.")
    common_note = (
        "Se evalúan los seis modelos exactos de la captura: RNN, LSTM y GRU con niveles y señales, "
        "semilla 42. Son los pesos y escalas ajustados hasta 2017, sin reentrenamiento, promedio de semillas ni mezcla con el ingenuo. "
        "La época de esos pesos se eligió con validación 2018–2022."
    )
    reuse_note = "Este bloque 2023–2024 ya se había evaluado con otra versión del modelo; aquí se realiza una comparación adicional de seis redes recuperadas."
    interval_note = (
        "Las bandas simétricas nominales del 80 % utilizan, por red y horizonte, el cuantil 80 % de los errores absolutos "
        "guardados en validación. Esa validación también intervino en la selección de la época; sus muestras se solapan. "
        "Son bandas empíricas, sin garantía conformal ni garantía de cobertura futura. No se ajusta su amplitud con el test."
    )
    metric_note = (
        "Ratio spot = RMSE del modelo / RMSE de mantener el último Brent spot publicado en el origen. "
        "Ratio futuros = RMSE del modelo / RMSE de mantener el último cierre de futuros conocido en el origen. "
        "Un ratio menor que 1 mejora esa referencia en ese bloque y métrica; no mide rentabilidad. "
        "MAE y RMSE se expresan en USD/barril; MSE en (USD/barril)². R² puede ser negativo incluso cuando se mejora al ingenuo."
    )
    text = f"""# Test de los seis modelos recuperados · 2023–2024

{common_note}

{reuse_note}

## Resultados completos

{markdown_table(pooled[scalar_columns])}

{metric_note}

{performance}. {future_note} Esta descripción no elige automáticamente un nuevo ganador con el test. Se muestran las seis redes acordadas, aunque alguna empeore la referencia o falle en un horizonte.

## Validación frente a test con los mismos pesos

{markdown_table(ratios)}

Estas cifras son de la semilla 42 de la captura. No son las medias de tres semillas publicadas en otros informes. La validación ayudó a elegir la época; el contraste con test permite observar cuánto se mantiene esa ventaja. Una mejora grande en un horizonte no demuestra sobreajuste por sí sola, ni un ratio cercano a 1 garantiza estabilidad.

![Precio real y predicciones de las seis redes](01_comparacion_completa.png)

![Validación frente a test](02_validacion_vs_test.png)

## Errores por año objetivo

{markdown_table(years[short_columns])}

El año se refiere al destino de la predicción, no al año del entrenamiento. Las predicciones a 252 días comparten muchos precios de entrada y destinos próximos; {long_n} pares evaluados por modelo no equivalen a {long_n} años independientes. No se interpreta este resultado como una prueba en múltiples ciclos económicos.

## Volatilidad conocida en el origen

{markdown_table(regimes[short_columns])}

La clasificación utiliza la desviación de cambios logarítmicos del Brent conocido en los últimos 21 días de lunes a viernes y un umbral del percentil 90 estimado antes de 2018. Un origen de volatilidad habitual puede sufrir un shock posteriormente. No se eliminan pandemias, guerras ni errores extremos de las métricas globales. Los grupos pequeños requieren especial cautela.

![Diagnóstico real frente a predicho](03_diagnostico.png)

## Bandas empíricas y cobertura observada

{markdown_table(pooled[interval_columns])}

{interval_note}

`interval_coverage` es la fracción observada dentro de la banda; `interval_mean_width` es su anchura media en USD/barril. Una banda muy amplia puede cubrir casi todos los puntos y seguir siendo poco útil.

## Fechas y etiquetas disponibles

{markdown_table(counts)}

Los horizontes 5, 63 y 252 cuentan días de lunes a viernes e incluyen festivos. Cada punto se representa en su fecha objetivo y utiliza una ventana que termina en su origen. Se conservan solo etiquetas reales válidas y publicadas antes del cierre 2025-01-01: no se rellenan precios objetivo ni se añaden datos de 2025 para completar el horizonte anual. Por eso el número de pares disminuye con el horizonte; los primeros destinos a 252 días están cerca de finales de 2023 y la mayor parte cae en 2024.

Las nuevas observaciones actualizan la ventana de entrada de la red manteniendo sus pesos y escalas. Incorporar un dato no implica reentrenar, de acuerdo con la recomendación del profesor.

## Pesos recuperados y selección original

{markdown_table(diagnostics) if not diagnostics.empty else 'La identidad de los seis modelos se conserva en `../plan.json`.'}

Los diagnósticos anteriores corresponden al ajuste original. Una señal de sobreajuste es un aviso metodológico, no una declaración de que todos sus pronósticos sean incorrectos. La comparación usa los checkpoints de la época elegida, no los de la última época. Los seis checkpoints se recargaron y sus predicciones de validación se contrastaron con los archivos originales antes del test; el detalle está en `../replay.json`.

## Archivos y reproducción

```powershell
python -m scripts.evaluate_recovered_test
python -m scripts.report_recovered_test
```

La evaluación reutiliza los resultados completados y verifica su manifiesto. El generador de este informe solo lee salidas guardadas: no entrena, cambia pesos, mezcla modelos ni recalcula métricas de evaluación. El plan identifica los seis checkpoints, sus escalas, configuraciones y amplitudes de bandas. Este informe no modifica el modelo final anterior.

- `../plan.json`: plan e identidad de las seis redes.
- `../replay.json`: contraste con la validación guardada.
- `../predictions.csv`: predicciones, etiquetas y bandas.
- `../metrics.csv`: métricas globales, por año y por régimen.
- `comparison_wide.csv`: validación y test unidos por red y horizonte.
- `ratios_validation_test.csv`: tabla compacta de ratios.
- `comparacion.html`: tablas y gráficas visibles en navegador.
- `../../../notebooks/05_test_seis_modelos.ipynb`: notebook con salidas estáticas.

Una decisión posterior basada en estas cifras debe describirse como selección después de observar el test. No se presenta la mejor fila retrospectiva como una validación nueva e independiente, ni se fuerza el resultado hacia una franja numérica concreta.
"""
    (out / "INFORME.md").write_text(text, encoding="utf-8")
    document = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test de los seis modelos recuperados · 2023–2024</title>
<style>body{{font:16px/1.6 system-ui;margin:auto;max-width:1400px;padding:28px;color:#172433;background:#f7f8fa}}h1,h2{{line-height:1.2}}figure{{margin:24px 0}}img{{width:100%;background:white}}figcaption{{color:#4c5a68;font-size:14px}}table{{border-collapse:collapse;background:white;font-size:13px;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#edf1f5}}.notice{{padding:16px;background:#fff0d6;border-left:5px solid #c57d1d}}a{{color:#12667b}}.scroll{{overflow:auto;margin:18px 0}}nav{{display:flex;gap:16px;flex-wrap:wrap}}code{{background:#eaf0f5;padding:2px 4px}}p{{max-width:1100px}}</style></head><body>
<h1>Test 2023–2024 de las seis redes de la captura</h1>
<p>{html.escape(common_note)}</p><p class="notice">{html.escape(reuse_note)}</p>
<nav><a href="#ratios">Ratios</a><a href="#curvas">Curvas</a><a href="#metricas">Todas las métricas</a><a href="#diagnostico">Diagnóstico</a><a href="#bandas">Bandas</a><a href="INFORME.md">Informe completo</a></nav>
<h2 id="ratios">Validación y test · mismos pesos hasta 2017</h2><p>Semilla 42 en ambos bloques. Las cifras de validación son de una sola semilla, no las medias de otros informes. {html.escape(performance)}.</p><div class="scroll">{html_table(ratios)}</div>
<p>{html.escape(metric_note)} {html.escape(future_note)}</p>
<figure><img src="02_validacion_vs_test.png" alt="Ratios de validación y test para cada una de las seis redes"><figcaption>No se ajusta ni elige un ganador automáticamente a partir de esta figura.</figcaption></figure>
<h2 id="curvas">Precio real y predicciones por fecha objetivo</h2><p>La disposición reproduce la captura: filas RNN/LSTM/GRU; columnas 5/63/252 días; niveles y señales en cada panel. Cada punto fue predicho desde un origen anterior.</p>
<figure><img src="01_comparacion_completa.png" alt="Rejilla de nueve curvas reales y predichas de las seis redes"><figcaption>Los precios objetivo ausentes no se rellenan. La curva del ingenuo mantiene el último spot publicado en cada origen.</figcaption></figure>
<h2 id="metricas">Métricas completas del test</h2><div class="scroll">{html_table(pooled[scalar_columns])}</div>
<h2>Por año objetivo</h2><div class="scroll">{html_table(years[short_columns])}</div>
<p>A 252 días hay {long_n} pares por modelo, con destinos muy solapados y mayoritariamente en 2024. No equivalen a {long_n} pruebas anuales independientes.</p>
<h2>Volatilidad disponible en el origen</h2><p>El umbral se fijó antes del test. Los errores en orígenes habituales también cuentan en el total; un shock puede llegar después de un origen tranquilo.</p><div class="scroll">{html_table(regimes[short_columns])}</div>
<h2 id="diagnostico">Real frente a predicho</h2><figure><img src="03_diagnostico.png" alt="Dispersión real y predicho para cada red y horizonte"><figcaption>La diagonal indica coincidencia con el precio real. El solapamiento temporal impide tratar todos los puntos como independientes.</figcaption></figure>
<h2 id="bandas">Bandas empíricas nominales del 80 %</h2><p>{html.escape(interval_note)}</p><div class="scroll">{html_table(pooled[interval_columns])}</div>
<p>La cobertura es una fracción de 0 a 1; la anchura se expresa en USD/barril. Una cobertura alta con bandas amplias no implica gran precisión.</p>
<h2>Observaciones disponibles</h2><div class="scroll">{html_table(counts)}</div><p>Los horizontes cuentan lunes a viernes, incluidos festivos. Se excluyen destinos sin precio válido publicado al cierre y los que caen fuera de 2024; no se rellenan etiquetas.</p>
<p><a href="../predictions.csv">Predicciones</a> · <a href="../metrics.csv">Métricas</a> · <a href="comparison_wide.csv">Comparación CSV</a> · <a href="../plan.json">Plan</a> · <a href="../replay.json">Verificación de pesos</a></p>
</body></html>"""
    (out / "comparacion.html").write_text(document, encoding="utf-8")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "oil-comparison"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    try:
        relative_report = out.relative_to(ROOT).as_posix()
    except ValueError:
        relative_report = str(out)
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Test de las seis redes recuperadas\n\n" + common_note + "\n\n" + reuse_note
            + "\n\nEste notebook lee resultados guardados: no entrena ni modifica modelos. El ratio 1 corresponde al ingenuo spot; menor que 1 significa menor RMSE en ese bloque, no rentabilidad."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
            "ROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\n"
            f"REPORT = ROOT / {relative_report!r}\n"
            "display(pd.read_csv(REPORT / 'ratios_validation_test.csv').round(4))\n"
            "display(pd.read_csv(REPORT / 'comparison_wide.csv').round(4))"),
        nbformat.v4.new_markdown_cell("## Precio real y predicho en el test\n\nMisma disposición que la captura: redes por fila, horizontes por columna, niveles y señales juntos. Las fechas son destinos de predicción. No se rellenan precios objetivo."),
        nbformat.v4.new_code_cell("display(Image(filename=str(REPORT / '01_comparacion_completa.png')))"),
        nbformat.v4.new_markdown_cell("## Validación frente a test\n\nSe mantienen los pesos y las escalas ajustados hasta 2017. La época se eligió con validación 2018–2022; no se reajusta nada con el test."),
        nbformat.v4.new_code_cell("display(Image(filename=str(REPORT / '02_validacion_vs_test.png')))"),
        nbformat.v4.new_markdown_cell("## Diagnóstico de errores\n\nLa diagonal indica coincidencia con el real. Los puntos a 252 días están muy solapados; no representan años independientes. Los errores extremos permanecen en la evaluación."),
        nbformat.v4.new_code_cell("display(Image(filename=str(REPORT / '03_diagnostico.png')))"),
        nbformat.v4.new_markdown_cell("Las bandas son empíricas, calculadas con residuos de la validación usada en selección; no garantizan cobertura futura. Consulta las métricas por año, régimen y cobertura en el [informe completo](../experiments/recovered_fixed_test_20261120/report/INFORME.md) y en la [versión HTML](../experiments/recovered_fixed_test_20261120/report/comparacion.html)."),
    ]
    notebook.metadata.saved_figure_sha256 = {name: digest(out / name) for name, _ in figures}
    notebook_path = ROOT / "notebooks/05_test_seis_modelos.ipynb"
    if notebook_path.exists():
        previous = nbformat.read(notebook_path, as_version=4)
        if (previous.metadata.get("saved_figure_sha256") == notebook.metadata.saved_figure_sha256
                and [cell.source for cell in previous.cells] == [cell.source for cell in notebook.cells]):
            notebook = previous
    nbformat.write(notebook, notebook_path)
    verification = {"input_sha256": {name: digest(output / name) for name in required},
                    "report_generator_sha256": digest(Path(__file__)), "models": 6, "seed": 42,
                    "forecasts": len(forecasts), "metric_rows_all": len(pooled), "figures": 3,
                    "fitting_performed": False, "evaluation_recomputed": False,
                    "report_files_sha256": {name: digest(out / name) for name in [
                        "comparison_wide.csv", "ratios_validation_test.csv", "INFORME.md", "comparacion.html",
                        *[name for name, _ in figures]]}}
    (out / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved six-network report, three figures, HTML and notebook: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Directory containing the completed saved evaluation")
    main(parser.parse_args().output)
