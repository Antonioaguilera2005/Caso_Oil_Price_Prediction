"""Report saved fixed-split models without fitting models or reading test data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from scripts.download_data import ROOT
from scripts.run_fixed_split import OUTPUT
from study.metrics import price_metrics
from study.search import write_json


def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def markdown_table(frame):
    rendered = frame.copy()
    for name in rendered.select_dtypes(include="float").columns:
        rendered[name] = rendered[name].map(lambda x: f"{x:.3f}")
    rows = ["| " + " | ".join(rendered.columns) + " |",
            "| " + " | ".join(["---"] * len(rendered.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in rendered.itertuples(index=False, name=None))
    return "\n".join(rows)


def report():
    completion = json.loads((OUTPUT / "completion.json").read_text(encoding="utf-8"))
    plan = json.loads((OUTPUT / "plan.json").read_text(encoding="utf-8"))
    rules = plan["rules"]
    assert not completion["test_evaluated"]
    assert completion["models_fitted"] == completion["checkpoints_verified"] == 18
    for name, digest in plan["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    predictions = pd.read_csv(OUTPUT / "predictions.csv", parse_dates=["origin", "target_date"])
    metrics = pd.read_csv(OUTPUT / "metrics.csv")
    assert (predictions.target_date < pd.Timestamp(rules["test_start"])).all()
    validation = predictions.loc[predictions.subset == "validation"].copy()
    train = predictions.loc[predictions.subset == "train"]
    assert (train.target_date < pd.Timestamp(rules["training_end_exclusive"])).all()
    assert (validation.origin >= pd.Timestamp(rules["validation_start"])).all()
    out = OUTPUT / "report"
    out.mkdir(exist_ok=True)
    validation_metrics = metrics.loc[metrics.subset == "validation"]
    columns = ["mae", "mse", "rmse", "r2", "relative_rmse", "relative_rmse_vs_future"]
    keys = ["architecture", "variant", "horizon"]
    summary = validation_metrics.groupby(keys)[columns].mean().reset_index()
    sizes = validation_metrics.groupby(keys).agg(n=("n", "first"), seeds=("seed", "nunique"),
                                                rmse_std=("rmse", "std")).reset_index()
    summary = sizes.merge(summary, on=keys)
    summary.to_csv(out / "validation_means.csv", index=False)
    validation_metrics.to_csv(out / "validation_by_seed.csv", index=False)
    scores = summary.groupby(["architecture", "variant"])[["relative_rmse", "relative_rmse_vs_future"]].mean().reset_index()
    scores = scores.sort_values("relative_rmse").reset_index(drop=True)
    scores.to_csv(out / "candidate_scores.csv", index=False)
    best = scores.iloc[0]
    selection = {"architecture": str(best.architecture), "variant": str(best.variant), "display_seed": 42,
                 "criterion": "Mean validation RMSE/known-spot RMSE, equal horizon and seed weights",
                 "status": "validation_candidate_only", "test_evaluated": False,
                 "relative_rmse": float(best.relative_rmse)}
    write_json(out / "best_validation_candidate.json", selection)
    annual_tables = []
    for axis, column in [("origin", "origin"), ("target", "target_date")]:
        rows = []
        for (architecture, variant, seed, horizon, year), group in validation.groupby(
                ["architecture", "variant", "seed", "horizon", validation[column].dt.year]):
            measured = price_metrics(group.actual, group.prediction)
            spot = price_metrics(group.actual, group.naive)
            future = price_metrics(group.actual, group.future_naive)
            rows.append({"architecture": architecture, "variant": variant, "seed": seed,
                         "horizon": horizon, "year": year, **measured,
                         "relative_rmse": measured["rmse"] / spot["rmse"],
                         "relative_rmse_vs_future": measured["rmse"] / future["rmse"],
                         "eligible_for_summary": len(group) >= rules["minimum_evaluation_samples"]})
        annual = pd.DataFrame(rows)
        annual.to_csv(out / f"annual_by_{axis}.csv", index=False)
        annual_tables.append(annual)
    diagnostics = []
    for path in OUTPUT.glob("*/*/seed_*/diagnostics.json"):
        diag = json.loads(path.read_text(encoding="utf-8"))
        diagnostics.append({"architecture": path.parents[2].name, "variant": path.parents[1].name,
                            "seed": int(path.parent.name.removeprefix("seed_")), **diag})
    diagnostics = pd.DataFrame(diagnostics).sort_values(["architecture", "variant", "seed"])
    diagnostics.to_csv(out / "training_diagnostics.csv", index=False)
    plt.rcParams.update({"figure.dpi": 115, "font.size": 10, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"levels": "#D5541F", "signals": "#7A44A0"}
    real_color, naive_color = "#17365D", "#738478"
    figures = []
    # All five validation years use a single checkpoint per cell and seed.
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), layout="constrained")
    for row, architecture in enumerate(rules["architectures"]):
        for col, horizon in enumerate(rules["horizons"]):
            ax = axes[row, col]
            group = validation.loc[(validation.architecture == architecture) & (validation.horizon == horizon)
                                   & (validation.seed == 42)].sort_values("target_date")
            base = group.loc[group.variant == "levels"]
            ax.plot(base.target_date, base.actual, lw=1.6, color=real_color, label="Real", zorder=4)
            for variant in rules["variants"]:
                selected = group.loc[group.variant == variant]
                ax.plot(selected.target_date, selected.prediction, lw=1.05, color=colors[variant], label=variant)
            ax.plot(base.target_date, base.naive, lw=.9, ls="--", color=naive_color, label="Ingenuo spot")
            ax.set_title(f"{architecture} · {horizon} días · N={len(base)}")
            ax.set_ylabel("USD/barril")
            date_axis(ax)
            ax.legend(fontsize=8)
    fig.suptitle("Entrenamiento hasta 2017 · validación 2018–2022 · semilla 42 · fecha objetivo", fontsize=14)
    figures.append(("01_comparacion_completa.png", "Comparación de todas las redes y horizontes"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)
    selected = validation.loc[(validation.architecture == best.architecture) & (validation.variant == best.variant)
                              & (validation.seed == 42)].sort_values("target_date")
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), layout="constrained")
    for ax, horizon in zip(axes, rules["horizons"]):
        group = selected.loc[selected.horizon == horizon]
        ax.plot(group.target_date, group.actual, color=real_color, lw=1.8, label="Real")
        ax.plot(group.target_date, group.prediction, color="#D5541F", lw=1.2, label="Predicho")
        ax.plot(group.target_date, group.naive, color=naive_color, lw=1, ls="--", label="Ingenuo spot")
        ax.set(ylabel="USD/barril", title=f"Horizonte {horizon} días · N={len(group)}")
        date_axis(ax)
        ax.legend()
    fig.suptitle(f"Candidato por validación: {best.architecture} / {best.variant} · semilla 42 · fecha objetivo", fontsize=14)
    figures.append(("02_detalle_candidato.png", "Precio real y predicho del candidato con menor ratio medio"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for row, architecture in enumerate(rules["architectures"]):
        for col, variant in enumerate(rules["variants"]):
            ax = axes[row, col]
            folder = OUTPUT / architecture / variant / "seed_42"
            curve = pd.read_csv(folder / "curve.csv")
            diag = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
            ax.plot(curve.epoch, curve.fit_mse_scaled, color=real_color, label="Entrenamiento")
            ax.plot(curve.epoch, curve.inner_mse_scaled, color="#D5541F", label="Validación 2018–2022")
            ax.axvline(diag["selected_epoch"], color="#7A44A0", ls="--", label="Época elegida por ratio")
            ax.set(title=f"{architecture} / {variant} · semilla 42", xlabel="Época", ylabel="MSE normalizado")
            ax.legend(fontsize=8)
    figures.append(("03_curvas_entrenamiento.png", "Pérdida de entrenamiento y validación"))
    fig.savefig(out / figures[-1][0])
    plt.close(fig)
    readable = summary[["architecture", "variant", "horizon", "n", *columns]].rename(columns={
        "architecture": "Red", "variant": "Entradas", "horizon": "Horizonte", "n": "N",
        "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²",
        "relative_rmse": "Ratio spot", "relative_rmse_vs_future": "Ratio futuros"})
    seconds = completion["sum_fit_seconds"]
    intro = (f"18 ajustes completados en {seconds:.1f} segundos acumulados de entrenamiento en CPU. "
             "Un ajuste por arquitectura, variante y semilla; los mismos pesos se usan para todos los años. "
             "Máximo 40 épocas, paciencia 10. No se ha ejecutado una nueva búsqueda de Optuna.")
    warning = ("Estas métricas pertenecen a la validación usada para seleccionar la época y comparar candidatos; "
               "no son resultados de un test independiente. El test 2023–2024 sigue reservado. "
               "Las cifras son medias de métricas de tres semillas, no métricas de una predicción promediada.")
    diagnosis = (f"La regla diagnóstica detecta señal de sobreajuste en {int(diagnostics.overfit_signal.sum())} "
                 "de 18 ajustes: la pérdida de entrenamiento baja y el ratio de validación empeora más del 2% "
                 "desde la época elegida hasta el final. Se conservan los pesos de la época elegida. "
                 "Este indicador no demuestra que el sobreajuste esté eliminado; las diferencias de régimen "
                 "entre 2012–2017 y 2018–2022 también afectan a las pérdidas.")
    lines = ["# Entrenamiento fijo hasta 2017 y validación 2018–2022", "", intro, "", warning, "",
             "Entradas disponibles desde 22/03/2012; primer origen con ventana de 60 días: 13/06/2012. "
             "Pesos y escalas ajustados solo antes de 2018. Etiquetas de entrenamiento antes de 2018 "
             "y de validación antes de 2023, respetando también su publicación. No se rellenan precios objetivo.", "",
             "## Métricas de validación", "", markdown_table(readable), "",
             "MAE y RMSE: USD/barril. MSE: (USD/barril)². Ratio spot/futuros: RMSE del modelo dividido "
             "por el de mantener el último spot publicado/cierre de futuros disponible en el origen. "
             "No son métricas de rentabilidad ni contratos ajustados al vencimiento de cada horizonte.", "",
             "## Sobreajuste", "", diagnosis, "",
             f"Candidato con menor media de ratios spot: **{best.architecture} / {best.variant}**. "
             "Es una selección de desarrollo; no hay modelo final certificado por test.", ""]
    for filename, title in figures:
        lines.extend([f"## {title}", "", f"![{title}]({filename})", ""])
    lines.extend(["## Archivos", "", "`validation_by_seed.csv`: detalle por semilla. "
                  "`annual_by_origin.csv` y `annual_by_target.csv`: desgloses con los mismos pesos, "
                  "sin reentrenamiento anual. Se señalan los grupos con menos de 20 etiquetas; "
                  "sí contribuyen al cálculo del bloque completo. "
                  "`training_diagnostics.csv`: épocas y evolución de pérdidas. "
                  "Los 18 checkpoints se han recargado y sus predicciones y métricas se han verificado.", ""])
    (out / "INFORME.md").write_text("\n".join(lines), encoding="utf-8")
    sections = "\n".join(f'<h2>{title}</h2><a href="{name}"><img src="{name}" alt="{title}"></a>' for name, title in figures)
    html = f'''<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Validación fija 2018–2022</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#192d43;background:#f4f6f9;margin:0}}
main{{max-width:1600px;padding:24px;margin:auto}}img{{width:100%;height:auto;background:white}}
table{{border-collapse:collapse;background:white;font-size:14px}}td,th{{padding:8px;border:1px solid #dce3e9}}
.scroll{{overflow:auto}}.note{{border-left:4px solid #D5541F;padding-left:16px}}</style></head>
<body><main><h1>Entrenamiento hasta 2017 · validación 2018–2022</h1><p>{intro}</p>
<p class="note">{warning}</p><p>Las figuras representan cada predicción en su fecha objetivo. Cada punto
procede de un origen distinto; no es una única trayectoria recursiva. Semilla 42 en las figuras.</p>
<h2>Métricas de validación</h2><div class="scroll">{readable.to_html(index=False, float_format=lambda v: f'{v:.3f}')}</div>
<p>MAE y RMSE en USD/barril; MSE en (USD/barril)². Ratios frente al último spot publicado
y al último cierre de futuros conocido; no representan una estrategia de inversión.</p>
<h2>Sobreajuste</h2><p>{diagnosis}</p>{sections}
<p><a href="../../../notebooks/02_particion_fija_2018_2022.ipynb">Abrir notebook</a> ·
<a href="validation_by_seed.csv">Métricas por semilla</a> · <a href="annual_by_origin.csv">Métricas anuales</a></p>
</main></body></html>'''
    (out / "comparacion.html").write_text(html, encoding="utf-8")
    # A separate notebook leaves the old experiment's figures identifiable.
    notebook_cells = [nbf.v4.new_markdown_cell("# Partición fija: entrenamiento hasta 2017 y validación 2018–2022\n\n"
        + intro + "\n\n" + warning + "\n\n[Vista HTML](../experiments/fixed_split_2017_2022/report/comparacion.html). "
        "Las figuras de comparación usan la semilla 42. "
        "Este notebook solo lee los resultados guardados; no entrena ni abre el test."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'experiments/fixed_split_2017_2022/report/validation_means.csv').is_file())\n"
            "REPORT = ROOT / 'experiments/fixed_split_2017_2022/report'\n"
            "display(pd.read_csv(REPORT / 'validation_means.csv'))")]
    for filename, title in figures:
        notebook_cells.extend([nbf.v4.new_markdown_cell(f"## {title}\n\n" + (diagnosis if "curvas" in filename else
            "Eje horizontal: fecha objetivo. Cada punto corresponde a un origen distinto. Los pesos no cambian por año.")),
            nbf.v4.new_code_cell(f"display(Image(filename=str(REPORT / '{filename}')))" )])
    notebook = nbf.v4.new_notebook(cells=notebook_cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python (.venv petróleo)", "language": "python"}})
    nbf.write(notebook, ROOT / "notebooks/02_particion_fija_2018_2022.ipynb")
    write_json(out / "verification.json", {**completion, "training_code_hashes_verified": True,
        "predictions_before_test": True, "models_trained_by_report": 0,
        "best_validation_candidate": selection, "report_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(readable.round(3).to_string(index=False))
    print(f"Report: {out / 'comparacion.html'}")


if __name__ == "__main__":
    report()
