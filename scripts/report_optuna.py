"""Comparación de búsqueda, semillas y sobreajuste sin volver a entrenar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.confirm_optuna import check_code
from study.metrics import price_metrics
from study.search import objective_score, write_json


def recommend(scores, policy):
    """Regla prefijada: ganancia media, mayoría de semillas y dispersión."""
    baseline = scores.loc[scores.trial == 0].set_index("seed").score
    candidates = []
    for trial, group in scores.loc[scores.trial != 0].groupby("trial"):
        paired = baseline - group.set_index("seed").score
        gain = float(paired.mean())
        majority = int((paired > 0).sum()) >= policy["recommendation_minimum_seeds_better_than_baseline"]
        stable = (group.score.mean() + group.score.std(ddof=1)) < (baseline.mean() + baseline.std(ddof=1))
        if gain >= policy["recommendation_minimum_absolute_score_gain"] and majority and (
                stable or not policy["recommendation_require_mean_plus_std_better"]):
            candidates.append((float(group.score.mean()), int(trial)))
    return min(candidates)[1] if candidates else 0


def report(directory):
    check_code(directory)
    confirmation = directory / "confirmation"
    selection = json.loads((confirmation / "selection.json").read_text(encoding="utf-8"))
    architectures = list(selection["candidates"])
    for architecture in architectures:
        if not (confirmation / f"{architecture}_finished.json").exists():
            raise ValueError(f"Falta terminar confirmación de {architecture}")
    metrics = pd.concat([pd.read_csv(confirmation / f"{a}_metrics.csv") for a in architectures], ignore_index=True)
    predictions = pd.concat([pd.read_csv(confirmation / f"{a}_predictions.csv") for a in architectures], ignore_index=True)
    protocol = json.loads((directory / "source_protocol.json").read_text(encoding="utf-8"))
    if (pd.to_datetime(predictions.target_date) >= pd.Timestamp(protocol["development_end_exclusive"])).any():
        raise ValueError("Destinos fuera de desarrollo")
    if predictions.duplicated(["architecture", "trial", "seed", "fold", "horizon", "origin"]).any():
        raise ValueError("Predicciones duplicadas")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    metrics.to_csv(output / "all_finalist_metrics.csv", index=False)
    scores = pd.DataFrame([{"architecture": a, "trial": int(t), "seed": int(s), "score": objective_score(g)}
        for (a, t, s), g in metrics.groupby(["architecture", "trial", "seed"])])
    scores.to_csv(output / "scores_by_seed.csv", index=False)
    summary = scores.groupby(["architecture", "trial"]).score.agg(["mean", "std", "min", "max"]).reset_index()
    summary.to_csv(output / "stability.csv", index=False)
    recommendations = {a: recommend(scores.loc[scores.architecture == a], selection["policy"]) for a in architectures}
    setting_rows = [{"architecture": a, "trial": c["trial"], **c["settings"],
                     "recommended": c["trial"] == recommendations[a]}
                    for a, candidates in selection["candidates"].items() for c in candidates]
    pd.DataFrame(setting_rows).to_csv(output / "candidate_settings.csv", index=False)
    write_json(output / "recommendations.json", {"candidates": recommendations,
        "status": "development_candidates_only", "policy": selection["policy"],
        "test_evaluated": False, "calibration_evaluated": False})

    # Métricas agrupadas por candidato y semilla; separadas de medias de ratios anuales.
    pooled = []
    for (a, t, s, h), group in predictions.loc[predictions.eligible_for_summary].groupby(
            ["architecture", "trial", "seed", "horizon"]):
        measured = price_metrics(group.actual, group.prediction)
        naive = price_metrics(group.actual, group.naive)
        rolling = price_metrics(group.actual, group.rolling_mean_252)
        pooled.append({"architecture": a, "trial": t, "seed": s, "horizon": h, **measured,
            "relative_rmse": measured["rmse"] / naive["rmse"], "naive_r2": naive["r2"],
            "relative_rmse_vs_rolling": measured["rmse"] / rolling["rmse"]})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(output / "pooled_by_seed.csv", index=False)
    pooled_average = pooled.groupby(["architecture", "trial", "horizon"])[
        ["mae", "mse", "rmse", "r2", "relative_rmse", "relative_rmse_vs_rolling"]].mean().reset_index()
    pooled_average.to_csv(output / "pooled_metric_means.csv", index=False)
    # Diferencia anual emparejada: muestra los años perjudicados, no solo el promedio.
    baseline = metrics.loc[metrics.trial == 0, ["architecture", "seed", "fold", "horizon", "relative_rmse"]].rename(
        columns={"relative_rmse": "baseline_relative_rmse"})
    paired = metrics.merge(baseline, on=["architecture", "seed", "fold", "horizon"], validate="many_to_one")
    paired["gain_vs_baseline"] = paired.baseline_relative_rmse - paired.relative_rmse
    paired.to_csv(output / "paired_yearly_comparison.csv", index=False)

    # Diagnóstico de curvas de TODOS los ajustes completos, no solo ganadores.
    diagnostics = []
    for path in list(directory.glob("*/trial_*/*/diagnostics.json")) + list(confirmation.glob("*/trial_*/seed_*/*/diagnostics.json")):
        if path.is_relative_to(confirmation):
            a, t, s, year = path.relative_to(confirmation).parts[:4]
            seed = int(s.split("_")[1])
        else:
            a, t, year = path.relative_to(directory).parts[:3]
            seed = 42
        row = json.loads(path.read_text(encoding="utf-8"))
        settings = json.loads((path.parent / "settings.json").read_text(encoding="utf-8"))
        row["best_at_epoch_cap"] = row["selected_epoch"] == settings["epochs"]
        row.update(architecture=a, trial=int(t.split("_")[1]), seed=seed, fold=int(year), path=str(path.parent))
        diagnostics.append(row)
    diagnoses = pd.DataFrame(diagnostics)
    diagnoses.to_csv(output / "overfit_diagnostics.csv", index=False)
    anchor_rows = []
    for a in architectures:
        for trial in [0, 1, 2]:
            diag = diagnoses.loc[(diagnoses.architecture == a) & (diagnoses.trial == trial) & (diagnoses.seed == 42)]
            anchor_metrics = pd.read_csv(directory / a / f"trial_{trial:04d}" / "metrics.csv")
            anchor_rows.append({"architecture": a, "trial": trial, "score": objective_score(anchor_metrics),
                "overfit_signals": int(diag.overfit_signal.sum()), "fits": len(diag),
                "median_inner_deterioration_pct": float(diag.inner_deterioration_pct.median()),
                "fit_seconds": float(diag.fit_seconds.sum()), "parameters": int(diag.parameters.iloc[0])})
    pd.DataFrame(anchor_rows).to_csv(output / "anchor_comparison.csv", index=False)
    fig, axes = plt.subplots(len(architectures), 4, figsize=(17, 10), squeeze=False)
    for i, a in enumerate(architectures):
        selected = recommendations[a]
        for j, year in enumerate(sorted(diagnoses.fold.unique())):
            axis = axes[i, j]
            curve = pd.read_csv(directory / a / f"trial_{selected:04d}" / str(year) / "curve.csv")
            diag = diagnoses.loc[(diagnoses.architecture == a) & (diagnoses.trial == selected)
                                & (diagnoses.seed == 42) & (diagnoses.fold == year)].iloc[0]
            axis.plot(curve.epoch, curve.fit_mse_scaled, label="Ajuste", color="#253c56")
            axis.plot(curve.epoch, curve.inner_mse_scaled, label="Interno", color="#bd6539")
            axis.axvline(diag.selected_epoch, color="#7b8d76", linestyle="--", linewidth=1, label="Época elegida")
            axis.set_title(f"{a} · candidato {selected} · {year}")
            axis.set_xlabel("Época")
            axis.set_ylabel("MSE escalado, sin dropout")
            axis.grid(alpha=.2)
            if i == 0 and j == 0:
                axis.legend(fontsize=8)
    fig.suptitle("Curvas de ajuste y validación interna · semilla 42 · pesos restaurados en la época elegida")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(architectures), 4, figsize=(17, 9), squeeze=False)
    for i, a in enumerate(architectures):
        for j, year in enumerate(sorted(diagnoses.fold.unique())):
            chosen = recommendations[a]
            curve = pd.read_csv(directory / a / f"trial_{chosen:04d}" / str(year) / "curve.csv")
            selected_epoch = int(curve.loc[curve.inner_relative_rmse.idxmin(), "epoch"])
            axis = axes[i, j]
            axis.plot(curve.epoch, curve.inner_relative_rmse, color="#765787", label="RMSE relativo interno")
            axis.axhline(1., color="grey", linestyle=":", linewidth=.8, label="Ingenuo")
            axis.axvline(selected_epoch, color="#7b8d76", linestyle="--", linewidth=1, label="Época elegida")
            axis.set_title(f"{a} · candidato {chosen} · {year}")
            axis.set_xlabel("Época")
            axis.set_ylabel("Ratio interno, menor es mejor")
            axis.grid(alpha=.2)
            if i == 0 and j == 0:
                axis.legend(fontsize=8)
    fig.suptitle("Criterio de selección de época · semilla 42")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "selection_curves.png", dpi=150)
    plt.close(fig)
    # Gráfico por intento: los podados solo tienen puntuación interna, no son comparables.
    fig, axes = plt.subplots(1, len(architectures), figsize=(13, 4), squeeze=False)
    counts = []
    for i, a in enumerate(architectures):
        trials = pd.read_csv(directory / f"{a}_trials.csv")
        complete = trials.loc[trials.state == "COMPLETE"].sort_values("number")
        axis = axes[0, i]
        axis.scatter(complete.number, complete.value, s=24, alpha=.7, label="Intento completo")
        axis.step(complete.number, complete.value.cummin(), where="post", label="Mejor acumulado")
        axis.axhline(float(complete.loc[complete.number == 0, "value"].iloc[0]), color="grey", linestyle="--", label="Control")
        axis.set_title(a)
        axis.set_xlabel("Número de intento")
        axis.set_ylabel("RMSE relativo medio de desarrollo")
        axis.grid(alpha=.2)
        counts.append({"architecture": a, "complete_trials": int((trials.state == "COMPLETE").sum()),
                       "pruned_trials": int((trials.state == "PRUNED").sum()), "failed_trials": int((trials.state == "FAIL").sum())})
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "search_progress.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(counts).to_csv(output / "trial_counts.csv", index=False)
    pruning = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*/trial_*/pruned_training.json")]
    new_models = sum(json.loads((confirmation / f"{a}_finished.json").read_text(encoding="utf-8"))["new_models_fitted"] for a in architectures)
    budget = {"counts": counts, "complete_search_fits": int(sum(c["complete_trials"] for c in counts) * 4),
        "partial_pruned_fits": len(pruning), "new_confirmation_fits": new_models,
        "total_fits_including_partial": len(diagnoses) + len(pruning),
        "total_epochs": int(diagnoses.epochs_run.sum() + sum(p["epochs_run"] for p in pruning)),
        "sum_fit_seconds": float(diagnoses.fit_seconds.sum() + sum(p["fit_seconds"] for p in pruning))}
    write_json(output / "budget.json", budget)

    # Exportación individual de la semilla 42: permite usar el informe estadístico común.
    export = directory / "development_candidates"
    export.mkdir(exist_ok=True)
    chosen_predictions = pd.concat([predictions.loc[(predictions.architecture == a) & (predictions.trial == t)
        & (predictions.seed == 42)] for a, t in recommendations.items()], ignore_index=True)
    chosen_predictions.to_csv(export / "validation_predictions.csv", index=False)
    write_json(export / "source_protocol.json", protocol)
    write_json(export / "status.json", {"phase": "validation_v2", "test_evaluated": False,
        "calibration_evaluated": False, "selected_during_development_search": True, "seed": 42})
    write_json(export / "selection.json", recommendations)
    # Si se promedian semillas después de verlas, no sería confirmación independiente.
    # Aquí se informa cada semilla y su media de métricas; no se inventa un ensemble posterior.
    lines = ["# Optuna en CPU: resultados de desarrollo", "",
        "Se comparan redes recurrentes con una o dos capas, cabeza densa opcional, agregación temporal, "
        "dropout, L2, MSE/Huber, learning rate, batch y reducción de learning rate. Sin convoluciones. "
        "Se mantienen ventana 60 y los mismos datos, fronteras y horizontes.", "",
        "La búsqueda elige hiperparámetros con los años exteriores de desarrollo 2018–2021; cada época "
        "se selecciona exclusivamente en un tramo interno anterior al año evaluado. Los resultados de "
        "desarrollo intervienen en la selección de candidatos y pueden ser optimistas. Las tres semillas "
        "comprueban sensibilidad a la inicialización sobre esos mismos periodos: no son un nuevo test. "
        "Calibración 2022–2023 y test 2024+ no se han evaluado.", "",
        "## Trabajo ejecutado", "", "| Red | Intentos completos | Podados | Fallidos |",
        "|---|---:|---:|---:|"]
    for c in counts:
        lines.append(f"| {c['architecture']} | {c['complete_trials']} | {c['pruned_trials']} | {c['failed_trials']} |")
    lines += ["", f"Ajustes completos de búsqueda: {budget['complete_search_fits']}. "
        f"Ajustes nuevos de confirmación: {new_models}. Ajustes parciales podados: {len(pruning)}. "
        f"Total: {budget['total_fits_including_partial']} ajustes, {budget['total_epochs']} épocas. "
        f"Suma de tiempos medidos de los ajustes: {budget['sum_fit_seconds'] / 3600:.2f} horas. "
        "Los procesos se han ejecutado en paralelo: esa suma no equivale al tiempo de reloj.", "",
        "Cada arquitectura tiene un presupuesto de hasta 40 intentos y 30 minutos. El límite se comprueba "
        "entre intentos; el que ya está en curso termina. Cada intento completo ajusta cuatro modelos "
        "temporales. Los podados solo ajustan parcialmente el primer corte. Sus valores internos no "
        "se comparan con las puntuaciones exteriores de los completos.", "",
        "## Estabilidad de las finalistas", "",
        "Puntuación = media de ratios RMSE/red ingenua por año, primero dentro de cada horizonte y "
        "después entre los tres horizontes. Menor es mejor. Solo se resumen grupos con al menos 20 etiquetas. "
        "Se usa una semilla en búsqueda y tres en confirmación; la primera se reutiliza.", "",
        "| Red | Candidato | Media | Desv. entre semillas | Mínimo | Máximo | Propuesto |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary.itertuples():
        lines.append(f"| {row.architecture} | {row.trial} | {row.mean:.4f} | {row.std:.4f} | "
                     f"{row.min:.4f} | {row.max:.4f} | {'Sí' if recommendations[row.architecture] == row.trial else ''} |")
    lines += ["", "Configuraciones propuestas para seguir investigando:", "",
        "| Red | Capas × unidades | Cabeza | Resumen | Dropout | LR | Batch | L2 | Pérdida | Scheduler |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    for row in setting_rows:
        if row["recommended"]:
            lines.append(f"| {row['architecture']} | {row['num_layers']} × {row['hidden_size']} | {row['head_size']} | "
                f"{row['pooling']} | {row['dropout']:.3f} | {row['learning_rate']:.6f} | {row['batch_size']} | "
                f"{row['weight_decay']:.5f} | {row['loss']} | {'Sí' if row['scheduler'] else 'No'} |")
    lines += ["", "El candidato 0 es el control de una capa de 32 unidades, con el mismo máximo de 40 épocas. "
        "La regla se fijó antes de confirmar: mejora media mínima de 0,005 puntos, ganancia en al menos "
        "dos semillas y menor media + desviación que el control. Si nadie la cumple se conserva el control. "
        "Tres semillas no permiten afirmar significación estadística ni estabilidad en otros regímenes.", "",
        "## Métricas en unidades originales", "",
        "Medias de métricas agrupadas por semilla para los candidatos propuestos. MAE/RMSE en USD/barril; "
        "MSE en USD²/barril². R² puede ser negativo. El ratio de esta tabla agrupa errores y difiere de "
        "la puntuación anterior, que promedia ratios anuales.", "",
        "| Red | Candidato | Horizonte | MAE | MSE | RMSE | R² | RMSE/ingenuo | RMSE/media móvil |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pooled_average.itertuples():
        if row.trial == recommendations[row.architecture]:
            lines.append(f"| {row.architecture} | {row.trial} | {row.horizon} | {row.mae:.2f} | "
                         f"{row.mse:.2f} | {row.rmse:.2f} | {row.r2:.3f} | {row.relative_rmse:.3f} | {row.relative_rmse_vs_rolling:.3f} |")
    lines += ["", "## Sobreajuste", "",
        "Se marca una señal cuando, después de la época elegida, el MSE de ajuste baja y el ratio interno "
        "empeora más del 2 %. Es una señal operativa; un error interno mayor que el de ajuste, por sí solo, "
        "también puede reflejar un cambio de régimen. Todas las curvas se calculan en modo evaluación, "
        "sin dropout, y con la misma escala del objetivo ajustada solo en entrenamiento.", "",
        f"Se observa esa señal en {int(diagnoses.overfit_signal.sum())} de {len(diagnoses)} ajustes completos. "
        "Se restauran siempre los pesos de la mejor época interna; no se utilizan los pesos finales deteriorados. "
        "`overfit_diagnostics.csv` conserva el detalle de todas las configuraciones completas.", "",
        f"En {int(diagnoses.best_at_epoch_cap.sum())} ajustes la mejor época coincide con el máximo permitido. "
        "En esos casos no se ha observado aún una parada por deterioro; aumentar el máximo es una "
        "posible prueba posterior, manteniendo la selección interna. No se presupone convergencia.", "",
        "Comparación de los tres controles prefijados, semilla 42. El control 1 añade capacidad y el 2 "
        "combina varias medidas de regularización; al cambiar varias cosas a la vez no se puede atribuir "
        "su efecto a un único componente.", "",
        "| Red | Control | Parámetros | Puntuación | Señales / 4 cortes | Deterioro interno mediano | Segundos |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in anchor_rows:
        lines.append(f"| {row['architecture']} | {row['trial']} | {row['parameters']} | {row['score']:.4f} | "
            f"{row['overfit_signals']} | {row['median_inner_deterioration_pct']:.1f} % | {row['fit_seconds']:.1f} |")
    lines += ["",
        "El MSE escalado y la media de ratios RMSE ponderan de forma distinta los horizontes y las "
        "etiquetas disponibles. La época se elige con el segundo criterio; puede no coincidir con "
        "el mínimo del MSE interno. Ambas curvas se muestran por separado.", "",
        "![Curvas de MSE](learning_curves.png)", "", "![Criterio de selección](selection_curves.png)", "",
        "![Búsqueda](search_progress.png)", "",
        "El dropout recurrente solo actúa con dos capas; además hay dropout explícito antes de la cabeza. "
        "L2, Huber, reducción de learning rate y early stopping se prueban, sin suponer que todos mejoren. "
        "Los tres controles iniciales se ejecutan sin poda, incluida la referencia sencilla y dos "
        "configuraciones de dos capas. La poda de los siguientes se basa en la curva interna del primer "
        "corte; puede descartar redes que mejoren más tarde o en otro periodo.", "",
        "## Archivos para revisar", "",
        "- `stability.csv` y `scores_by_seed.csv`: sensibilidad a inicialización.",
        "- `all_finalist_metrics.csv`: R², MAE, MSE, RMSE, ratios y épocas por año/horizonte/semilla.",
        "- `paired_yearly_comparison.csv`: periodos donde cada finalista mejora o empeora respecto al control.",
        "- `pooled_by_seed.csv`: métricas agrupadas y comparación con la media móvil de 252 observaciones.",
        "- `overfit_diagnostics.csv`: deterioro interno y pérdidas de ajuste.",
        "- `../confirmation/selection.json`: finalistas y regla congeladas antes de las semillas adicionales.",
        "- [Reales frente a predicciones](../development_candidates/statistical_report/INFORME.md): "
        "semilla 42 de los candidatos propuestos, seleccionados durante desarrollo.", "",
        "Los horizontes se cuentan sobre un calendario de lunes a viernes. Las etiquetas faltantes no "
        "se imputan; a 252 días, el grupo 2021 tiene solo cinco etiquetas y queda fuera del resumen. "
        "Los errores entre orígenes solapados están correlacionados. Una mejora de RMSE histórico "
        "no demuestra rentabilidad operable.", ""]
    (output / "INFORME.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "verification.json", {"prediction_sha256": hashlib.sha256(
        b"".join((confirmation / f"{a}_predictions.csv").read_bytes() for a in architectures)).hexdigest(),
        "test_evaluated": False, "calibration_evaluated": False, "models_trained_by_report": 0})
    print(summary.to_string(index=False))
    print("Propuestas:", recommendations)
    print("Presupuesto:", budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    report(parser.parse_args().experiment.resolve())
