"""Create four slide-ready figures from the selected model's frozen CSV outputs.

This reads existing predictions and training history only. No training, model
loading, hyperparameter selection or new historical evaluation takes place.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation" / "assets"
HORIZONS = (5, 63, 252)
NAVY, UAX, MUTED, AMBER = "#16364F", "#0060AF", "#617181", "#B07D32"
GRID, PALE = "#DFE6EC", "#83B5DE"
MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(frame: pd.DataFrame, delivery: dict) -> pd.DataFrame:
    result = frame.loc[(frame.architecture == delivery["architecture"])
                       & (frame.variant == delivery["variant"])
                       & (frame.seed == delivery["seed"])].copy()
    if result.empty:
        raise ValueError("Selected frozen model is absent from the saved results.")
    return result


def style(ax, *, label_size: float = 12) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B4C0CA")
    ax.tick_params(labelsize=label_size, colors=MUTED, length=3, pad=5)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def dates(ax, values: pd.Series) -> None:
    first, last = values.min(), values.max()
    # Three ticks leave enough room between adjacent panels at their slide size.
    ticks = [first, first + (last - first) / 2, last]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{MONTHS[mdates.num2date(v).month - 1]} {mdates.num2date(v).year % 100:02d}"))
    margin = pd.Timedelta(days=max(8, (last - first).days * .055))
    ax.set_xlim(first - margin, last + margin)


def save(fig, name: str, entries: list[dict]) -> None:
    png, svg = OUT / f"{name}.png", OUT / f"{name}.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    entries.append({"name": name, "width_inches": float(fig.get_figwidth()),
                    "height_inches": float(fig.get_figheight()),
                    "png": {"path": png.relative_to(ROOT).as_posix(), "sha256": sha256(png)},
                    "svg": {"path": svg.relative_to(ROOT).as_posix(), "sha256": sha256(svg)}})
    plt.close(fig)


def test_curves(predictions: pd.DataFrame, entries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.65), sharey=True)
    fig.subplots_adjust(left=.061, right=.987, bottom=.20, top=.76, wspace=.23)
    limits = predictions[["actual", "prediction", "naive"]]
    for ax, h in zip(axes, HORIZONS):
        rows = predictions.loc[predictions.horizon == h].sort_values("target_date")
        style(ax)
        # Lines connect observed targets; they do not interpolate missing labels.
        ax.plot(rows.target_date, rows.naive, color=AMBER, linewidth=1.25,
                linestyle=(0, (4, 3)), label="Ingenuo spot", zorder=2)
        ax.plot(rows.target_date, rows.actual, color=NAVY, linewidth=1.75,
                label="Real", zorder=3)
        ax.plot(rows.target_date, rows.prediction, color=UAX, linewidth=1.5,
                label="RNN con señales", zorder=4)
        ax.set_title(f"{h} días · n = {len(rows)}", fontsize=15, color=NAVY,
                     weight="bold", pad=11)
        dates(ax, rows.target_date)
        ax.set_ylim(limits.min().min() - 3, limits.max().max() + 3)
        ax.set_xlabel("Fecha objetivo", fontsize=12.5, color=MUTED, labelpad=8)
        if h == 5:
            ax.set_ylabel("USD/barril", fontsize=12.5, color=MUTED, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    order = (1, 2, 0)
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(.52, .995), ncol=3,
               fontsize=13, frameon=False, labelcolor=NAVY, columnspacing=2.2)
    save(fig, "final_test_curves", entries)


def validation_test(validation: pd.DataFrame, test: pd.DataFrame, entries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    fig.subplots_adjust(left=.14, right=.985, bottom=.21, top=.79)
    style(ax, label_size=11.5)
    xpos = np.arange(3)
    for dx, frame, color, label in ((-.19, validation, PALE, "Validación"),
                                      (.19, test, UAX, "Test histórico")):
        values = frame.set_index("horizon").loc[list(HORIZONS), "relative_rmse"].to_numpy()
        bars = ax.bar(xpos + dx, values, width=.33, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.4f}".replace(".", ",") for v in values],
                     fontsize=10.5, padding=-16,
                     color=NAVY if dx < 0 else "white", weight="bold")
    line = ax.axhline(1, color=AMBER, linewidth=1.4, linestyle=(0, (4, 3)),
                     label="Ingenuo = 1", zorder=4)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, .5, 1])
    ax.set_xticks(xpos, [f"{h} días" for h in HORIZONS])
    ax.set_ylabel("RMSE relativo", fontsize=12, color=MUTED, labelpad=8)
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib lists the reference line before the bars; explicit order avoids ambiguity.
    lookup = dict(zip(labels, handles))
    names = ["Validación", "Test histórico", "Ingenuo = 1"]
    fig.legend([lookup[n] for n in names], names, loc="upper center", ncol=3,
               bbox_to_anchor=(.53, 1.0), fontsize=10.5, frameon=False,
               labelcolor=NAVY, columnspacing=.9, handlelength=1.4)
    save(fig, "final_validation_test", entries)


def training(curve: pd.DataFrame, diagnostics: dict, entries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 2.7))
    fig.subplots_adjust(left=.16, right=.975, bottom=.21, top=.79)
    style(ax, label_size=11.5)
    epoch = int(diagnostics["selected_epoch"])
    final_epoch = int(curve.epoch.max())
    ax.axvspan(epoch, final_epoch, facecolor="#F5F0E7", zorder=0)
    ax.plot(curve.epoch, curve.fit_mse_scaled, color=NAVY, linewidth=1.8, label="Entrenamiento")
    ax.plot(curve.epoch, curve.inner_mse_scaled, color=UAX, linewidth=1.8, label="Validación")
    ax.axvline(epoch, color=AMBER, linewidth=1.3, linestyle=(0, (3, 3)))
    best = curve.loc[curve.epoch == epoch].iloc[0]
    ax.scatter([epoch], [best.inner_mse_scaled], s=28, color=UAX, zorder=4)
    ax.annotate(f"Época {epoch}", (epoch, best.inner_mse_scaled),
                xytext=(0, 14), textcoords="offset points", ha="center", color=AMBER,
                fontsize=11.5, weight="bold")
    ax.set_xlim(1, final_epoch)
    ax.set_ylim(.5, 2.15)
    ax.set_xticks([1, 10, epoch, final_epoch])
    ax.set_xlabel("Época", fontsize=12, color=MUTED, labelpad=5)
    ax.set_ylabel("MSE normalizado", fontsize=12, color=MUTED, labelpad=8)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.54, 1), ncol=2, fontsize=11.5, frameon=False,
               labelcolor=NAVY, columnspacing=1.2, handlelength=1.7)
    save(fig, "final_training_curve", entries)


def errors(predictions: pd.DataFrame, entries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    fig.subplots_adjust(left=.066, right=.987, bottom=.22, top=.78, wspace=.23)
    bound = np.ceil((predictions.prediction - predictions.actual).abs().max() / 5) * 5
    for ax, h in zip(axes, HORIZONS):
        rows = predictions.loc[predictions.horizon == h].sort_values("target_date")
        residual = rows.prediction - rows.actual
        style(ax)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.fill_between(rows.target_date, residual, 0, color=PALE, alpha=.28)
        ax.plot(rows.target_date, residual, color=UAX, linewidth=1.5)
        ax.set_ylim(-bound - 1, bound + 1)
        ax.set_yticks([-20, 0, 20])
        dates(ax, rows.target_date)
        ax.set_title(f"{h} días", fontsize=15, weight="bold", color=NAVY, pad=12)
        ax.set_xlabel("Fecha objetivo", fontsize=12.5, color=MUTED, labelpad=8)
        if h == 5:
            ax.set_ylabel("Predicho − real (USD/barril)", fontsize=12.5,
                          color=MUTED, labelpad=8)
    save(fig, "final_test_errors", entries)


def error_regimes(metrics: pd.DataFrame, entries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    fig.subplots_adjust(left=.14, right=.985, bottom=.28, top=.81)
    style(ax, label_size=11.5)
    xpos = np.arange(3)
    counts = []
    for dx, group, color, label in ((-.19, "origin_usual", UAX, "Habitual"),
                                    (.19, "origin_high_volatility", AMBER, "Alta vol. (origen)")):
        rows = metrics.loc[metrics.group == group].set_index("horizon").loc[list(HORIZONS)]
        values = rows.relative_rmse.to_numpy()
        bars = ax.bar(xpos + dx, values, width=.33, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.3f}".replace(".", ",") for v in values],
                     fontsize=10.5, padding=4, color=NAVY, weight="bold",
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": .3}, zorder=5)
        counts.append(rows.n.to_numpy())
    ax.axhline(1, color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)),
               label="Ingenuo = 1", zorder=4)
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_xticks(xpos, [f"{h} días\n{counts[0][i]} / {counts[1][i]}" for i, h in enumerate(HORIZONS)])
    ax.set_ylabel("RMSE relativo", fontsize=12, color=MUTED, labelpad=8)
    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    names = ["Habitual", "Alta vol. (origen)", "Ingenuo = 1"]
    fig.legend([lookup[n] for n in names], names, loc="upper center", ncol=3,
               bbox_to_anchor=(.53, 1.0), fontsize=10.5, frameon=False,
               labelcolor=NAVY, columnspacing=.9, handlelength=1.3)
    fig.text(.54, .025, "n habitual / alta volatilidad", ha="center", fontsize=10.5, color=MUTED)
    save(fig, "final_error_regimes", entries)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 12,
                         "text.color": NAVY, "svg.fonttype": "none",
                         "axes.unicode_minus": True})
    config_path = ROOT / "config" / "delivery_model.json"
    delivery = json.loads(config_path.read_text(encoding="utf-8"))
    model = (ROOT / delivery["checkpoint"]).parent
    evaluation = ROOT / delivery["evaluation"]
    sources = {"delivery_config": config_path, "checkpoint": ROOT / delivery["checkpoint"],
               "test_predictions": evaluation / "predictions.csv",
               "test_metrics": evaluation / "metrics.csv",
               "validation_metrics": evaluation / "validation_metrics.csv",
               "training_history": model / "curve.csv", "training_diagnostics": model / "diagnostics.json"}
    before = {key: sha256(path) for key, path in sources.items()}
    predictions = selected(pd.read_csv(sources["test_predictions"]), delivery)
    predictions["target_date"] = pd.to_datetime(predictions.target_date)
    predictions["origin"] = pd.to_datetime(predictions.origin)
    predictions = predictions.sort_values(["horizon", "target_date"])
    metrics = selected(pd.read_csv(sources["test_metrics"]), delivery)
    all_groups = metrics.copy()
    metrics = metrics.loc[metrics.group == "all"].sort_values("horizon")
    validation = selected(pd.read_csv(sources["validation_metrics"]), delivery).sort_values("horizon")
    curve = pd.read_csv(sources["training_history"])
    diagnostics = json.loads(sources["training_diagnostics"].read_text(encoding="utf-8"))
    assert dict(predictions.groupby("horizon").size()) == {5: 499, 63: 441, 252: 258}
    assert len(curve) == 27 and diagnostics["selected_epoch"] == 17
    assert not predictions.duplicated(["horizon", "origin", "target_date"]).any()
    # Check plotted rows exactly reproduce the saved aggregate errors.
    summaries = []
    for row in metrics.itertuples():
        rows = predictions.loc[predictions.horizon == row.horizon]
        error = rows.prediction - rows.actual
        np.testing.assert_allclose(float(np.sqrt(np.mean(error ** 2))), row.rmse, rtol=1e-12)
        worst = rows.loc[error.abs().nlargest(3).index].copy()
        summaries.append({"horizon": int(row.horizon), "n": int(len(rows)),
                          "mean_signed_error": float(error.mean()),
                          "first_target": rows.target_date.min().strftime("%Y-%m-%d"),
                          "last_target": rows.target_date.max().strftime("%Y-%m-%d"),
                          "target_year_counts": {str(k): int(v) for k, v in rows.groupby(rows.target_date.dt.year).size().items()},
                          "worst_errors": [{"origin": r.origin.strftime("%Y-%m-%d"),
                                            "target_date": r.target_date.strftime("%Y-%m-%d"),
                                            "actual": float(r.actual), "prediction": float(r.prediction),
                                            "signed_error": float(r.prediction-r.actual)} for r in worst.itertuples()]})
    figures: list[dict] = []
    test_curves(predictions, figures)
    validation_test(validation, metrics, figures)
    training(curve, diagnostics, figures)
    error_regimes(all_groups, figures)
    assert before == {key: sha256(path) for key, path in sources.items()}
    manifest = {"model_id": delivery["id"], "no_training_or_new_evaluation": True,
                "template": r"C:\Users\anton\Downloads\PPT_Template (4).pptx",
                "palette": {"actual": NAVY, "rnn_uax_blue": UAX, "naive": AMBER, "validation": PALE},
                "figures": figures, "test_metrics": metrics.to_dict(orient="records"),
                "validation_metrics": validation.to_dict(orient="records"),
                "training_diagnostics": diagnostics, "error_summaries": summaries,
                "origin_regime_metrics": all_groups.loc[all_groups.group.isin(["origin_usual", "origin_high_volatility"])].to_dict(orient="records"),
                "sources": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": before[k]} for k, p in sources.items()},
                "interpretation": ["At 5 days, the model improves RMSE against known spot by 11.44%, but is worse than the known-futures reference.",
                                   "At 63 and 252 days, the test performance is approximately the spot naive baseline; negative R² remains visible in the saved metrics.",
                                   "Validation 2018–2022 selected epoch 17; later validation loss increases while training loss falls. Early stopping limits the observed fitting, without guaranteeing unseen-period generalization.",
                                   "All curves use observed target dates. No missing targets are imputed or smoothed.",
                                   "At 252 days, six target observations lie in 2023 and 252 in 2024. All 258 remain in the charts and metrics; overlapping horizons are not independent tests.",
                                   "The error sign is predicted minus observed. No error is attributed to a historical event without separate supporting evidence.",
                                   "High-volatility groups use origin-only 21-weekday known Brent log-change volatility above a 90th-percentile pre-2018 threshold. Their samples contain 26, 11 and 9 overlapping observations, so comparisons are descriptive rather than strong statistical evidence.",
                                   "High origin volatility does not imply greater absolute error at every horizon: the figure compares RMSE relative to each same-group spot naive baseline."]}
    (OUT / "final_figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [f["name"] for f in figures], "source_files_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
