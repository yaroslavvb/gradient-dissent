#!/usr/bin/env python3
"""Render manuscript figures solely from the verified saved A100 summary.

Reproduce: python paper/figures/make_figures.py --summary experiments/a100_transfer/results/summary.json --out paper/figures
Requires matplotlib==3.10.8. No training, services, or raw-results inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

COLORS = {"dense": "#546478", "constant_ild": "#216aab", "decreasing_ild": "#bb5425"}
LABELS = {"dense": "Dense", "constant_ild": "Constant ILD", "decreasing_ild": "Decreasing ILD"}
TASKS = ("gpt_wikitext103", "convnext_cifar100", "vit_cifar100")
TITLES = {"gpt_wikitext103": "GPT", "convnext_cifar100": "ConvNeXt", "vit_cifar100": "ViT"}
UNITS = {"gpt_wikitext103": "nats / token", "convnext_cifar100": "nats / image", "vit_cifar100": "nats / image"}
RECIPES = tuple(COLORS)
SEEDS = (2000, 2001, 2002)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#8c969f",
        "axes.linewidth": .65, "axes.titlepad": 9,
        "text.color": "#202b35", "axes.labelcolor": "#202b35",
        "xtick.color": "#394754", "ytick.color": "#394754",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 150,
        "savefig.dpi": 300, "lines.solid_capstyle": "round",
    })


def lookup(summary, task, recipe, mask):
    matches = [x for x in summary["curves"] if (x["task"], x["recipe"], x["mask_name"]) == (task, recipe, mask)]
    assert len(matches) == 1
    row = matches[0]
    assert row["seeds"] == list(SEEDS)
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e6eaee", linewidth=.6)
    axis.tick_params(length=3, width=.6)


def save(fig, out, name):
    # Preserve exactly 6.5-inch width; never crop with bbox_inches="tight".
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def quality_figure(summary, out):
    """Lines show means; numeric tables prevent overlapping endpoint labels."""
    fig = plt.figure(figsize=(6.5, 3.15))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.55, .75],
                           left=.085, right=.98, bottom=.12, top=.75,
                           hspace=.33, wspace=.32)
    handles = [Line2D([], [], color=COLORS[r], marker="o", lw=1.6, ms=4, label=LABELS[r]) for r in RECIPES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=3, frameon=False, columnspacing=1.2, handlelength=1.6)
    rows = []
    for index, task in enumerate(TASKS):
        axis = fig.add_subplot(grid[0, index])
        axis.set_title(f"{TITLES[task]}\nCE ({UNITS[task]})", pad=6)
        table_values = []
        for recipe in RECIPES:
            values = []
            for mask in ("full", "primary_two_thirds"):
                record = lookup(summary, task, recipe, mask)
                values.append(record["ce"]["mean"])
                rows.append({"task": task, "recipe": recipe, "mask": mask, "retained_blocks": record["retained"],
                             "ce_mean": record["ce"]["mean"], "ce_unit": UNITS[task], "n_training_seeds": 3,
                             **{f"ce_seed_{seed}": value for seed, value in zip(SEEDS, record["ce"]["values"])}})
            axis.plot([0, 1], values, color=COLORS[recipe], marker="o", ms=4.7, linewidth=1.75,
                      markeredgecolor="white", markeredgewidth=.5)
            table_values.append([f"{values[0]:.3f}", f"{values[1]:.3f}"])
        axis.set_xticks([0, 1], ["Full", "Retained"])
        axis.set_xlim(-.13, 1.13)
        axis.margins(y=.13)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        clean_axis(axis)
        table_axis = fig.add_subplot(grid[1, index])
        table_axis.axis("off")
        table = table_axis.table(cellText=table_values, colLabels=["Full CE", "Retained CE"],
                                 colWidths=[.5, .5], cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.PAD = .05
            if row == 0:
                cell.set_facecolor("#edf0f3")
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(COLORS[RECIPES[row-1]])
        # Table order and colors follow the common legend.
    fig.text(.5, .04, "Retained: prefix 8 / 12 (GPT, ViT); stagewise 12 / 18 (ConvNeXt).", ha="center", fontsize=9)
    save(fig, out, "fig1_full_to_pruned_ce")
    write_csv(out / "fig1_full_to_pruned_ce.csv", rows)


def accuracy_figure(summary, out):
    masks = ("full", "primary_two_thirds", "random_keeps_first_0", "random_keeps_first_1", "random_keeps_first_2", "delete_first_only")
    labels = ("Full\n12 / 12", "Prefix 8\n8 / 12", "Random 0\n8 / 12", "Random 1\n8 / 12", "Random 2\n8 / 12", "Delete first\n11 / 12")
    fig, axis = plt.subplots(figsize=(6.5, 3.1))
    fig.subplots_adjust(left=.105, right=.98, bottom=.22, top=.80)
    fig.text(.5, .985, "Secondary outcomes · CIFAR-100 · same final classifier after pruning", ha="center", va="top", fontsize=9)
    handles = [Line2D([], [], marker="s", color=COLORS[r], linestyle="none", markersize=5, label=LABELS[r]) for r in RECIPES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False, columnspacing=1.2)
    rows = []
    for j, mask in enumerate(masks):
        if j % 2 == 0:
            axis.axvspan(j-.48, j+.48, color="#f5f7f9", zorder=0)
        for recipe, offset in zip(RECIPES, (-.25, 0, .25)):
            record = lookup(summary, "vit_cifar100", recipe, mask)
            values = np.array(record["accuracy"]["values"]) * 100
            mean = record["accuracy"]["mean"] * 100
            # Offset the mean to the right of its three seed dots so the mean
            # marker cannot hide a seed when that seed is close to the mean.
            axis.scatter(j+offset+np.array([-.075, -.035, .015]), values, s=17,
                         facecolors="white", edgecolors=COLORS[recipe], linewidths=.8, zorder=3)
            axis.scatter([j+offset+.085], [mean], marker="s", s=28, c=COLORS[recipe],
                         edgecolors="white", linewidths=.45, zorder=4)
            for seed, value in zip(SEEDS, values):
                rows.append({"task": "vit_cifar100", "recipe": recipe, "mask": mask, "retained_blocks": record["retained"],
                             "seed": seed, "accuracy_percent": float(value), "mean_accuracy_percent": mean})
    axis.set_ylabel("Top-1 test accuracy (%)")
    axis.set_xticks(range(len(masks)), labels)
    axis.set_xlim(-.52, len(masks)-.48)
    axis.set_ylim(0, 60)
    axis.set_yticks([0, 15, 30, 45, 60])
    clean_axis(axis)
    fig.text(.5, .065, "Open dots: three individual training seeds. Filled squares: means.",
             ha="center", va="center", fontsize=9)
    save(fig, out, "fig2_vit_accuracy_secondary")
    write_csv(out / "fig2_vit_accuracy_secondary.csv", rows)


def effects_figure(summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.35), sharey=True)
    fig.subplots_adjust(left=.15, right=.98, bottom=.35, top=.70, wspace=.26)
    fig.text(.5, .955, "Primary paired effects on signed CE change", ha="center", va="top", fontsize=11, weight="bold")
    fig.text(.5, .892, "Squares: means · dots: three paired seeds · bars: 95% Student-t intervals (df = 2)",
             ha="center", va="top", fontsize=9)
    ranges = {"gpt_wikitext103": (-6, .7), "convnext_cifar100": (-12, 2), "vit_cifar100": (-.1, .7)}
    ticks = {"gpt_wikitext103": [-6, -4, -2, 0], "convnext_cifar100": [-12, -8, -4, 0], "vit_cifar100": [0, .2, .4, .6]}
    rows = []
    for axis, task in zip(axes, TASKS):
        axis.set_title(f"{TITLES[task]}\n{UNITS[task]}")
        axis.axvline(0, color="#667481", linewidth=.9, linestyle=(0, (3, 2)), zorder=0)
        axis.grid(axis="x", color="#e6eaee", linewidth=.6)
        for y, recipe in zip((1, 0), RECIPES[1:]):
            row = next(x for x in summary["primary_comparisons"] if x["task"] == task and x["recipe"] == recipe)
            assert row["n"] == 3 and row["df"] == 2 and row["seeds"] == list(SEEDS)
            mean, low, high = row["mean"], row["ci95_low"], row["ci95_high"]
            axis.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="none", ecolor=COLORS[recipe],
                          elinewidth=1.4, capsize=3, capthick=1, zorder=2)
            axis.scatter(row["values"], y+np.array([-.105, 0, .105]), s=15, c=COLORS[recipe],
                         marker="o", alpha=.72, zorder=3)
            axis.scatter([mean], [y], marker="s", s=31, c=COLORS[recipe], edgecolor="white", linewidth=.5, zorder=4)
            for seed, value in zip(SEEDS, row["values"]):
                rows.append({"task": task, "recipe": recipe, "seed": seed, "paired_signed_ce_change_difference": value,
                             "mean": mean, "ci95_low": low, "ci95_high": high, "n": 3, "df": 2, "unit": UNITS[task]})
        axis.set_xlim(*ranges[task])
        axis.set_xticks(ticks[task])
        axis.set_ylim(-.45, 1.45)
        axis.set_yticks([1, 0], ["Constant\nILD", "Decreasing\nILD"])
        axis.tick_params(length=3, width=.6)
        axis.spines["left"].set_visible(False)
    fig.text(.55, .244, "Paired difference in signed CE change: ILD − dense", ha="center", fontsize=9)
    fig.text(.5, .145, "Negative: lower signed CE change after pruning. Panel x-scales differ; units are shown.",
             ha="center", fontsize=9)
    fig.text(.5, .075, "Positive ViT effects do not, by themselves, imply worse absolute pruned prediction.",
             ha="center", fontsize=9)
    save(fig, out, "fig3_primary_paired_effects")
    write_csv(out / "fig3_primary_paired_effects.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path(__file__).resolve().parents[2]/"experiments/a100_transfer/results/summary.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source = args.summary.read_bytes()
    summary = json.loads(source)
    assert summary["verification"]["passed"] and summary["verification"]["complete_final_runs"] == 27
    assert len(summary["primary_comparisons"]) == 6
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    quality_figure(summary, args.out)
    accuracy_figure(summary, args.out)
    effects_figure(summary, args.out)
    metadata = {"source_summary_file": str(args.summary.resolve()), "source_summary_sha256": hashlib.sha256(source).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": platform.python_version(),
                "matplotlib": matplotlib.__version__, "numpy": np.__version__, "figure_width_inches": 6.5, "dpi": 300,
                "figure_heights_inches": {"fig1_full_to_pruned_ce": 3.15, "fig2_vit_accuracy_secondary": 3.1, "fig3_primary_paired_effects": 3.35},
                "data_source": "Verified aggregate summary only; no experiments rerun", "palette": COLORS,
                "figure_1_note": "Independent CE scales; table rows follow recipe legend order. Retained is prefix8/12 for GPT/ViT and stagewise12/18 for ConvNeXt.",
                "figure_2_note": "Secondary categorical masks; seed dots are training replicates, not extra mask replicates.",
                "figure_3_note": "Independent horizontal scales and target units; intervals have df2. Positive relative signed-change effects do not alone determine absolute pruned quality."}
    (args.out/"figure-provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Rendered three 6.5-inch / 300-dpi PNGs and their CSVs in", args.out)


if __name__ == "__main__":
    main()
