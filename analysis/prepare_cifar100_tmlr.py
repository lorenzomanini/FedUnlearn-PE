"""Reproduce the CIFAR-100 within-superclass paper analysis from saved artifacts.

Run from the repository root:
    python analysis/prepare_cifar100_tmlr.py --input stat_tests/CIFAR100_within_class
Reads every seed_*/experiment and its completed controls (seed_*/controls or
seed_*/controls_job_*/controls). Model checkpoints are optional: without them the
audit comes from checkpoint_audit.csv written by experiments/within_class_finalize.py.
Requires NumPy, SciPy, Matplotlib and PyTorch; never modifies experiment inputs.

Primary endpoint: apple accuracy closure (paper Eq. 55). The single pre-specified
confirmatory contrast is PRIMARY_CONTRAST in experiments/configs/cifar100_within_class.py;
everything else is reported as per-seed paired differences with t-intervals.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fedunlearn-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.configs.cifar100_within_class import PILOT_SEEDS, PRIMARY_CONTRAST  # noqa: E402

SOURCE = ROOT / "stat_tests/CIFAR100_within_class"
OUT = ROOT / "paper/cifar100_within_class"
MAIN_ARMS = ("core", "swap", "cross", "leverage", "random")
FRONTIER_ARMS = ("core", "random", "leverage", "swap", "cross", "diagonal")
BUDGETS = (8, 10, 15)
GROUPS = ("target", "siblings", "other_superclasses", "overall")
FINE_NAMES = ("apple", "mushroom", "orange", "pear", "sweet_pepper")
LABELS = {"core": "Core", "swap": "Orange-swap", "cross": "Cross-superclass swap",
          "leverage": "Leverage", "random": "Matched random", "diagonal": "Diagonal",
          "finetune_core": "Fine-tune core 10% (no reset)", "finetune_all": "Fine-tune all (no reset)",
          "original": "Original", "gold_mean": "Gold mean"}
# Reference categorical palette, fixed slot order (dataviz skill); markers double-encode.
COLORS = {"core": "#2a78d6", "random": "#eb6834", "leverage": "#1baf7a",
          "swap": "#eda100", "cross": "#e87ba4", "diagonal": "#008300"}
MARKERS = {"core": "o", "random": "s", "leverage": "^", "swap": "D", "cross": "v", "diagonal": "P"}
INK, MUTED = "#222222", "#6b6b6b"
PRIMARY = "target_accuracy_closure"


def write_csv(name, rows, fieldnames=None):
    if not rows:
        return
    # Pilot and new seeds carry different optional fields: use the ordered union.
    fieldnames = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def exact_signflip(values):
    """Two-sided paired mean test under independent symmetric seed differences."""
    observed = abs(statistics.mean(values))
    null = [abs(statistics.mean(s * v for s, v in zip(signs, values)))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(x >= observed - 1e-12 for x in null) / len(null)


def describe(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"n": 1, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = statistics.stdev(values)
    half = float(stats.t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def closure(original, candidate, gold):
    denominator = original - gold
    return None if abs(denominator) <= 1e-9 else (original - candidate) / denominator


def gold_mean(golds):
    return {group: {metric: statistics.mean(g["metrics"][group][metric] for g in golds)
                    for metric in ("accuracy", "loss", "margin")}
            for group in GROUPS} | {"membership": {"loss_auc": statistics.mean(
                g["metrics"]["membership"]["loss_auc"] for g in golds)}}


def retained(metrics, metric):
    """Accuracy/loss/margin over all non-apple test images, from the disjoint groups."""
    sib, other = metrics["siblings"], metrics["other_superclasses"]
    count_sib = sib.get("count", 400)
    count_other = other.get("count", 9500)
    return (count_sib * sib[metric] + count_other * other[metric]) / (count_sib + count_other)


def flatten(seed, arm, budget, metrics, original, reference):
    row = {"seed": seed, "arm": arm, "budget_percent": budget}
    for group in GROUPS:
        for metric in ("accuracy", "loss", "margin"):
            row[f"{group}_{metric}" + ("_percent" if metric == "accuracy" else "")] = (
                metrics[group][metric] * (100 if metric == "accuracy" else 1))
    for metric in ("accuracy", "loss", "margin"):
        row[f"retained_{metric}" + ("_percent" if metric == "accuracy" else "")] = (
            retained(metrics, metric) * (100 if metric == "accuracy" else 1))
    margin_gap = original["target"]["margin"] - reference["target"]["margin"]
    assert margin_gap > 1e-6, "A nonpositive/near-zero gap needs separate interpretation"
    acc = lambda m, g: m[g]["accuracy"]
    row[PRIMARY] = closure(acc(original, "target"), acc(metrics, "target"), acc(reference, "target"))
    row["target_margin_gap_closure"] = closure(original["target"]["margin"], metrics["target"]["margin"],
                                               reference["target"]["margin"])
    # Apple change beyond the siblings' change, as a fraction of the gold's own contrast.
    specific = lambda m, key: (m["target"][key] - original["target"][key]) - (m["siblings"][key] - original["siblings"][key])
    for key, name in (("accuracy", "apple_specific_accuracy_fraction"), ("margin", "apple_specific_margin_fraction")):
        denominator = specific(reference, key)
        row[name] = specific(metrics, key) / denominator if abs(denominator) > 1e-9 else None
    gold_retained = retained(reference, "accuracy")
    row.update(
        target_accuracy_delta_gold_pp=100 * (acc(metrics, "target") - acc(reference, "target")),
        siblings_accuracy_delta_gold_pp=100 * (acc(metrics, "siblings") - acc(reference, "siblings")),
        overall_accuracy_delta_gold_pp=100 * (acc(metrics, "overall") - acc(reference, "overall")),
        retained_accuracy_delta_gold_pp=100 * (retained(metrics, "accuracy") - gold_retained),
        retained_accuracy_drop_vs_gold_pp=100 * (gold_retained - retained(metrics, "accuracy")),
        other_superclasses_margin_delta_gold=(metrics["other_superclasses"]["margin"]
                                              - reference["other_superclasses"]["margin"]),
        loss_auc=metrics["membership"]["loss_auc"],
        loss_auc_distance_half=abs(metrics["membership"]["loss_auc"] - 0.5))
    return row


def control_arm(name):
    """Map controls report names onto (arm, budget) rows."""
    if name == "no_reset_core_10":
        return "finetune_core", 10.0
    if name == "no_reset_all":
        return "finetune_all", 100.0
    if name.startswith("rank_"):
        _, rank, arm, budget = name.split("_")
        return f"{arm}_r{rank}", float(budget)
    arm, budget = name.rsplit("_", 1)
    return arm, float(budget)


def find_controls(job):
    reports = []
    for path in sorted(job.glob("controls/results.json")) + sorted(job.glob("controls_job_*/controls/results.json")):
        report = json.loads(path.read_text())
        if report.get("status") == "complete":
            reports.append((path, report))
        else:
            print(f"Ignoring incomplete controls {path} (status {report.get('status')!r})", flush=True)
    assert len(reports) <= 1, f"Several complete controls for {job}: {[str(p) for p, _ in reports]}"
    return reports[0] if reports else (None, None)


def load_masks(experiment):
    path = experiment / "masks.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        total = int(stored["parameter_numels"].sum())
        return {key: np.unpackbits(stored[key], count=total).astype(bool)
                for key in stored.files if key not in ("parameter_names", "parameter_numels")}


def checkpoint_audit(experiment, result):
    saved = experiment / "checkpoint_audit.csv"
    if saved.exists():
        with saved.open() as stream:
            return list(csv.DictReader(stream)), "verified_on_cluster"
    if (experiment / "original.pt").exists():
        from experiments.within_class_finalize import audit
        load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
        scores = load(experiment / "scores.pt")["scores"]
        shapes = {name: tuple(value.shape) for name, value in scores["core"].items()}
        return audit(experiment, result, scores, load(experiment / "original.pt"), shapes), "verified_locally"
    return [], "unavailable"


def b_spectrum(evals, projection):
    evals, projection = np.asarray(evals, float), np.asarray(projection, float)
    whitened = projection / np.sqrt(np.outer(evals, evals))
    eig = np.linalg.eigvalsh((whitened + whitened.T) / 2)
    trace_sq = float((eig ** 2).sum())
    return {"rank": len(evals), "trace_B": float(eig.sum()), "trace_B2": trace_sq,
            "max_eig_B": float(eig.max()), "leverage_share": len(evals) / trace_sq if trace_sq else None,
            "lambda1_over_lambdar": float(evals.max() / evals.min())}


def primary_contrast(rows, seeds, variant):
    spec = PRIMARY_CONTRAST
    lookup = {(r["seed"], r["arm"], r["budget_percent"]): r for r in rows}
    records = []
    for seed in seeds:
        core = lookup.get((seed, spec["arm"], spec["arm_budget_percent"]))
        controls = [lookup[(seed, spec["control"], b)] for b in spec["control_budgets_percent"]
                    if (seed, spec["control"], b) in lookup]
        record = {"seed": seed, "confirmatory": seed in spec["seeds"] and variant == spec["variant"],
                  "control_budgets_available": " ".join(f"{c['budget_percent']:g}" for c in controls)}
        if core is None or len(controls) < 2:
            records.append(record | {"bracketed": False, "excluded_reason": "missing arm or control points"})
            continue
        x = np.array([c[spec["matching"]] for c in controls])
        y = np.array([c[spec["endpoint"]] for c in controls])
        x_core = core[spec["matching"]]
        record.update(arm_matching=x_core, control_matching_min=float(x.min()),
                      control_matching_max=float(x.max()), arm_endpoint=core[spec["endpoint"]])
        if not x.min() <= x_core <= x.max():
            records.append(record | {"bracketed": False, "excluded_reason": "control does not bracket arm (no extrapolation)"})
            continue
        order = np.argsort(x, kind="stable")
        interpolated = float(np.interp(x_core, x[order], y[order]))
        records.append(record | {"bracketed": True, "excluded_reason": "",
                                 "control_interpolated": interpolated,
                                 "difference": core[spec["endpoint"]] - interpolated})
    usable = [r for r in records if r["bracketed"]]
    confirmatory = [r["difference"] for r in usable if r["confirmatory"]]
    summary = {"specification": spec, "variant": variant,
               "excluded": [{"seed": r["seed"], "reason": r["excluded_reason"]} for r in records if not r["bracketed"]],
               "exploratory_all_seeds": describe([r["difference"] for r in usable])}
    if len(confirmatory) >= 2:
        test = stats.ttest_1samp(confirmatory, 0.0)
        summary["confirmatory"] = {**describe(confirmatory), "t": float(test.statistic),
                                   "df": len(confirmatory) - 1, "p_two_sided": float(test.pvalue),
                                   "reject_at_alpha": bool(test.pvalue < spec["alpha"]),
                                   "signflip_p_two_sided": exact_signflip(confirmatory)}
    else:
        summary["confirmatory"] = {"status": f"not available: {len(confirmatory)} confirmatory seed(s) usable"}
    return records, summary


def paired_differences(rows, seeds, pairs, metrics):
    lookup = {(r["seed"], r["arm"], r["budget_percent"]): r for r in rows}
    out = []
    for (first, second, budget), metric in itertools.product(pairs, metrics):
        diffs = {}
        for seed in seeds:
            a, b = lookup.get((seed, first, budget)), lookup.get((seed, second, budget))
            if a and b and a.get(metric) is not None and b.get(metric) is not None:
                diffs[seed] = a[metric] - b[metric]
        if len(diffs) < 2:
            continue
        out.append({"contrast": f"{first}_minus_{second}", "budget_percent": budget, "metric": metric,
                    **describe(list(diffs.values())), "same_sign_seeds": max(sum(v > 0 for v in diffs.values()),
                                                                             sum(v < 0 for v in diffs.values())),
                    "per_seed": " ".join(f"{s}:{v:+.4g}" for s, v in sorted(diffs.items()))})
    return out


def mask_overlaps(seed, masks):
    rows = []
    for budget in BUDGETS:
        core = masks.get(f"core_{budget:g}")
        if core is None:
            continue
        for arm in MAIN_ARMS[1:]:
            other = masks.get(f"{arm}_{budget:g}")
            if other is not None:
                shared = int((core & other).sum())
                rows.append({"seed": seed, "budget_percent": budget, "comparison": f"core_vs_{arm}",
                             "shared_fraction": shared / int(core.sum()),
                             "jaccard": shared / int((core | other).sum())})
    return rows


def style_axes(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def frontier_figure(rows, seeds, arms):
    """Retained-accuracy drop vs apple accuracy closure; gold at (0, 1)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    ax.axhline(1, color=MUTED, ls="--", lw=0.8)
    ax.axvline(0, color=MUTED, ls="--", lw=0.8)
    for arm in arms:
        budgets = sorted({r["budget_percent"] for r in rows if r["arm"] == arm})
        if not budgets:
            continue
        for seed in seeds:
            points = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            ax.scatter([r["retained_accuracy_drop_vs_gold_pp"] for r in points], [r[PRIMARY] for r in points],
                       s=14, marker=MARKERS[arm], color=COLORS[arm], alpha=0.3, lw=0)
        means = [(statistics.mean(r["retained_accuracy_drop_vs_gold_pp"] for r in rows if r["arm"] == arm and r["budget_percent"] == b),
                  statistics.mean(r[PRIMARY] for r in rows if r["arm"] == arm and r["budget_percent"] == b))
                 for b in budgets]
        ax.plot(*zip(*means), color=COLORS[arm], lw=2, marker=MARKERS[arm], ms=7,
                markeredgecolor="white", markeredgewidth=1, label=LABELS[arm])
        if arm in ("core", "random"):  # label only the ends of each budget path
            for budget, (x, y) in [(budgets[0], means[0]), (budgets[-1], means[-1])]:
                ax.annotate(f"{budget:g}%", (x, y), xytext=(5, -10), textcoords="offset points",
                            fontsize=7, color=MUTED)
    original = [r for r in rows if r["arm"] == "original"]
    ax.scatter([statistics.mean(r["retained_accuracy_drop_vs_gold_pp"] for r in original)], [0],
               marker="X", s=70, color=MUTED, zorder=5, label="Original (before unlearning)")
    ax.scatter([0], [1], marker="*", s=160, color=INK, zorder=5, label="Gold retraining")
    ax.set(xlabel="Retained test accuracy drop vs. gold (pp; right = worse utility)",
           ylabel="Apple accuracy closure (1 = gold)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Forgetting vs. utility cost, {len(seeds)} seeds (faint: per seed; line: mean)",
                 fontsize=9, color=INK, loc="left")
    fig.savefig(OUT / "frontier_retained_drop_vs_apple_closure.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "frontier_retained_drop_vs_apple_closure.png", dpi=220)
    plt.close(fig)


def figures(rows, summary, fine, seeds, arms):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    frontier_figure(rows, seeds, [a for a in FRONTIER_ARMS if any(r["arm"] == a for r in rows)])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    endpoints = ((PRIMARY, "Apple accuracy closure", 1),
                 ("siblings_accuracy_delta_gold_pp", "Sibling accuracy vs. gold (pp)", 0),
                 ("retained_accuracy_delta_gold_pp", "Retained accuracy vs. gold (pp)", 0))
    for ax, (metric, label, reference) in zip(axes, endpoints):
        ax.axhline(reference, color=MUTED, ls="--", lw=1)
        for j, arm in enumerate(arms):
            x = np.arange(len(BUDGETS)) + (j - (len(arms) - 1) / 2) * 0.09
            entries = [next(s for s in summary if s["arm"] == arm and s["budget_percent"] == b and s["metric"] == metric) for b in BUDGETS]
            ax.plot(x, [s["mean"] for s in entries], marker=MARKERS[arm], color=COLORS[arm], label=LABELS[arm])
            for seed in seeds:
                values = [next((r[metric] for r in rows if r["seed"] == seed and r["arm"] == arm and r["budget_percent"] == b), None) for b in BUDGETS]
                ax.scatter(x, [np.nan if v is None else v for v in values], s=12, alpha=0.35, color=COLORS[arm], lw=0)
        ax.set(xticks=range(len(BUDGETS)), xticklabels=[f"{b}%" for b in BUDGETS], xlabel="Parameter budget", ylabel=label)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "budget_tradeoffs.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "budget_tradeoffs.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.55 * len(arms) + 1.6), layout="constrained", sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        matrix = [[statistics.mean(r["accuracy_delta_gold_pp"] for r in fine
                                   if r["arm"] == arm and r["budget_percent"] == budget and r["fine_class"] == name)
                   for name in FINE_NAMES] for arm in arms]
        im = ax.imshow(matrix, cmap="RdBu", vmin=-25, vmax=25, aspect="auto")
        for i in range(len(arms)):
            for j in range(len(FINE_NAMES)):
                ax.text(j, i, f"{matrix[i][j]:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[i][j]) > 17 else "black")
        ax.set(xticks=range(len(FINE_NAMES)), xticklabels=["Apple", "Mushroom", "Orange", "Pear", "Sweet pepper"],
               yticks=range(len(arms)), yticklabels=[LABELS[a] for a in arms], title=f"{budget}% parameter budget")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy vs. gold (pp)")
    fig.savefig(OUT / "fine_class_accuracy.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "fine_class_accuracy.png", dpi=220)
    plt.close(fig)
    core_gold_accuracy(rows, fine, seeds)
    before_after_accuracy(rows)


def _band(ax, values, color, style, label):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    ax.axhspan(mean - sd, mean + sd, color=color, alpha=0.12)
    ax.axhline(mean, color=color, ls=style, lw=1.8, label=label)
    return mean


def before_after_accuracy(rows):
    """Include the model trained on all apples as the before-unlearning baseline."""
    blue, green, gray = COLORS["core"], "#008300", MUTED
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, (group, title) in zip(axes, (("target", "Apple images"),
                                      ("siblings", "Four retained siblings"),
                                      ("retained", "All retained images (excludes apple)"))):
        metric = f"{group}_accuracy_percent"
        _band(ax, [r[metric] for r in rows if r["arm"] == "original"], gray, ":", "Before unlearning")
        _band(ax, [r[metric] for r in rows if r["arm"] == "gold_mean"], green, "--", "Gold retraining")
        values = [[r[metric] for r in rows if r["arm"] == "core" and r["budget_percent"] == b] for b in BUDGETS]
        means = [statistics.mean(v) for v in values]
        ax.errorbar(range(3), means, yerr=[statistics.stdev(v) for v in values],
                    fmt="o-", color=blue, capsize=4, lw=1.8, label="After Core unlearning")
        for x, value in enumerate(means):
            ax.annotate(f"{value:.1f}%", (x, value), xytext=(0, 12 if group == "target" else -25),
                        textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.set(title=title, xticks=range(3), xticklabels=[f"{b}%" for b in BUDGETS],
               xlabel="Parameter budget", ylim=(0, 100), xlim=(-0.3, 2.3))
        style_axes(ax)
    axes[0].set_ylabel("Correct-superclass test accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    n = sum(r["arm"] == "original" for r in rows)
    fig.suptitle("Before unlearning, after Core, and gold retraining", fontsize=12)
    fig.supxlabel(f"Means over {n} seeds; bars/bands show ±1 sample SD. Gold retrains are averaged within each seed.", fontsize=8)
    fig.savefig(OUT / "before_after_accuracy.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "before_after_accuracy.png", dpi=220)
    plt.close(fig)


def core_gold_accuracy(rows, fine, seeds):
    """Show actual accuracies against nested gold means, rather than differences."""
    blue, green = COLORS["core"], "#008300"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained", sharey=True)
    for ax, (group, title) in zip(axes, (("target", "Apple (forget target)"),
                                      ("siblings", "Four retained siblings"),
                                      ("other_superclasses", "Other retained superclasses"))):
        metric = f"{group}_accuracy_percent"
        gold = _band(ax, [r[metric] for r in rows if r["arm"] == "gold_mean"], green, "--", "Gold retraining")
        core = [[r[metric] for r in rows if r["arm"] == "core" and r["budget_percent"] == b] for b in BUDGETS]
        means = [statistics.mean(values) for values in core]
        ax.errorbar(range(3), means, yerr=[statistics.stdev(values) for values in core],
                    fmt="o-", capsize=4, color=blue, lw=1.8, label="Core")
        for x, value in enumerate(means):
            ax.annotate(f"{value:.1f}%", (x, value), xytext=(0, 12 if group == "target" else -25), textcoords="offset points",
                        ha="center", color=INK, fontsize=9)
        ax.text(0.03, 0.05, f"Gold: {gold:.1f}%", color=INK, transform=ax.transAxes)
        ax.set(title=title, xticks=range(3), xticklabels=[f"{b}%" for b in BUDGETS],
               xlabel="Parameter budget", ylim=(0, 100), xlim=(-0.3, 2.3))
        style_axes(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Core versus gold retraining: superclass accuracy", fontsize=12)
    fig.supxlabel(f"Means across {len(seeds)} seeds; error bars and gold bands show ±1 sample SD. Gold retrains are averaged within each seed.", fontsize=8)
    fig.savefig(OUT / "core_vs_gold_accuracy.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "core_vs_gold_accuracy.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), layout="constrained", sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        for offset, arm, label, color in ((-0.19, "core", "Core", blue), (0.19, "gold", "Gold retraining", green)):
            means, deviations = [], []
            for name in FINE_NAMES:
                if arm == "core":
                    values = [r["accuracy_percent"] for r in fine if r["fine_class"] == name
                              and r["arm"] == arm and r["budget_percent"] == budget]
                else:
                    values = [statistics.mean(r["accuracy_percent"] for r in fine if r["fine_class"] == name
                                              and r["arm"].startswith("gold_") and r["seed"] == seed)
                              for seed in seeds]
                means.append(statistics.mean(values))
                deviations.append(statistics.stdev(values))
            ax.bar(np.arange(5) + offset, means, width=0.36, color=color, label=label,
                   yerr=deviations, capsize=2, error_kw={"elinewidth": 0.8})
        ax.set(title=f"{budget}% parameter budget", ylim=(0, 100), xticks=range(5),
               xticklabels=["Apple\n(forget)", "Mushroom", "Orange", "Pear", "Sweet\npepper"])
        ax.tick_params(axis="x", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        style_axes(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Core versus gold on apple and each retained sibling", fontsize=12)
    fig.supxlabel(f"Correct-superclass accuracy within each fine class; mean ±1 sample SD over {len(seeds)} seeds (golds averaged within seed).", fontsize=8)
    fig.savefig(OUT / "core_vs_gold_fine_classes.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "core_vs_gold_fine_classes.png", dpi=220)
    plt.close(fig)


def tables(summary, arms, seeds):
    def cell(arm, budget, metric, digits=2):
        s = next((s for s in summary if s["arm"] == arm and s["budget_percent"] == budget and s["metric"] == metric), None)
        if s is None or s["mean"] is None:
            return "--"
        return f"${s['mean']:.{digits}f}$" if s["sd"] is None else f"${s['mean']:.{digits}f} \\pm {s['sd']:.{digits}f}$"
    columns = ((PRIMARY, 3), ("target_margin_gap_closure", 3), ("siblings_accuracy_delta_gold_pp", 2),
               ("retained_accuracy_delta_gold_pp", 2), ("other_superclasses_margin", 2))
    main = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Budget & Selection & Apple acc.\ closure & Margin closure & Siblings $\Delta$ (pp) & Retained $\Delta$ (pp) & Other-superclass margin \\",
            r"\midrule"]
    for name, budget in (("original", 0), ("gold_mean", 0)):
        main.append(f"-- & {LABELS[name]} & " + " & ".join(cell(name, budget, m, d) for m, d in columns) + r" \\")
    main.append(r"\midrule")
    for budget in BUDGETS:
        for arm in arms:
            main.append(f"{budget}\\% & {LABELS[arm]} & " + " & ".join(cell(arm, budget, m, d) for m, d in columns) + r" \\")
        if budget != BUDGETS[-1]:
            main.append(r"\midrule")
    main += [r"\bottomrule", r"\end{tabular}",
             rf"\caption{{Within-superclass deletion on CIFAR-100. Mean $\pm$ sample standard deviation over {len(seeds)} original-training seeds; the three gold retrains are averaged within each seed. Apple accuracy closure (primary endpoint; NFS$_{{\mathrm{{TargetAcc}}}}$, Eq.~55) is 1 at the gold mean and 0 at the original. $\Delta$ columns are accuracy differences from the gold mean on the four retained siblings and on all retained (non-apple) test images. The gold mean's other-superclass margin is the reference for global confidence loss.}}",
             r"\label{tab:within-class}", r"\end{table}"]
    (OUT / "main_table.tex").write_text("\n".join(main) + "\n")
    supp = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Budget & Selection & Apple (\%) & Apple margin & Overall (\%) & Loss AUC \\", r"\midrule"]
    for arm, budget in [("original", 0), ("gold_mean", 0)] + [(a, b) for b in BUDGETS for a in arms]:
        supp.append((f"{budget}\\%" if budget else "--") + f" & {LABELS[arm]} & " + " & ".join(
            cell(arm, budget, m, 3 if m == "loss_auc" else 2) for m in (
                "target_accuracy_percent", "target_margin", "overall_accuracy_percent", "loss_auc")) + r" \\")
    supp += [r"\bottomrule", r"\end{tabular}",
             rf"\caption{{Additional diagnostics (mean $\pm$ sample standard deviation over {len(seeds)} seeds). Gold mean denotes the seed-level average of three retrains, whose standard deviation across seeds is not the variation across individual gold models. Loss AUC uses 500 apple training members and 100 apple test nonmembers; it is a diagnostic rather than a privacy guarantee.}}",
             r"\label{tab:within-class-extra}", r"\end{table}"]
    (OUT / "diagnostic_table.tex").write_text("\n".join(supp) + "\n")


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE, help="parent directory containing seed jobs")
    parser.add_argument("--output", type=Path, default=OUT, help="directory for summaries and figures")
    args = parser.parse_args()
    SOURCE, OUT = args.input.resolve(), args.output.resolve()
    assert exact_signflip([1, 2, 3]) == 0.25
    assert exact_signflip([0, 0, 0]) == 1
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob("*/experiment/results.json"))
    assert len(paths) >= 3, f"Need at least three seeds under {SOURCE}, found {len(paths)}"
    rows, fine, audit, diagnostics, manifest, golds_raw = [], [], [], [], [], []
    overlaps, timings, stationarity, rank_comparisons = [], [], [], []
    status, seen, variants = [], set(), set()
    for path in paths:
        experiment, job = path.parent, path.parent.parent
        result = json.loads(path.read_text())
        protocol = json.loads((experiment / "protocol.json").read_text())
        config, seed = protocol["config"], result["seed"]
        assert seed == protocol["seed"] and seed not in seen, f"Duplicate or inconsistent seed {seed}"
        seen.add(seed)
        variants.add(config.get("variant", "standard"))
        assert config["num_classes"] == 20 and config["model_name"] == "resnet18"
        assert config["dataset_name"] == "cifar100_superclass"
        assert (config["target_fine_class"], config["swap_fine_class"]) == (0, 53)
        assert config["train_epochs"] == 40 and config["recovery_epochs"] == 1
        assert config["parameter_budgets"] == list(BUDGETS)
        assert len(result["golds"]) == 3
        target, kept = protocol["target_indices"], protocol["retained_indices"]
        assert len(target) == len(set(target)) == 500 and len(kept) == len(set(kept)) == 49500
        assert not set(target) & set(kept) and set(target + kept) == set(range(50000))
        assert protocol["client_indices"][0] == target
        assert set(protocol["swap_indices"]).issubset(kept)
        arms = {(a["arm"], a["parameter_budget_percent"]) for a in result["arms"]}
        present = {a for a, _ in arms}
        assert arms == set(itertools.product(present, BUDGETS)) and {"core", "random"} <= present
        reference = gold_mean(result["golds"])
        control_path, control = find_controls(job)
        entries = [("original", 0, result["original"]), ("gold_mean", 0, reference)]
        entries += [(f"gold_{i}", 0, g["metrics"]) for i, g in enumerate(result["golds"])]
        entries += [(a["arm"], a["parameter_budget_percent"], a["metrics"]) for a in result["arms"]]
        if control is not None:
            entries += [(*control_arm(a["arm"]), a["metrics"]) for a in control["arms"]]
        keys = [(arm, budget) for arm, budget, _ in entries]
        assert len(keys) == len(set(keys)), f"Duplicate arms for seed {seed}"
        for arm, budget, metrics in entries:
            row = flatten(seed, arm, budget, metrics, result["original"], reference)
            rows.append(row)
            if arm != "gold_mean":
                for group, count in zip(GROUPS, (100, 400, 9500, 10000)):
                    assert metrics[group]["count"] == count
                assert metrics["membership"]["members"] == 500 and metrics["membership"]["nonmembers"] == 100
                for metric in ("accuracy", "loss", "margin"):
                    weighted = sum(metrics[g][metric] * n for g, n in zip(GROUPS[:3], (100, 400, 9500))) / 10000
                    assert math.isclose(weighted, metrics["overall"][metric], abs_tol=3e-6)
                for class_id, values in metrics["per_fine_class"].items():
                    class_ref = {m: statistics.mean(g["metrics"]["per_fine_class"][class_id][m] for g in result["golds"])
                                 for m in ("accuracy", "margin", "loss")}
                    fine.append({"seed": seed, "arm": arm, "budget_percent": budget, "fine_id": int(class_id),
                                 "fine_class": protocol["fine_classes"][int(class_id)],
                                 "accuracy_percent": values["accuracy"] * 100,
                                 "accuracy_delta_gold_pp": (values["accuracy"] - class_ref["accuracy"]) * 100,
                                 "margin": values["margin"], "margin_delta_gold": values["margin"] - class_ref["margin"],
                                 "loss": values["loss"]})
        for i, gold in enumerate(result["golds"]):
            golds_raw.append({"seed": seed, "gold_index": i, "initialization_seed": gold["initialization_seed"],
                              "training_seed": gold["training_seed"],
                              "training_seconds": gold.get("training_seconds"),
                              "target_margin": gold["metrics"]["target"]["margin"],
                              "target_accuracy_percent": gold["metrics"]["target"]["accuracy"] * 100,
                              "loss_auc": gold["metrics"]["membership"]["loss_auc"]})
        records, how = checkpoint_audit(experiment, result)
        audit.extend(records)
        status.append({"seed": seed, "checkpoint_audit": how, "controls": str(control_path) if control_path else None,
                       "timings_recorded": "stage_timings" in result})
        masks = load_masks(experiment)
        if masks is not None:
            overlaps.extend(mask_overlaps(seed, masks))
        if control is not None:
            for item in control["mask_comparisons"]:
                rank_comparisons.append({"seed": seed, **{k: item.get(k) for k in (
                    "rank", "previous_rank", "budget", "comparison", "shared_fraction", "jaccard")}})
        if "stage_timings" in result:
            t, score = result["stage_timings"], result["stage_timings"]["score"]
            gold_seconds = statistics.mean(g["training_seconds"] for g in result["golds"])
            core10 = next(a for a in result["arms"] if a["arm"] == "core" and a["parameter_budget_percent"] == 10)
            method = score["subspace_seconds"] + score["target_request_seconds"] + core10["recovery_seconds"]
            timings.append({"seed": seed, "gold_training_seconds_mean": gold_seconds,
                            "original_training_seconds": t["original_training_seconds"],
                            "subspace_seconds": score["subspace_seconds"],
                            "target_request_seconds": score["target_request_seconds"],
                            "core10_recovery_seconds": core10["recovery_seconds"],
                            "rtr_per_request": (score["target_request_seconds"] + core10["recovery_seconds"]) / gold_seconds,
                            "rtr_including_subspace": method / gold_seconds})
        grads = result.get("stationarity") or (json.loads((experiment / "stationarity.json").read_text())
                                               if (experiment / "stationarity.json").exists() else None)
        if grads:
            stationarity.append({"seed": seed, **{k: v for k, v in grads.items() if k != "batchnorm"}})
        diag_file = json.loads((experiment / "score_diagnostics.json").read_text())
        diag = diag_file["diagnostics"]
        spectra = {f"source_rank_{k}": v for k, v in b_spectrum(diag_file["evals_H"], diag_file["projected_target"]).items()}
        diagnostics.append({"seed": seed, "rank": diag["requested_rank"], "retained_rank": diag["retained_rank"],
                            "converged": diag["subspace"]["converged"], "power_iterations": diag["subspace"]["power_iterations"],
                            "max_relative_residual": max(diag["retained_eigenpair_relative_residuals"]),
                            "sum_rule_relative_error": diag["sum_rule_relative_error"],
                            "score_seconds": diag["elapsed_seconds"],
                            "core_swap_log_score_correlation": diag["control"]["log_score_correlation"],
                            "core_swap_top10_overlap": diag["control"]["top_set_overlap"],
                            "core_cross_log_score_correlation": diag.get("cross_control", {}).get("log_score_correlation"),
                            "core_cross_top10_overlap": diag.get("cross_control", {}).get("top_set_overlap"),
                            **spectra})
        if control is not None:
            for rank, scoring in control["scoring"].items():
                if rank == "diagonal":
                    continue
                d = scoring["diagnostics"]
                diagnostics.append({"seed": seed, "rank": int(rank), "retained_rank": d["retained_rank"],
                                    "converged": d["subspace"]["converged"], "power_iterations": d["subspace"]["power_iterations"],
                                    "max_relative_residual": max(d["retained_eigenpair_relative_residuals"]),
                                    "sum_rule_relative_error": d["sum_rule_relative_error"],
                                    "score_seconds": d["elapsed_seconds"],
                                    "core_swap_log_score_correlation": d["control"]["log_score_correlation"],
                                    "core_swap_top10_overlap": d["control"]["top_set_overlap"],
                                    "core_cross_log_score_correlation": d.get("cross_control", {}).get("log_score_correlation"),
                                    "core_cross_top10_overlap": d.get("cross_control", {}).get("top_set_overlap"),
                                    **{f"source_rank_{k}": v for k, v in b_spectrum(scoring["evals_H"], scoring["projected_target"]).items()}})
        files = [path, experiment / "protocol.json", experiment / "score_diagnostics.json"]
        files += [job / n for n in ("protocol.json", "source_revision.txt", "source.diff", "source_status.txt")]
        files += [control_path] if control_path else []
        for file in files:
            if file.exists():
                manifest.append({"path": os.path.relpath(file, ROOT), "bytes": file.stat().st_size,
                                 "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    assert len(variants) == 1, f"Mixed training variants in one input root: {variants}"
    variant = variants.pop()
    seeds = sorted(seen)
    # Pilot seeds lack the main-run cross arm: summaries report n per cell.
    arms = [a for a in MAIN_ARMS if any(r["arm"] == a for r in rows)]
    metrics = [k for k in rows[0] if k not in ("seed", "arm", "budget_percent")]
    summary = []
    for arm, budget in sorted({(r["arm"], r["budget_percent"]) for r in rows}, key=lambda k: (str(k[0]), k[1])):
        selected = [r for r in rows if r["arm"] == arm and r["budget_percent"] == budget]
        for metric in metrics:
            summary.append({"arm": arm, "budget_percent": budget, "metric": metric,
                            **describe([r[metric] for r in selected])})
    primary_rows, primary = primary_contrast(rows, seeds, variant)
    endpoints = (PRIMARY, "target_margin_gap_closure", "apple_specific_accuracy_fraction",
                 "siblings_accuracy_delta_gold_pp", "retained_accuracy_delta_gold_pp",
                 "other_superclasses_margin", "loss_auc_distance_half")
    pairs = [("core", other, b) for b in BUDGETS for other in ("swap", "cross", "leverage", "random", "diagonal")]
    ranks = sorted({int(a.rsplit("_r", 1)[1]) for a in {r["arm"] for r in rows} if "_r" in a})
    pairs += [(f"core_r{k}", f"{other}_r{k}", b) for k in ranks for b in BUDGETS for other in ("swap", "cross")]
    pairs += [("core", "finetune_core", 10.0)]
    comparisons = paired_differences(rows, seeds, pairs, endpoints)
    for name, data in (("per_seed.csv", rows), ("summary.csv", summary), ("paired_differences.csv", comparisons),
                       ("primary_contrast_per_seed.csv", primary_rows), ("per_fine_class.csv", fine),
                       ("gold_models.csv", golds_raw), ("spectral_diagnostics.csv", diagnostics),
                       ("mask_overlaps.csv", overlaps), ("rank_mask_comparisons.csv", rank_comparisons),
                       ("timings.csv", timings), ("stationarity.csv", stationarity), ("checkpoint_audit.csv", audit)):
        write_csv(name, data)
    (OUT / "primary_contrast.json").write_text(json.dumps(primary, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "inputs": manifest, "seed_status": status, "variant": variant,
        "pilot_seeds_exploratory": list(PILOT_SEEDS),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__, "torch": torch.__version__},
        "unit_of_replication": "original-training seed; gold retrains nested within seed",
        "ci": "pointwise Student t interval over seeds; normal seed differences assumed for paired contrasts",
        "tests": "only the pre-specified primary contrast is tested; all other contrasts are descriptive"},
        indent=2) + "\n")
    figures(rows, summary, fine, seeds, arms)
    tables(summary, arms, seeds)
    confirmatory = primary["confirmatory"]
    verdict = (f"p={confirmatory['p_two_sided']:.3g}, mean difference {confirmatory['mean']:+.3f} "
               f"[{confirmatory['ci95_low']:+.3f}, {confirmatory['ci95_high']:+.3f}] over {confirmatory['n']} seeds"
               if "p_two_sided" in confirmatory else confirmatory["status"])
    print(f"PASS: {len(seeds)} seeds ({variant}), arms {arms}; checkpoint audits "
          f"{sum(s['checkpoint_audit'] != 'unavailable' for s in status)}/{len(seeds)}; "
          f"primary contrast: {verdict}; summaries saved to {OUT}")


if __name__ == "__main__":
    main()
