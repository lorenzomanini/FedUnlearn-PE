"""Read-only, reproducible audit of saved CIFAR predictions and LiRA statistics.

Run from the repository root with .venv/bin/python
validation/accuracy_audit/analyze_saved_cifar.py. No training or downloads occur.
The input pickles are the experiment's own trusted local artifacts.
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "stat_tests" / "CIFAR"
OUTPUT = Path(__file__).resolve().parent


def read_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def fit_gaussian(scores, membership):
    counts = membership.sum(axis=0)
    assert np.all(counts >= 2)
    mean = np.where(membership, scores, 0).sum(axis=0) / counts
    variance = np.where(membership, (scores - mean) ** 2, 0).sum() / (counts - 1).sum()
    return mean, max(float(np.sqrt(variance)), 1e-6)


def summarize(rows, group_fields):
    groups = {}
    for row in rows:
        group = tuple(row[field] for field in group_fields)
        groups.setdefault(group, []).append(row)
    output = []
    for group, members in groups.items():
        result = dict(zip(group_fields, group))
        result["repetitions"] = len(members)
        for name in members[0]:
            if name in group_fields or name == "repetition":
                continue
            values = [row[name] for row in members]
            if isinstance(values[0], (float, int, np.number)):
                result[name + "_mean"] = float(np.mean(values))
                result[name + "_std"] = float(np.std(values))
        output.append(result)
    return output


def save_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    labels = {name: np.asarray(value) for name, value in read_pickle(INPUT / "labels.pkl").items()}
    clients = read_pickle(INPUT / "clients_indices.pkl")
    configuration = read_pickle(INPUT / "init_params.pkl")
    cases = read_pickle(INPUT / "test_params.pkl")
    target = np.asarray(clients[configuration["target_client"]])
    bank = dict(np.load(INPUT / "lira_shadow_bank.npz"))
    membership = bank["candidate_membership"]
    candidates = bank["candidate_complete_indices"]
    scores = bank["scores"].astype(np.float64)
    shadow_membership = bank["shadow_membership"]
    assert np.isfinite(scores).all()
    assert np.unique(candidates).size == candidates.size
    assert set(candidates[membership]) == set(target)
    mean_in, std_in = fit_gaussian(scores, shadow_membership)
    mean_out, std_out = fit_gaussian(scores, ~shadow_membership)

    def privacy_metrics(values):
        values = np.asarray(values, dtype=np.float64)
        assert values.shape == membership.shape and np.isfinite(values).all()
        llr = -np.log(std_in) - 0.5 * ((values - mean_in) / std_in) ** 2
        llr += np.log(std_out) + 0.5 * ((values - mean_out) / std_out) ** 2
        fpr, tpr, _ = roc_curve(membership, llr)
        return {
            "auc": float(roc_auc_score(membership, llr)),
            "tpr_at_fpr_0_001": float(tpr[fpr <= 0.001].max()),
            "tpr_at_fpr_0_01": float(tpr[fpr <= 0.01].max()),
        }

    utility_rows, privacy_rows, timing_rows, diagnostic_rows = [], [], [], []
    for repetition in range(configuration["num_tests"]):
        directory = INPUT / f"test_{repetition}"
        diagnostics = read_pickle(directory / "score_diagnostics.pkl")
        training = np.asarray(diagnostics["training_indices"])
        retained = np.setdiff1d(training, target)
        heldout = np.setdiff1d(np.arange(len(labels["train"])), training)
        assert len(training) == len(np.unique(training)) == 45500
        assert len(retained) == 40500 and len(target) == 5000 and len(heldout) == 4500
        assert set(target).issubset(set(training))
        assert not set(candidates[~membership]).intersection(training)
        extra = read_pickle(directory / "extra_results.pkl")
        stage = read_pickle(directory / "stage_timings.pkl")
        initial_test = read_pickle(directory / "initial_eval_test_results.pkl")
        initial_train = read_pickle(directory / "initial_eval_train_results.pkl")
        initial_lira = read_pickle(directory / "initial_lira_results.pkl")
        eval_test = dict(np.load(directory / "eval_test_results.npz"))
        eval_train = dict(np.load(directory / "eval_train_results.npz"))
        eval_lira = dict(np.load(directory / "eval_lira_results.npz"))
        lira_cases = read_pickle(directory / "lira_case_indices.pkl")
        assert extra["case_index"] == list(range(len(cases)))
        assert lira_cases == [0, 5, 9]
        assert all(array.shape[0] == len(cases) for array in eval_test.values())
        assert all(array.shape[0] == len(cases) for array in eval_train.values())
        assert all(array.shape[0] == len(lira_cases) for array in eval_lira.values())

        def utility_metrics(test, train):
            assert test["pred"].shape == labels["test"].shape
            assert train["pred"].shape == labels["train"].shape
            assert np.isfinite(test["loss"]).all() and np.isfinite(train["loss"]).all()
            result = {
                "test_accuracy_pct": float(np.mean(test["pred"] == labels["test"]) * 100),
                "test_disagreement_from_gold_pct": float(np.mean(test["pred"] != initial_test["shadow_out"]["pred"]) * 100),
                "test_true_class_margin": float(test["loss"].mean()),
            }
            for name, indices in [("retain_train", retained), ("forget", target), ("retain_heldout", heldout)]:
                result[name + "_accuracy_pct"] = float(np.mean(train["pred"][indices] == labels["train"][indices]) * 100)
                result[name + "_true_class_margin"] = float(train["loss"][indices].mean())
                result[name + "_disagreement_from_gold_pct"] = float(np.mean(train["pred"][indices] != initial_train["shadow_out"]["pred"][indices]) * 100)
            return result

        for name in initial_test:
            model = "original" if name == "trained" else "gold"
            base = {"repetition": repetition, "case_index": -1, "model": model, "score_mass_pct": 0.0, "parameters_reset_pct": 0.0}
            utility_rows.append({**base, **utility_metrics(initial_test[name], initial_train[name])})
            privacy_rows.append({**base, **privacy_metrics(initial_lira[name])})

        for case_index, case in enumerate(cases):
            base = {"repetition": repetition, "case_index": case_index, "score_mass_pct": case["unlearning_percentage"], "parameters_reset_pct": extra["reset_params_percentage"][case_index]}
            for name in ("reset", "retrained", "random_reset", "random_retrained"):
                test = {field: eval_test[name + "__" + field][case_index] for field in ("pred", "loss")}
                train = {field: eval_train[name + "__" + field][case_index] for field in ("pred", "loss")}
                utility_rows.append({**base, "model": name, **utility_metrics(test, train)})
                if case_index in lira_cases:
                    privacy_rows.append({**base, "model": name, **privacy_metrics(eval_lira[name][lira_cases.index(case_index)])})
                if case_index == 0:
                    assert np.array_equal(test["pred"], initial_test["trained"]["pred"])
                    assert np.array_equal(train["pred"], initial_train["trained"]["pred"])
            timing_rows.append({
                **base,
                "score_seconds": stage["score_seconds"],
                "selection_seconds": extra["selection_seconds"][case_index],
                "recovery_seconds": extra["recovery_seconds"][case_index],
                "unlearning_with_score_seconds": extra["unlearning_with_score_seconds"][case_index],
                "gold_retraining_seconds": stage["gold_retraining_seconds"],
                "cost_fraction_of_gold": extra["unlearning_with_score_seconds"][case_index] / stage["gold_retraining_seconds"],
                "random_baseline_seconds": extra["random_baseline_seconds"][case_index],
                "evaluation_seconds": extra["evaluation_seconds"][case_index],
                "lira_evaluation_seconds": extra["lira_evaluation_seconds"][case_index],
            })
        inner = diagnostics["diagnostics"]
        residuals = inner["retained_eigenpair_relative_residuals"]
        diagnostic_rows.append({
            "repetition": repetition,
            "retained_rank": inner["retained_rank"],
            "converged": inner["subspace"]["converged"],
            "eigenpair_residual_min": min(residuals),
            "eigenpair_residual_max": max(residuals),
            "sum_rule_relative_error": inner["sum_rule_relative_error"],
            "full_samples": diagnostics["full_samples"],
            "target_samples": diagnostics["target_samples"],
            "score_seconds": stage["score_seconds"],
            "gold_retraining_seconds": stage["gold_retraining_seconds"],
        })

    for name, rows in [("utility", utility_rows), ("privacy", privacy_rows), ("timings", timing_rows)]:
        save_csv("cifar_" + name + "_per_run.csv", rows)
        groups = ["case_index", "model"] if name != "timings" else ["case_index"]
        save_csv("cifar_" + name + "_summary.csv", summarize(rows, groups))
    save_csv("cifar_score_diagnostics.csv", diagnostic_rows)
    metadata = {
        "input": str(INPUT),
        "checks_passed": ["no missing sweep cases", "same LiRA cases [0,5,9] in each repetition", "finite stored outputs", "zero-reset predictions equal original", "LiRA positives are all and only target records", "LiRA negatives excluded from original training", "45500 original training / 40500 retained training / 5000 forget / 4500 retained holdout"],
        "notes": [
            "All *_pct values are percentages; AUC/TPR use fractions.",
            "Standard deviations use population convention (ddof=0) over three repetitions.",
            "retrained means one epoch recovering only reset coordinates; gold means 40 epochs from fresh initialization.",
            "Artifact field loss is true-class logit margin, not cross-entropy; CE cannot be reconstructed from it.",
            "Prediction disagreement from gold is a behavioral proxy, not parameter or probability distance.",
            "No model checkpoints/full logits saved; cannot measure parameter distance or rerun curvature/recovery on the identical trained model.",
            "Single-deletion cost includes score construction, parameter selection, reset/recovery; excludes baseline training, random comparator, reporting evaluation and shared LiRA shadows.",
            "LiRA uses 8 shadows, 4 IN and 4 OUT per record, a shared fitted bank, and a globally pooled IN variance and OUT variance.",
            "Retained heldout predictions cover all 4500 original-training exclusions, including LiRA candidate nonmembers removed from validation.",
        ],
        "lira_candidate_members": int(membership.sum()),
        "lira_candidate_nonmembers": int((~membership).sum()),
        "lira_nonmembers_train_holdout": int(np.sum(candidates[~membership] < len(labels["train"]))),
        "lira_nonmembers_test": int(np.sum(candidates[~membership] >= len(labels["train"]))),
        "lira_shadow_in_counts": np.unique(shadow_membership.sum(axis=0)).tolist(),
        "lira_global_std_in": std_in,
        "lira_global_std_out": std_out,
        "suite_timings": read_pickle(INPUT / "stage_timings.pkl"),
    }
    (OUTPUT / "cifar_audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
