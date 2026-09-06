import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_curve, auc
from experiments import persistence

def merge_initial_results(accumulated_results):
    merged_results = {}
    for key in accumulated_results[0].keys():
        merged_results[key] = [[accumulated_results[j][key] for j in range(len(accumulated_results))]]
    return merged_results

def merge_results(accumulated_results):
    merged_results = {}
    for key in accumulated_results[0].keys():
        merged_results[key] = []
        first_entry = accumulated_results[0][key]
        if isinstance(first_entry, dict) and "pred" in first_entry:
            num_tests = first_entry["pred"].shape[0]
            for i in range(num_tests):
                merged_results[key].append(
                    [
                        {
                            "pred": accumulated_results[j][key]["pred"][i],
                            "loss": accumulated_results[j][key]["loss"][i]
                        }
                        for j in range(len(accumulated_results))
                    ]
                )
        else:
            for i in range(len(first_entry)):
                merged_results[key].append(
                    [accumulated_results[j][key][i] for j in range(len(accumulated_results))]
                )
    return merged_results

def compute_accuracy(outputs, labels):
    outputs = np.array(outputs)
    predictions = outputs if outputs.ndim == 1 else np.argmax(outputs, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def compute_metrics(res, model_key):
    mia_type_key = 'mia_nn'
    mia_result_key = 'accuracy'

    model_test_acc = res[model_key + '_test_accuracy']
    model_target_acc = res[model_key + '_target_accuracy']

    benchmark_test_acc = res['benchmark_test_accuracy']
    benchmark_target_acc = res['benchmark_target_accuracy']

    trained_test_acc = res['trained_test_accuracy']
    trained_target_acc = res['trained_target_accuracy']

    nta = model_test_acc / benchmark_test_acc
    nfs = (model_target_acc - trained_target_acc) / (benchmark_target_acc - trained_target_acc) if benchmark_target_acc != trained_target_acc else 0

    try:
        model_mia_acc = res[model_key + '_' + mia_type_key][mia_result_key]
        benchmark_mia_acc = res['benchmark_' + mia_type_key][mia_result_key]
        trained_mia_acc = res['trained_' + mia_type_key][mia_result_key]
        nfs_mia = (model_mia_acc - trained_mia_acc) / (benchmark_mia_acc - trained_mia_acc) if benchmark_mia_acc != trained_mia_acc else 0
    except KeyError:
        nfs_mia = None

    return nta, nfs, nfs_mia

def compute_avg_std(subtest_results, key):
    avg = []
    std = []
    for subtest_result in subtest_results:
        values = [test_result[key] for test_result in subtest_result]
        avg.append(np.mean(values))
        std.append(np.std(values))
    return avg, std

def find_best_reset_indices(subtest_results, distribution_type):
    best_reset_idxs = []
    for i in range(len(subtest_results[0])):
        reset_nta = [subtest_results[j][i]['reset_nta'] for j in range(len(subtest_results))]
        if distribution_type == "random":
            reset_nfs_mia = [subtest_results[j][i]['reset_nfs_mia'] for j in range(len(subtest_results))]
            best_reset_idx = np.argmin([np.sqrt((1 - nfs_mia)**2 + (1 - nta)**2) for nfs_mia, nta in zip(reset_nfs_mia, reset_nta)])
            best_reset_idxs.append(best_reset_idx)
        else:
            reset_nfs = [subtest_results[j][i]['reset_nfs'] for j in range(len(subtest_results))]
            best_reset_idx = np.argmin([np.sqrt((1 - nfs)**2 + (1 - nta)**2) for nfs, nta in zip(reset_nfs, reset_nta)])
            best_reset_idxs.append(best_reset_idx)
    return best_reset_idxs

def find_best_backdoor_reset_indices(subtest_results):
    best_reset_idxs = []
    for i in range(len(subtest_results[0])):
        reset_nta = [subtest_results[j][i]['reset_nta'] for j in range(len(subtest_results))]
        reset_nfs = [subtest_results[j][i]['reset_nfs'] for j in range(len(subtest_results))]
        best_reset_idx = np.argmin([np.sqrt((1 - nfs)**2 + (1 - nta)**2) for nfs, nta in zip(reset_nfs, reset_nta)])
        best_reset_idxs.append(best_reset_idx)
    return best_reset_idxs

def apply_backdoor_metric_aliases(tests_results):
    for i in range(len(tests_results)):
        for j in range(len(tests_results[i])):
            tests_results[i][j]['trained_test_accuracy'] = tests_results[i][j]['trained_clean_backdoor_accuracy']
            tests_results[i][j]['benchmark_test_accuracy'] = tests_results[i][0]['benchmark_clean_backdoor_accuracy']
            tests_results[i][j]['reset_test_accuracy'] = tests_results[i][j]['reset_clean_backdoor_accuracy']
            tests_results[i][j]['random_reset_test_accuracy'] = tests_results[i][j]['random_reset_clean_backdoor_accuracy']
            tests_results[i][j]['retrained_test_accuracy'] = tests_results[i][j]['retrained_clean_backdoor_accuracy']
            tests_results[i][j]['random_retrained_test_accuracy'] = tests_results[i][j]['random_retrained_clean_backdoor_accuracy']

            tests_results[i][j]['trained_target_accuracy'] = tests_results[i][j]['trained_poisoned_backdoor_accuracy']
            tests_results[i][j]['benchmark_target_accuracy'] = tests_results[i][0]['benchmark_poisoned_backdoor_accuracy']
            tests_results[i][j]['reset_target_accuracy'] = tests_results[i][j]['reset_poisoned_backdoor_accuracy']
            tests_results[i][j]['random_reset_target_accuracy'] = tests_results[i][j]['random_reset_poisoned_backdoor_accuracy']
            tests_results[i][j]['retrained_target_accuracy'] = tests_results[i][j]['retrained_poisoned_backdoor_accuracy']
            tests_results[i][j]['random_retrained_target_accuracy'] = tests_results[i][j]['random_retrained_poisoned_backdoor_accuracy']
    return tests_results

def get_predictions(outputs):
    if isinstance(outputs, dict) and "pred" in outputs:
        return np.array(outputs["pred"])
    outputs = np.array(outputs)
    return outputs if outputs.ndim == 1 else np.argmax(outputs, axis=1)

def compute_accuracies(merged_results, labels, subset=None):
    labels = np.array(labels)
    idx = slice(None) if subset is None else subset
    accuracies = {}
    for model_key in merged_results.keys():
        accuracies[model_key] = []
        for test_outputs in merged_results[model_key]:
            test_accuracies = []
            for outputs in test_outputs:
                predictions = get_predictions(outputs)
                acc = np.mean(predictions[idx] == labels[idx])
                test_accuracies.append(acc)
            accuracies[model_key].append(test_accuracies)
    return accuracies

def compute_shadow_losses(merged_initial_results, labels, subset=None):
    idx = slice(None) if subset is None else subset
    labels_tensor = torch.tensor(labels)
    loss_fn = CrossEntropyLoss(reduction='none')
    losses = {}
    for model_key in merged_initial_results.keys():
        losses_list= []
        for outputs in merged_initial_results[model_key][0]:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = np.array(outputs["loss"])[idx]
                loss = torch.tensor(loss)
            else:
                outputs_tensor = torch.tensor(outputs)
                loss = loss_fn(outputs_tensor[idx], labels_tensor[idx])
            losses_list.append(loss)
        losses_tensor = torch.stack(losses_list, dim=0)
        losses[model_key] = losses_tensor.numpy()
    return losses

def compute_shadow_losses_dists(losses_members, losses_nonmembers, global_var=False):
    member_dists = {}
    nonmember_dists = {}
    for model_key in losses_members.keys():
        avg_member = np.mean(losses_members[model_key], axis=0)
        avg_nonmember = np.mean(losses_nonmembers[model_key], axis=0)
        if global_var:
            losses = np.concatenate([losses_members[model_key], losses_nonmembers[model_key]], axis=1)
            residuals = losses - np.mean(losses, axis=0, keepdims=True)
            degrees_of_freedom = residuals.size - residuals.shape[1]
            if degrees_of_freedom <= 0:
                raise ValueError("At least two shadow observations are required per example.")
            std = np.sqrt(np.sum(residuals ** 2) / degrees_of_freedom)
            std_member = std
            std_nonmember = std
        else:
            std_member = np.std(losses_members[model_key], axis=0, ddof=1)
            std_nonmember = np.std(losses_nonmembers[model_key], axis=0, ddof=1)
        member_dists[model_key] = {
            'avg': avg_member,
            'std': std_member
        }
        nonmember_dists[model_key] = {
            'avg': avg_nonmember,
            'std': std_nonmember
        }

    return member_dists, nonmember_dists

def log_gaussian_pdf(x, mu, sigma):
    sigma = np.asarray(sigma)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Gaussian standard deviations must be finite and positive.")
    coeff = - np.log(sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff + exponent


def fit_lira_distributions(shadow_scores, shadow_membership, global_var=True, min_std=1e-6):
    """Fit the paper's per-example IN/OUT Gaussian distributions."""
    scores = np.asarray(shadow_scores, dtype=np.float64)
    membership = np.asarray(shadow_membership, dtype=bool)
    if scores.ndim != 2 or scores.shape != membership.shape:
        raise ValueError("Shadow scores and membership mask must be equally shaped 2D arrays.")

    def fit(mask):
        counts = mask.sum(axis=0)
        if np.any(counts < 2):
            raise ValueError("Every example needs at least two IN and two OUT shadows.")
        means = np.sum(np.where(mask, scores, 0.0), axis=0) / counts
        residuals = np.where(mask, scores - means, 0.0)
        squared = np.sum(residuals ** 2, axis=0)
        if global_var:
            variance = squared.sum() / np.sum(counts - 1)
        else:
            variance = squared / (counts - 1)
        std = np.sqrt(np.maximum(variance, min_std ** 2))
        return {"avg": means, "std": std}

    return fit(membership), fit(~membership)


def compute_online_lira_scores(losses_in, losses_out, merged_results):
    lira_scores = {}
    for model_key, model_results in merged_results.items():
        lira_scores[model_key] = []
        for test_outputs in model_results:
            lira_scores[model_key].append([
                log_gaussian_pdf(np.asarray(outputs), losses_in["avg"], losses_in["std"])
                - log_gaussian_pdf(np.asarray(outputs), losses_out["avg"], losses_out["std"])
                for outputs in test_outputs
            ])
    return lira_scores


def split_lira_scores(lira_scores, candidate_membership):
    membership = np.asarray(candidate_membership, dtype=bool)
    members = {}
    nonmembers = {}
    for model_key, model_results in lira_scores.items():
        members[model_key] = [
            [np.asarray(scores)[membership] for scores in test_outputs]
            for test_outputs in model_results
        ]
        nonmembers[model_key] = [
            [np.asarray(scores)[~membership] for scores in test_outputs]
            for test_outputs in model_results
        ]
    return members, nonmembers
    
def compute_lira_scores(losses_in, losses_out, merged_results, labels, subset=None):
    labels_tensor = torch.tensor(labels)
    idx = slice(None) if subset is None else subset
    loss_fn = CrossEntropyLoss(reduction='none')
    lira_scores = {}
    for model_key in merged_results.keys():
        lira_scores[model_key] = []
        for test_outputs in merged_results[model_key]:
            test_lira_scores = []
            for outputs in test_outputs:
                if isinstance(outputs, dict) and "loss" in outputs:
                    losses = np.array(outputs["loss"])[idx]
                else:
                    outputs_tensor = torch.tensor(outputs)
                    losses = loss_fn(outputs_tensor[idx], labels_tensor[idx]).numpy()
                lira = log_gaussian_pdf(losses, losses_in['avg'], losses_in['std']) - log_gaussian_pdf(losses, losses_out['avg'], losses_out['std'])
                test_lira_scores.append(lira)

            lira_scores[model_key].append(test_lira_scores)
    return lira_scores

def compute_roc_curves(lira_member_scores, lira_nonmember_scores):
    roc_curves = {}
    for model_key in lira_member_scores.keys():
        roc_curves[model_key] = {'fpr': [], 'tpr': [], 'auc': []}
        for test_idx in range(len(lira_member_scores[model_key])):
            test_roc_fprs = []
            test_roc_tprs = []
            test_roc_aucs = []
            for iter_idx in range(len(lira_member_scores[model_key][test_idx])):
                member_scores = lira_member_scores[model_key][test_idx][iter_idx]
                nonmember_scores = lira_nonmember_scores[model_key][test_idx][iter_idx]
                member_scores = member_scores[np.isfinite(member_scores)]
                nonmember_scores = nonmember_scores[np.isfinite(nonmember_scores)]
                if not len(member_scores) or not len(nonmember_scores):
                    raise ValueError("LiRA ROC requires finite member and non-member scores.")
                all_scores = np.concatenate([member_scores, nonmember_scores], axis=0)
                labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
                fpr, tpr, _ = roc_curve(labels, all_scores)
                roc_auc = auc(fpr, tpr)
                test_roc_fprs.append(fpr)
                test_roc_tprs.append(tpr)
                test_roc_aucs.append(roc_auc)
            roc_curves[model_key]['fpr'].append(test_roc_fprs)
            roc_curves[model_key]['tpr'].append(test_roc_tprs)
            roc_curves[model_key]['auc'].append(test_roc_aucs)
    return roc_curves

def compute_tpr_at_fpr(roc_curve_data, target_fpr=0.01):
    tpr_at_fpr = {}
    for model_key in roc_curve_data.keys():
        tpr_at_fpr[model_key] = []
        for i in range(len(roc_curve_data[model_key]['fpr'])):
            test_tpr_at_fpr = []
            for j in range(len(roc_curve_data[model_key]['fpr'][i])):
                fpr = roc_curve_data[model_key]['fpr'][i][j]
                tpr = roc_curve_data[model_key]['tpr'][i][j]
                idx = np.where(fpr <= target_fpr)[0]
                if len(idx) > 0:
                    tpr_value = tpr[idx[-1]]
                else:
                    tpr_value = 0.0
                test_tpr_at_fpr.append(tpr_value)
            tpr_at_fpr[model_key].append(test_tpr_at_fpr)
    return tpr_at_fpr

def plot_error_bars(x, y, **kwargs):
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    plt.errorbar(x_mean, y_mean, yerr=y_std, **kwargs)

def unpack_eval_results(npz_dict):
    if any("__" in key for key in npz_dict.keys()):
        unpacked = {}
        for key, value in npz_dict.items():
            if "__" not in key:
                continue
            model_key, suffix = key.split("__", 1)
            if suffix not in {"pred", "loss"}:
                continue
            unpacked.setdefault(model_key, {})[suffix] = value
        if unpacked:
            return unpacked
    return npz_dict

def summarize_baseline(values):
    values = np.array(values)
    return float(np.mean(values)), float(np.std(values))

def add_baseline_band(ax, mean, std, label, color):
    ax.axhline(mean, color=color, linewidth=1.6, label=label)
    if std > 0:
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.12)

def plot_metric_vs_unlearning(ax, x_values, metric_dict, label_map, ylabel, title, baselines=None):
    x_values = np.array(x_values)
    if x_values.ndim == 1:
        x_mean = x_values
    else:
        x_mean = np.mean(x_values, axis=1)
    for key, style in label_map.items():
        if key not in metric_dict:
            continue
        y = np.array(metric_dict[key])
        if y.ndim == 1:
            y_mean = y
            y_std = np.zeros_like(y_mean)
        else:
            if y.shape[0] != x_mean.shape[0] and y.shape[1] == x_mean.shape[0]:
                y = y.T
            y_mean = np.mean(y, axis=1)
            y_std = np.std(y, axis=1)
        sort_idx = np.argsort(x_mean)
        ax.errorbar(
            x_mean[sort_idx],
            y_mean[sort_idx],
            yerr=y_std[sort_idx],
            label=style["label"],
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=1.6,
            markersize=5,
            capsize=3
        )
    if baselines:
        for baseline in baselines:
            add_baseline_band(ax, baseline["mean"], baseline["std"], baseline["label"], baseline["color"])
    ax.set_title(title)
    ax.set_xlabel("Unlearning Percentage")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

_merge_initial_results = merge_initial_results
_merge_results = merge_results
_unpack_eval_results = unpack_eval_results
_get_predictions = get_predictions
_compute_accuracies = compute_accuracies
_compute_shadow_losses = compute_shadow_losses
_compute_shadow_losses_dists = compute_shadow_losses_dists
_log_gaussian_pdf = log_gaussian_pdf
_compute_lira_scores = compute_lira_scores
_fit_lira_distributions = fit_lira_distributions
_compute_online_lira_scores = compute_online_lira_scores
_split_lira_scores = split_lira_scores
_compute_roc_curves = compute_roc_curves
_compute_tpr_at_fpr = compute_tpr_at_fpr
_summarize_baseline = summarize_baseline
_add_baseline_band = add_baseline_band
_plot_metric_vs_unlearning = plot_metric_vs_unlearning


def plot_experiment_results(test_path, num_tests=None, target_fpr=0.001):
    """Load saved experiment results and produce diagnostic plots.

    Parameters
    ----------
    test_path : str
        Path to the experiment directory produced by run_repeated_tests.
    num_tests : int or None
        Number of test iterations to load. If None, inferred from saved init_params.pkl.
    target_fpr : float
        FPR threshold for TPR@FPR computation.
    """
    import matplotlib.pyplot as plt

    with open(os.path.join(test_path, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(test_path, "clients_indices.pkl"), "rb") as f:
        client_indices = pickle.load(f)
    shadow_bank_path = os.path.join(test_path, persistence.LIRA_SHADOW_BANK)
    if not os.path.exists(shadow_bank_path):
        raise FileNotFoundError(
            "This experiment predates the explicit online-LiRA shadow bank; regenerate it."
        )
    with open(shadow_bank_path, "rb") as f:
        shadow_bank = dict(np.load(f))

    saved_params = {}
    params_path = os.path.join(test_path, "init_params.pkl")
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            saved_params = pickle.load(f)
    if num_tests is None:
        num_tests = saved_params.get('num_tests', 1)

    target_client_idx = saved_params.get('target_client', 0)
    target_indices = client_indices[target_client_idx]

    acc_initial_eval_test, acc_initial_eval_train = [], []
    acc_tests_eval_test, acc_tests_eval_train, acc_tests_extra = [], [], []
    initial_lira_results, tests_lira_results = [], []

    for i in range(num_tests):
        iter_path = os.path.join(test_path, f"test_{i}")
        with open(os.path.join(iter_path, "initial_eval_test_results.pkl"), "rb") as f:
            acc_initial_eval_test.append(pickle.load(f))
        with open(os.path.join(iter_path, "initial_eval_train_results.pkl"), "rb") as f:
            acc_initial_eval_train.append(pickle.load(f))
        with open(os.path.join(iter_path, "eval_test_results.npz"), "rb") as f:
            acc_tests_eval_test.append(_unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(iter_path, "eval_train_results.npz"), "rb") as f:
            acc_tests_eval_train.append(_unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(iter_path, "extra_results.pkl"), "rb") as f:
            acc_tests_extra.append(pickle.load(f))
        with open(os.path.join(iter_path, persistence.INITIAL_LIRA_RESULTS), "rb") as f:
            initial_lira_results.append(pickle.load(f))
        with open(os.path.join(iter_path, persistence.EVAL_LIRA_RESULTS), "rb") as f:
            tests_lira_results.append(dict(np.load(f)))

    initial_eval_test = _merge_initial_results(acc_initial_eval_test)
    initial_eval_train = _merge_initial_results(acc_initial_eval_train)
    unlearned_eval_test = _merge_results(acc_tests_eval_test)
    unlearned_eval_train = _merge_results(acc_tests_eval_train)
    unlearned_extra = _merge_results(acc_tests_extra)

    initial_test_accuracies = _compute_accuracies(initial_eval_test, labels["test"])
    unlearned_test_accuracies = _compute_accuracies(unlearned_eval_test, labels["test"])

    losses_in, losses_out = _fit_lira_distributions(
        shadow_bank["scores"],
        shadow_bank["shadow_membership"],
        global_var=bool(saved_params.get("lira_global_variance", True)),
    )
    candidate_membership = shadow_bank["candidate_membership"]
    unlearned_lira = _compute_online_lira_scores(
        losses_in, losses_out, _merge_results(tests_lira_results)
    )
    initial_lira = _compute_online_lira_scores(
        losses_in, losses_out, _merge_initial_results(initial_lira_results)
    )
    unlearned_members, unlearned_nonmembers = _split_lira_scores(
        unlearned_lira, candidate_membership
    )
    initial_members, initial_nonmembers = _split_lira_scores(
        initial_lira, candidate_membership
    )
    unlearned_roc_curves = _compute_roc_curves(
        unlearned_members, unlearned_nonmembers
    )
    initial_roc_curves = _compute_roc_curves(initial_members, initial_nonmembers)

    unlearned_tpr_at_fpr = _compute_tpr_at_fpr(unlearned_roc_curves, target_fpr=target_fpr)
    initial_tpr_at_fpr = _compute_tpr_at_fpr(initial_roc_curves, target_fpr=target_fpr)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: ROC curves for initial models
    plt.figure(figsize=(8, 6))
    min_rate = 1 / max(1, np.count_nonzero(~candidate_membership))
    for model_key, style in {
        "trained": {"label": "Trained", "linestyle": "-", "color": "tab:blue"},
        "shadow_out": {"label": "Shadow Out", "linestyle": "--", "color": "tab:orange"},
    }.items():
        fpr_list = initial_roc_curves[model_key]['fpr'][0]
        tpr_list = initial_roc_curves[model_key]['tpr'][0]
        for fpr, tpr in zip(fpr_list, tpr_list):
            plt.plot(fpr, tpr, linestyle=style["linestyle"], color=style["color"], alpha=0.3)
        mean_fpr = np.geomspace(min_rate, 1, 200)
        mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fpr_list, tpr_list)], axis=0)
        plt.plot(mean_fpr, mean_tpr, linestyle=style["linestyle"], color=style["color"],
                 label=style["label"], linewidth=2)
    plt.plot([min_rate, 1], [min_rate, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min_rate, 1)
    plt.ylim(min_rate, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Initial Models')
    plt.legend(frameon=False)
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

    accuracy_styles = {
        "reset":           {"label": "Reset",           "linestyle": "-",  "marker": "o"},
        "random_reset":    {"label": "Random Reset",    "linestyle": "--", "marker": "s"},
        "retrained":       {"label": "Retrained",       "linestyle": "-",  "marker": "D"},
        "random_retrained":{"label": "Random Retrained","linestyle": "--", "marker": "^"},
    }

    trained_acc_mean, trained_acc_std = _summarize_baseline(initial_test_accuracies["trained"][0])
    benchmark_acc_mean, benchmark_acc_std = _summarize_baseline(initial_test_accuracies["shadow_out"][0])

    # Plot 2: Test accuracy vs unlearning percentage
    _, ax = plt.subplots(figsize=(9, 5))
    _plot_metric_vs_unlearning(
        ax=ax, x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_test_accuracies, label_map=accuracy_styles,
        ylabel="Test Accuracy", title="Test Accuracy vs Unlearning Percentage",
        baselines=[
            {"mean": trained_acc_mean, "std": trained_acc_std,
             "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_acc_mean, "std": benchmark_acc_std,
             "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    trained_tpr_mean, trained_tpr_std = _summarize_baseline(initial_tpr_at_fpr["trained"][0])
    benchmark_tpr_mean, benchmark_tpr_std = _summarize_baseline(initial_tpr_at_fpr["shadow_out"][0])

    # Plot 3: TPR at FPR vs unlearning percentage
    _, ax = plt.subplots(figsize=(9, 5))
    _plot_metric_vs_unlearning(
        ax=ax, x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_tpr_at_fpr, label_map=accuracy_styles,
        ylabel=f"TPR at FPR={target_fpr*100:.2f}%",
        title=f"TPR at FPR={target_fpr*100:.2f}% vs Unlearning Percentage",
        baselines=[
            {"mean": trained_tpr_mean, "std": trained_tpr_std,
             "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_tpr_mean, "std": benchmark_tpr_std,
             "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    print(f"Trained baseline TPR at FPR={target_fpr*100:.2f}%: "
          f"{trained_tpr_mean:.4f} ± {trained_tpr_std:.4f}")


if __name__ == "__main__":
    plot_experiment_results(
        os.path.join("stat_tests/EXPERIMENTS", "MNIST_pref"), target_fpr=0.001
    )
    raise SystemExit

    plt.style.use("seaborn-v0_8-whitegrid")
    stat_test_name = "MNIST_pref"
    stat_tests_path = "stat_tests/EXPERIMENTS" 
    overload_num_tests = 1  # Set to an integer to override the number of tests

    stat_test_path = os.path.join(stat_tests_path, stat_test_name)
    if not os.path.exists(stat_test_path):
        raise FileNotFoundError(f"Stat test path {stat_test_path} does not exist.")

    with open(os.path.join(stat_test_path, "init_params.pkl"), "rb") as f:
        init_params = pickle.load(f)
    with open(os.path.join(stat_test_path, "test_params.pkl"), "rb") as f:
        test_params = pickle.load(f)
    with open(os.path.join(stat_test_path, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(stat_test_path, "clients_indices.pkl"), "rb") as f:
        client_indices = pickle.load(f)

    target_indices = client_indices[init_params["target_client"]]

    print("Init params:")
    for k, v in init_params.items():
        print(f"{k}: {v}")

    num_tests = init_params["num_tests"]
    if overload_num_tests is not None:
        num_tests = overload_num_tests

    distribution_type = init_params["distribution_type"]

    acc_initial_eval_test = []
    acc_initial_eval_train = []
    acc_tests_eval_test = []
    acc_tests_eval_train = []
    acc_tests_extra = []

    for i in range(num_tests):
        test_path = os.path.join(stat_test_path, f"test_{i}")
        with open(os.path.join(test_path, "initial_eval_test_results.pkl"), "rb") as f:
            acc_initial_eval_test.append(pickle.load(f))
        with open(os.path.join(test_path, "initial_eval_train_results.pkl"), "rb") as f:
            acc_initial_eval_train.append(pickle.load(f))
        with open(os.path.join(test_path, "eval_test_results.npz"), "rb") as f:
            acc_tests_eval_test.append(unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(test_path, "eval_train_results.npz"), "rb") as f:
            acc_tests_eval_train.append(unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(test_path, "extra_results.pkl"), "rb") as f:
            acc_tests_extra.append(pickle.load(f))

    print("Unlearned models keys:", acc_tests_eval_test[0].keys())
    print("Initial models keys:", acc_initial_eval_test[0].keys())
            
    initial_eval_test = merge_initial_results(acc_initial_eval_test)
    initial_eval_train = merge_initial_results(acc_initial_eval_train)
    unlearned_eval_test = merge_results(acc_tests_eval_test)
    unlearned_eval_train = merge_results(acc_tests_eval_train)
    unlearned_extra = merge_results(acc_tests_extra)

    # [model][test][repetition][entry]

    initial_test_accuracies = compute_accuracies(initial_eval_test, labels["test"])
    initial_train_accuracies = compute_accuracies(initial_eval_train, labels["train"])
    initial_target_accuracies = compute_accuracies(initial_eval_train, labels["train"], subset=target_indices)
    unlearned_test_accuracies = compute_accuracies(unlearned_eval_test, labels["test"])
    unlearned_train_accuracies = compute_accuracies(unlearned_eval_train, labels["train"])
    unlearned_target_accuracies = compute_accuracies(unlearned_eval_train, labels["train"], subset=target_indices)

    shadow_target_losses = compute_shadow_losses(initial_eval_train, labels["train"], subset=target_indices)
    shadow_test_losses = compute_shadow_losses(initial_eval_test, labels["test"])
    shadow_target_dists, shadow_test_dists = compute_shadow_losses_dists(shadow_target_losses, shadow_test_losses, global_var=True)


    unlearned_lira_target = compute_lira_scores(
        shadow_target_dists['shadow_in'], shadow_target_dists['shadow_out'], unlearned_eval_train, labels["train"], subset=target_indices)
    unlearned_lira_test = compute_lira_scores(
        shadow_test_dists['shadow_in'], shadow_test_dists['shadow_out'], unlearned_eval_test, labels["test"])
    unlearned_roc_curves = compute_roc_curves(unlearned_lira_target, unlearned_lira_test)

    initial_lira_target = compute_lira_scores(
        shadow_target_dists['shadow_in'], shadow_target_dists['shadow_out'], initial_eval_train, labels["train"], subset=target_indices)
    initial_lira_test = compute_lira_scores(
        shadow_test_dists['shadow_in'], shadow_test_dists['shadow_out'], initial_eval_test, labels["test"])
    initial_roc_curves = compute_roc_curves(initial_lira_target, initial_lira_test)

    target_fpr = 0.01

    unlearned_tpr_at_fpr = compute_tpr_at_fpr(unlearned_roc_curves, target_fpr=target_fpr)
    initial_tpr_at_fpr = compute_tpr_at_fpr(initial_roc_curves, target_fpr=target_fpr)

    # Plot ROC curves for initial models
    plt.figure(figsize=(8, 6))
    for model_key, style in {
        "trained": {"label": "Trained", "linestyle": "-", "color": "tab:blue"},
        "shadow_out": {"label": "Shadow Out", "linestyle": "--", "color": "tab:orange"},
    }.items():
        fpr_list = initial_roc_curves[model_key]['fpr'][0]
        tpr_list = initial_roc_curves[model_key]['tpr'][0]
        for fpr, tpr in zip(fpr_list, tpr_list):
            plt.plot(fpr, tpr, linestyle=style["linestyle"], color=style["color"], alpha=0.3)
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
        plt.plot(mean_fpr, mean_tpr, linestyle=style["linestyle"], color=style["color"], label=style["label"], linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Initial Models')
    plt.legend(frameon=False)
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

    accuracy_styles = {
        "reset": {"label": "Reset", "linestyle": "-", "marker": "o"},
        "random_reset": {"label": "Random Reset", "linestyle": "--", "marker": "s"},
        "retrained": {"label": "Retrained", "linestyle": "-", "marker": "D"},
        "random_retrained": {"label": "Random Retrained", "linestyle": "--", "marker": "^"},
    }

    trained_acc_mean, trained_acc_std = summarize_baseline(initial_test_accuracies["trained"][0])
    benchmark_acc_mean, benchmark_acc_std = summarize_baseline(initial_test_accuracies["shadow_out"][0])

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metric_vs_unlearning(
        ax=ax,
        x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_test_accuracies,
        label_map=accuracy_styles,
        ylabel="Test Accuracy",
        title="Test Accuracy vs Unlearning Percentage",
        baselines=[
            {"mean": trained_acc_mean, "std": trained_acc_std, "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_acc_mean, "std": benchmark_acc_std, "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    trained_tpr_mean, trained_tpr_std = summarize_baseline(initial_tpr_at_fpr["trained"][0])
    benchmark_tpr_mean, benchmark_tpr_std = summarize_baseline(initial_tpr_at_fpr["shadow_out"][0])

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metric_vs_unlearning(
        ax=ax,
        x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_tpr_at_fpr,
        label_map=accuracy_styles,
        ylabel=f"TPR at FPR={target_fpr*100:.2f}%",
        title=f"TPR at FPR={target_fpr*100:.2f}% vs Unlearning Percentage",
        baselines=[
            {"mean": trained_tpr_mean, "std": trained_tpr_std, "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_tpr_mean, "std": benchmark_tpr_std, "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    print(
        f"Trained baseline TPR at FPR={target_fpr*100:.2f}%: "
        f"{trained_tpr_mean:.4f} ± {trained_tpr_std:.4f}"
    )




    
    
