import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_curve, auc

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
            std = np.std(losses, ddof=1)
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
    coeff = - np.log(sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff + exponent
    
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
                member_scores = member_scores[~np.isnan(member_scores)]
                nonmember_scores = nonmember_scores[~np.isnan(nonmember_scores)]
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

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    stat_test_name = "MNIST_random"
    stat_tests_path = "stat_tests/EXPERIMENTS" 
    overload_num_tests = 1  # Set to an integer to override the number of tests

    stat_test_path = os.path.join(stat_tests_path, stat_test_name)
    if not os.path.exists(stat_test_path):
        raise FileNotFoundError(f"Stat test path {stat_test_path} does not exist.")

    with open(os.path.join(stat_test_path, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(stat_test_path, "clients_indices.pkl"), "rb") as f:
        client_indices = pickle.load(f)

    target_indices = client_indices[0]

    num_tests = 1
    if overload_num_tests is not None:
        num_tests = overload_num_tests

    acc_initial_eval_test = []
    acc_initial_eval_train = []
    acc_tests_eval_test = []
    acc_tests_eval_train = []
    acc_tests_extra = []

    for i in range(num_tests):
        test_path = os.path.join(stat_test_path, f"iter_{i}")
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

    target_fpr = 0.001

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




    
    
