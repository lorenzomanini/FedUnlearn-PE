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
        for i in range(len(accumulated_results[0][key])):
            merged_results[key].append(
                [accumulated_results[j][key][i] for j in range(len(accumulated_results))]
            )
    return merged_results

def compute_accuracy(outputs, labels):
    predictions = np.argmax(outputs, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def compute_accuracies(merged_results, labels, subset=None):
    labels = np.array(labels)
    idx = slice(None) if subset is None else subset
    accuracies = {}
    for model_key in merged_results.keys():
        accuracies[model_key] = []
        for test_outputs in merged_results[model_key]:
            test_accuracies = []
            for outputs in test_outputs:
                outputs = np.array(outputs)
                acc = compute_accuracy(outputs[idx], labels[idx])
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
            var = np.std(losses)
            std_member = var
            std_nonmember = var
        else:
            std_member = np.std(losses_members[model_key], axis=0)
            std_nonmember = np.std(losses_nonmembers[model_key], axis=0)
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
                outputs_tensor = torch.tensor(outputs)
                losses = loss_fn(outputs_tensor[idx], labels_tensor[idx]).numpy()
                lira = log_gaussian_pdf(losses, losses_in['avg'], losses_in['std']) - log_gaussian_pdf(losses, losses_out['avg'], losses_out['std'])
                test_lira_scores.append(lira)

            lira_scores[model_key].append(test_lira_scores)
    return lira_scores

def compute_roc_curves(lira_member_scores, lira_nonmember_scores):
    roc_curves = {}
    for model_key in lira_member_scores.keys():
        roc_curves[model_key] = []
        for test_idx in range(len(lira_member_scores[model_key])):
            test_roc_curves = []
            for iter_idx in range(len(lira_member_scores[model_key][test_idx])):
                member_scores = lira_member_scores[model_key][test_idx][iter_idx]
                nonmember_scores = lira_nonmember_scores[model_key][test_idx][iter_idx]
                member_scores = member_scores[~np.isnan(member_scores)]
                nonmember_scores = nonmember_scores[~np.isnan(nonmember_scores)]
                all_scores = np.concatenate([member_scores, nonmember_scores], axis=0)
                labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
                fpr, tpr, _ = roc_curve(labels, all_scores)
                roc_auc = auc(fpr, tpr)
                test_roc_curves.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})
            roc_curves[model_key].append(test_roc_curves)
    return roc_curves

def compute_tpr_at_fpr(roc_curve_data, target_fpr=0.01):
    tpr_at_fpr = {}
    for model_key in roc_curve_data.keys():
        tpr_at_fpr[model_key] = []
        for test_roc_curves in roc_curve_data[model_key]:
            test_tpr_at_fpr = []
            for curve in test_roc_curves:
                fpr = curve['fpr']
                tpr = curve['tpr']
                idx = np.where(fpr <= target_fpr)[0]
                if len(idx) > 0:
                    tpr_value = tpr[idx[-1]]
                else:
                    tpr_value = 0.0
                test_tpr_at_fpr.append(tpr_value)
            tpr_at_fpr[model_key].append(test_tpr_at_fpr)
    return tpr_at_fpr

if __name__ == "__main__":
    stat_test_name = "MNIST_pref"
    stat_tests_path = "stat_tests/NEW_TESTER" 
    overload_num_tests = None  # Set to an integer to override the number of tests

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
            acc_tests_eval_test.append(dict(np.load(f)))
        with open(os.path.join(test_path, "eval_train_results.npz"), "rb") as f:
            acc_tests_eval_train.append(dict(np.load(f)))
        with open(os.path.join(test_path, "extra_results.pkl"), "rb") as f:
            acc_tests_extra.append(pickle.load(f))
            
    initial_eval_test = merge_initial_results(acc_initial_eval_test)
    initial_eval_train = merge_initial_results(acc_initial_eval_train)
    tests_eval_test = merge_results(acc_tests_eval_test)
    tests_eval_train = merge_results(acc_tests_eval_train)
    tests_extra = merge_results(acc_tests_extra)

    # [model][test][repetition][entry]

    initial_test_accuracies = compute_accuracies(initial_eval_test, labels["test"])
    initial_train_accuracies = compute_accuracies(initial_eval_train, labels["train"])
    tests_test_accuracies = compute_accuracies(tests_eval_test, labels["test"])
    tests_train_accuracies = compute_accuracies(tests_eval_train, labels["train"])
    tests_target_accuracies = compute_accuracies(tests_eval_train, labels["train"], subset=target_indices)

    shadow_target_losses = compute_shadow_losses(initial_eval_train, labels["train"], subset=target_indices)
    shadow_test_losses = compute_shadow_losses(initial_eval_test, labels["test"])
    shadow_target_dists, shadow_test_dists = compute_shadow_losses_dists(shadow_target_losses, shadow_test_losses, global_var=True)
    lira_target_scores = compute_lira_scores(
        shadow_target_dists['shadow_in'], shadow_target_dists['shadow_out'], tests_eval_train, labels["train"], subset=target_indices)
    lira_test_scores = compute_lira_scores(
        shadow_test_dists['shadow_in'], shadow_test_dists['shadow_out'], tests_eval_test, labels["test"])

    roc_curves = compute_roc_curves(lira_target_scores, lira_test_scores)
    plt.plot(roc_curves['random_reset'][0][0]['fpr'], roc_curves['random_reset'][0][0]['tpr'], label=f"Trained AUC={roc_curves['random_reset'][0][0]['auc']:.2f}")
    plt.plot(roc_curves['reset'][1][0]['fpr'], roc_curves['reset'][1][0]['tpr'], label=f"Reset AUC={roc_curves['reset'][1][0]['auc']:.2f}")
    plt.plot(roc_curves['random_reset'][1][0]['fpr'], roc_curves['random_reset'][1][0]['tpr'], label=f"Random Reset AUC={roc_curves['random_reset'][1][0]['auc']:.2f}")
    plt.legend()
    plt.show()

    plt.plot(roc_curves['random_reset'][0][0]['fpr'], roc_curves['random_reset'][0][0]['tpr'], label=f"Trained AUC={roc_curves['random_reset'][0][0]['auc']:.2f}")
    plt.plot(roc_curves['retrained'][1][0]['fpr'], roc_curves['retrained'][1][0]['tpr'], label=f"Retrained AUC={roc_curves['retrained'][1][0]['auc']:.2f}")
    plt.plot(roc_curves['random_retrained'][1][0]['fpr'], roc_curves['random_retrained'][1][0]['tpr'], label=f"Random Retrained AUC={roc_curves['random_retrained'][1][0]['auc']:.2f}")
    plt.legend()
    plt.show()

    tpr_at_1pct_fpr = compute_tpr_at_fpr(roc_curves, target_fpr=0.01)
    plt.boxplot(tpr_at_1pct_fpr['retrained'])
    plt.title("TPR at 1% FPR for Reset Attack")
    plt.show()
    

    
    