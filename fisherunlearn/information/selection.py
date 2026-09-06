import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_information_parameters_tradeoff(
    information, method, whitelist=None, blacklist=None
):
    percentages = np.arange(0, 100, 0.1)
    information_values = np.zeros(len(percentages))
    params_values = np.zeros(len(percentages))
    total_information = 0
    total_params = 0

    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        sorted_layer_info = np.sort(layer_info.flatten())[::-1]
        cumulative_sum = np.cumsum(sorted_layer_info)
        total_information += cumulative_sum[-1]
        total_params += len(sorted_layer_info)
        for i, percentage in enumerate(percentages):
            if method == "parameters":
                threshold_idx = int(len(sorted_layer_info) / 100 * percentage)
                information_values[i] += cumulative_sum[threshold_idx]
                params_values[i] += threshold_idx
            elif method == "information":
                threshold_idx = np.argmin(
                    np.abs(cumulative_sum - cumulative_sum[-1] * percentage / 100)
                )
                information_values[i] += cumulative_sum[threshold_idx]
                params_values[i] += threshold_idx
            else:
                raise ValueError("Invalid method. Use 'information' or 'parameters'.")

    for i in range(len(information_values)):
        information_values[i] = information_values[i] / total_information * 100
        params_values[i] = 100 - params_values[i] / total_params * 100
    plt.plot(percentages, information_values, label="Information erased")
    plt.plot(percentages, params_values, label="Remaining parameters")
    plt.xlabel(f"Layer {method} percentage resetted")
    plt.ylabel("Total percentage")
    plt.title("Information vs Parameters tradeoff")
    plt.legend()
    plt.grid()
    plt.show()


def find_informative_params(
    information,
    method,
    percentage,
    whitelist=None,
    blacklist=None,
    graph=False,
    tuple_out=False,
):
    informative_params = {}
    thresholds = {}

    if method == "random":
        random_information = {}
        for name, layer_info in information.items():
            layer = torch.zeros_like(layer_info)
            flat_view = layer.view(-1)
            flat_view[
                torch.randperm(flat_view.numel())[
                    : int(flat_view.numel() * percentage / 100)
                ]
            ] = 1
            thresholds[name] = 0.5
            random_information[name] = layer
        information = random_information
    else:
        for name, layer_info in information.items():
            if whitelist is not None and name not in whitelist:
                continue
            if blacklist is not None and name in blacklist:
                continue
            if method == "information":
                sorted_layer_info = np.sort(layer_info.flatten())[::-1]
                cumulative_sum = np.cumsum(sorted_layer_info)
                threshold_idx = np.argmin(
                    np.abs(cumulative_sum - cumulative_sum[-1] * percentage / 100)
                )
                threshold_idx = min(threshold_idx, len(sorted_layer_info) - 1)
                thresholds[name] = sorted_layer_info[threshold_idx]
            elif method == "parameters":
                sorted_layer_info = np.sort(layer_info.flatten())[::-1]
                threshold_idx = int(len(sorted_layer_info) / 100 * percentage)
                threshold_idx = min(threshold_idx, len(sorted_layer_info) - 1)
                thresholds[name] = sorted_layer_info[threshold_idx]
            else:
                raise ValueError("Invalid method. Use 'information' or 'parameters'.")

        if graph:
            plt.figure(figsize=(10, 5))
            plt.title(name)
            plt.plot(sorted_layer_info)
            plt.axvline(threshold_idx, color="r", linestyle="--")
            plt.xlabel("Parameters")
            plt.ylabel("Information")
            plt.show()

    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        if tuple_out:
            informative_params[name] = tuple(
                torch.argwhere(layer_info > thresholds[name]).t()
            )
        else:
            informative_params[name] = torch.argwhere(layer_info > thresholds[name])
    return informative_params
