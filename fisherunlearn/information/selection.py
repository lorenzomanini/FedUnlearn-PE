"""Select reset coordinates from per-parameter scores."""

import math

import torch


def _sorted_scores(layer_info):
    scores = torch.as_tensor(layer_info).detach().reshape(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Parameter scores must be finite.")
    if (scores < 0).any():
        raise ValueError("Parameter scores must be nonnegative.")
    return torch.sort(scores, descending=True, stable=True)


def _cumulative_mass(sorted_scores):
    # Normalize before accumulating to avoid overflow for large ratio scores.
    dtype = torch.float32 if sorted_scores.device.type == "mps" else torch.float64
    scores = sorted_scores.to(dtype)
    if scores.numel() and scores[0] > 0:
        scores = scores / scores[0]
    return scores.cumsum(0)


def _prefix_count(cumulative_mass, percentage):
    if percentage == 0 or not cumulative_mass.numel() or cumulative_mass[-1] == 0:
        return 0
    target = cumulative_mass[-1] * (percentage / 100)
    return int(torch.searchsorted(cumulative_mass, target, right=False).item()) + 1


def plot_information_parameters_tradeoff(
    information, method, whitelist=None, blacklist=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    if method not in {"information", "parameters"}:
        raise ValueError("Invalid method. Use 'information' or 'parameters'.")
    percentages = np.linspace(0, 100, 1001)
    information_values = np.zeros(len(percentages))
    params_values = np.zeros(len(percentages))
    total_information = 0.0
    total_params = 0
    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        sorted_scores, _ = _sorted_scores(layer_info)
        scores = sorted_scores.to(device="cpu", dtype=torch.float64).numpy()
        if not scores.size:
            continue
        cumulative_sum = np.concatenate(([0.0], np.cumsum(scores)))
        if method == "parameters":
            counts = (scores.size * percentages / 100).astype(int)
        elif cumulative_sum[-1] == 0:
            counts = np.zeros(len(percentages), dtype=int)
        else:
            counts = np.searchsorted(cumulative_sum, cumulative_sum[-1] * percentages / 100)
        information_values += cumulative_sum[counts]
        params_values += counts
        total_information += cumulative_sum[-1]
        total_params += scores.size
    if total_information:
        information_values *= 100 / total_information
    if total_params:
        params_values = 100 - params_values / total_params * 100
    else:
        params_values.fill(100)
    plt.plot(percentages, information_values, label="Selected score mass")
    plt.plot(percentages, params_values, label="Remaining parameters")
    plt.xlabel(f"Layer {method} percentage reset")
    plt.ylabel("Total percentage")
    plt.title("Score mass vs Parameters tradeoff")
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
    """Return coordinates selected independently within each parameter group.

    ``information`` selects the smallest descending prefix reaching the requested
    score mass. A group with zero total score selects nothing. ``parameters`` and
    ``random`` select floor(group_size * percentage / 100) coordinates. Tied
    scores are resolved in flattened coordinate order, so the count stays exact.
    """
    if method not in {"information", "parameters", "random"}:
        raise ValueError("Invalid method. Use 'information', 'parameters', or 'random'.")
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100.")

    informative_params = {}
    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        layer_info = torch.as_tensor(layer_info)
        size = layer_info.numel()
        if method == "random":
            count = int(size * percentage / 100)
            selected = torch.randperm(size, device=layer_info.device)[:count]
        else:
            sorted_scores, order = _sorted_scores(layer_info)
            count = (
                _prefix_count(_cumulative_mass(sorted_scores), percentage)
                if method == "information"
                else int(size * percentage / 100)
            )
            selected = order[:count]
            if graph:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 5))
                plt.title(name)
                plt.plot(sorted_scores.cpu().numpy())
                plt.axvline(count, color="r", linestyle="--")
                plt.xlabel("Parameters")
                plt.ylabel("Information")
                plt.show()

        # Preserve the coordinate order previously returned by argwhere.
        mask = torch.zeros(size, dtype=torch.bool, device=layer_info.device)
        mask[selected] = True
        indices = torch.argwhere(mask.reshape(layer_info.shape))
        informative_params[name] = tuple(indices.t()) if tuple_out else indices
    return informative_params
