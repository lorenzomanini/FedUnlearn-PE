from collections import defaultdict

import numpy as np
from torch.utils.data import Subset


def split_dataset_by_class_distribution(dataset, class_distributions):
    for dist in class_distributions:
        assert np.isclose(np.sum(dist), 1.0), "Class distribution must sum to 1."

    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "_samples"):
        targets = np.array([s[1] for s in dataset._samples])

    num_classes = len(class_distributions[0])
    num_clients = len(class_distributions)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    client_indices = [[] for _ in range(num_clients)]
    total_requested_per_class = sum(class_distributions)
    total_requested = np.sum(total_requested_per_class)
    requested_class_dist = total_requested_per_class / total_requested
    available_class_dist = np.array(
        [len(class_indices[cls]) for cls in range(num_classes)]
    ) / len(targets)
    limiting_class = np.argmax(requested_class_dist - available_class_dist)
    total_client_samples = np.floor(
        len(class_indices[limiting_class]) / total_requested_per_class[limiting_class]
    )
    samples_distributions = np.floor(
        class_distributions * total_client_samples
    ).astype(int)

    for cls in range(num_classes):
        indices = class_indices[cls]
        samples_distribution = samples_distributions[:, cls]
        start = 0
        for client_id, count in enumerate(samples_distribution):
            client_indices[client_id].extend(indices[start : start + count])
            start += count

    return [Subset(dataset, indices) for indices in client_indices]


def concatenate_subsets(subsets):
    indices = []
    for subset in subsets:
        indices.extend(subset.indices)
    return Subset(subsets[0].dataset, indices)


def random_split_subset(subset, lengths):
    total_length = sum(lengths)
    assert total_length <= len(subset), "Sum of lengths exceeds subset size."
    indices = np.array(subset.indices)
    np.random.shuffle(indices)
    split_subsets = []
    start = 0
    for length in lengths:
        split_indices = indices[start : start + length].tolist()
        split_subsets.append(Subset(subset.dataset, split_indices))
        start += length
    return split_subsets
