import numpy as np
from collections import defaultdict
from torch.utils.data import Subset

def split_dataset_by_class_distribution(dataset, class_distributions):

    for dist in class_distributions:
        assert np.isclose(np.sum(dist), 1.0), "Class distribution must sum to 1."

    targets = np.array(dataset.targets)
    num_classes = len(class_distributions[0])
    num_clients = len(class_distributions)

    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    client_indices = [[] for _ in range(num_clients)]

    # Calculate the number of samples for each client
    total_requested_per_class = sum(class_distributions)
    total_requested = np.sum(total_requested_per_class)
    requested_class_dist = total_requested_per_class / total_requested
    available_class_dist = np.array([len(class_indices[cls]) for cls in range(num_classes)]) / len(targets)
    normalization_factor = np.max(requested_class_dist / available_class_dist)

    samples_distributions = np.round(class_distributions * len(targets) / (normalization_factor * num_clients)).astype(int)

    for cls in range(num_classes):
        indices = class_indices[cls]
        samples_per_client = np.clip(samples_distributions[:, cls], 0, len(indices))

        start = 0
        for client_id, count in enumerate(samples_per_client):
            client_indices[client_id].extend(indices[start:start + count])
            start += count

    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets

def concatenate_subsets(subsets):
    # THE SUBSETS MUST BE NON OVERLAPPING
    indices = []
    for subset in subsets:
        indices.extend(subset.indices)
    full_dataset = subsets[0].dataset
    return Subset(full_dataset, indices)