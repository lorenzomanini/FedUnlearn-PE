from torch.utils.data import Subset
from collections import defaultdict
import numpy as np

def split_dataset_by_class_distribution(dataset, class_distributions):
    targets = np.array(dataset.targets)
    num_classes = np.max(targets) + 1
    num_clients = len(class_distributions)

    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    client_indices = [[] for _ in range(num_clients)]

    # Crop samples so that each client has roughly the same number of samples
    total_requested = sum(class_distributions)
    most_requested = np.max(total_requested)
    class_proportions = total_requested / most_requested
    samples_per_class = np.array([int(class_proportions[cls] * len(class_indices[cls])) for cls in range(num_classes)])
    class_indices = [class_indices[cls][:samples_per_class[cls]] for cls in range(num_classes)]


    for cls in range(num_classes):
        indices = class_indices[cls]
        total = len(indices)

        proportions = np.array([dist[cls] for dist in class_distributions])
        proportions /= proportions.sum()  # just in case

        # Only floor allocation, no rounding up
        counts = np.floor(proportions * total).astype(int)

        # Assign indices, ignore leftovers
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(indices[start:start + count])
            start += count
        # Remaining indices are discarded

    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets
