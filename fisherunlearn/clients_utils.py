import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
from typing import Optional, List

def generate_dirichlet_distributions(
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates class distributions for clients using a symmetric Dirichlet distribution.

    Args:
        num_clients: The number of client distributions to generate.
        num_classes: The number of classes in the dataset.
        alpha: The concentration parameter for the Dirichlet distribution.
               - Smaller alpha (<1.0) -> more skewed distributions per client.
               - Larger alpha (>1.0) -> more uniform distributions per client.
        seed: Optional random seed for reproducibility.

    Returns:
        A numpy array of shape (num_clients, num_classes) where each row
        represents the probability distribution over classes for a client.
        Each row sums to 1.0.
    """
    if seed is not None:
        np.random.seed(seed)

    if num_clients <= 0 or num_classes <= 0 or alpha <= 0:
        raise ValueError("num_clients, num_classes, and alpha must be positive.")

    # Sample distributions: shape (num_clients, num_classes)
    # Each row is a sample from Dirichlet(alpha, alpha, ..., alpha)
    class_distributions = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    return class_distributions

def split_dataset_by_class_distribution(dataset, class_distributions):
    for dist in class_distributions:
        assert np.isclose(np.sum(dist), 1.0), "Class distribution must sum to 1."

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
    return subsets, client_indices
