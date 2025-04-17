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


def split_dataset_by_distribution( # Renamed for clarity
    dataset,
    # Option 1: Manual Distribution
    class_distributions: Optional[np.ndarray] = None,
    # Option 2: Dirichlet Distribution Parameters
    num_clients: Optional[int] = None,
    alpha: Optional[float] = None,
    # Other Parameters
    num_classes: Optional[int] = None,
    seed: Optional[int] = None
) -> List[Subset]:
    """
    Splits a dataset among clients based either on manually specified class
    distributions or distributions sampled from a Dirichlet distribution,
    using a simplified allocation strategy.

    Provide EITHER `class_distributions` OR (`num_clients` AND `alpha`).

    Args:
        dataset: The dataset to split (must have a .targets attribute).
        class_distributions (np.ndarray, optional): A pre-defined matrix of shape
            (num_clients, num_classes) representing desired class proportions per client.
            If provided, `num_clients` and `alpha` must be None.
        num_clients (int, optional): The number of client subsets to create.
            Required if using Dirichlet distribution (alpha is provided).
            Ignored if class_distributions is provided.
        alpha (float, optional): The concentration parameter for the Dirichlet distribution.
            Required if `num_clients` is provided and `class_distributions` is None.
        num_classes (int, optional): The number of classes in the dataset.
            If None, it's inferred from dataset.targets.
        seed (int, optional): Random seed for reproducibility (especially for Dirichlet).

    Returns:
        A list of Subset objects, one for each client.

    Raises:
        ValueError: If inputs are invalid or incompatible.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Input Validation and Setup ---
    if not hasattr(dataset, 'targets'):
        raise ValueError("Dataset must have a .targets attribute.")
    try:
        targets = np.array(dataset.targets)
    except Exception as e:
        raise ValueError(f"Could not convert dataset.targets to numpy array: {e}")

    if num_classes is None:
        try:
            num_classes = int(np.max(targets)) + 1
        except ValueError:
             raise ValueError("Could not determine num_classes from empty or non-numeric targets.")
    elif not isinstance(num_classes, int) or num_classes <= 0:
         raise ValueError("num_classes must be a positive integer.")

    _distributions: np.ndarray
    _num_clients: int

    if class_distributions is not None:
        # --- Mode 1: Manual Distribution ---
        if num_clients is not None or alpha is not None:
            raise ValueError("Provide EITHER class_distributions OR (num_clients AND alpha).")
        if not isinstance(class_distributions, np.ndarray) or class_distributions.ndim != 2:
             raise ValueError("class_distributions must be a 2D numpy array.")
        _num_clients = class_distributions.shape[0]
        if class_distributions.shape[1] != num_classes:
             raise ValueError(f"class_distributions shape ({class_distributions.shape}) incompatible with num_classes ({num_classes}).")
        _distributions = class_distributions
        print(f"Using provided manual class distributions for {_num_clients} clients.")

    elif num_clients is not None and alpha is not None:
        # --- Mode 2: Dirichlet Distribution ---
        if num_clients <= 0:
            raise ValueError("num_clients must be positive for Dirichlet distribution.")
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive.")
        _num_clients = num_clients
        # Sample distributions from Dirichlet: shape (num_clients, num_classes)
        _distributions = np.random.dirichlet([alpha] * num_classes, _num_clients)
        print(f"Generated distributions for {_num_clients} clients using Dirichlet(alpha={alpha}).")
    else:
        raise ValueError("Provide EITHER class_distributions OR (num_clients AND alpha).")

    # --- Prepare Indices ---
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
         label_int = int(label)
         if 0 <= label_int < num_classes:
             class_indices[label_int].append(idx)

    # Shuffle indices within each class
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # --- Allocate Indices  ---
    client_indices: List[List[int]] = [[] for _ in range(_num_clients)]
    indices_allocated_count = 0
    total_samples = len(targets)

    # Allocate indices class by class
    for cls in range(num_classes):
        if cls not in class_indices or not class_indices[cls]:
            continue # Skip classes with no samples

        indices_for_class = class_indices[cls]
        num_samples_in_class = len(indices_for_class)

        
        proportions = _distributions[:, cls] # Shape: (num_clients,)

        prop_sum = proportions.sum()
        if prop_sum > 0:
            proportions = proportions / prop_sum
        else:
            continue

        # Calculate counts based on proportions of this class's samples
        counts = np.floor(proportions * num_samples_in_class).astype(int)

        # Correct potential over-allocation due to sum(proportions) slightly > 1 after normalization
        current_sum = counts.sum()
        if current_sum > num_samples_in_class:
            # Simple correction: remove excess from largest counts
             for _ in range(current_sum - num_samples_in_class):
                 counts[np.argmax(counts)] -= 1

        # Assign indices using floor counts, discard leftovers for this class
        start = 0
        for client_id in range(_num_clients):
            count = counts[client_id]
            if count > 0:
                 # Ensure we don't go past the end of the list
                 end_pos = min(start + count, num_samples_in_class)
                 assigned_indices = indices_for_class[start : end_pos]
                 client_indices[client_id].extend(assigned_indices)
                 indices_allocated_count += len(assigned_indices)
                 start = end_pos
        # Leftover indices indices_for_class[start:] are discarded for this class

    print(f"Allocated {indices_allocated_count}/{total_samples} samples based on floor proportions (class leftovers discarded).")
    print("\nFinal Client Sample Counts:")
    for client_id, indices in enumerate(client_indices):
         count = len(indices)
         print(f"  Client {client_id}: {count} samples")
         if count == 0:
             print(f"    Warning: Client {client_id} has ZERO samples. Check distributions/alpha.")


    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets


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
