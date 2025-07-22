import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import torch
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

def split_dataset_by_class_distribution(dataset, class_distributions):

    for dist in class_distributions:
        assert np.isclose(np.sum(dist), 1.0), "Class distribution must sum to 1."

    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, '_samples'):
        targets = np.array([s[1] for s in dataset._samples])

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
    limiting_class = np.argmax(requested_class_dist - available_class_dist)
    total_client_samples = np.floor(len(class_indices[limiting_class]) / total_requested_per_class[limiting_class])

    samples_distributions = np.floor(class_distributions * total_client_samples).astype(int)

    for cls in range(num_classes):
        indices = class_indices[cls]
        samples_distribution = samples_distributions[:, cls]

        start = 0
        for client_id, count in enumerate(samples_distribution):
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

def poisoning_data(clients_subsets, init_params_dict):

    target_client = init_params_dict["target_client"]
    num_classes = init_params_dict["num_classes"]
    target_label = init_params_dict.get("target_label", 9)

    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.zeros(num_classes)
    example_target[target_label] = 1

    target_subset = clients_subsets[target_client]
    underlying_dataset = target_subset.dataset

    local_indices_to_poison = []
    true_labels_of_poisoned_samples = []

    for local_idx, global_idx in enumerate(target_subset.indices):
        true_label = underlying_dataset[global_idx][1]
        if true_label != target_label:
            local_indices_to_poison.append(local_idx)
            true_labels_of_poisoned_samples.append(true_label)

    if not local_indices_to_poison:
        return clients_subsets, TensorDataset(torch.empty(0), torch.empty(0))

    data_to_poison = np.array([
        np.transpose(underlying_dataset[target_subset.indices[local_idx]][0].numpy(), (1, 2, 0))
        for local_idx in local_indices_to_poison
    ])

    poisoned_data, poisoned_labels = backdoor.poison(
        data_to_poison, y=example_target, broadcast=True
    )

    poisoned_images_for_eval = []

    for i, local_idx in enumerate(local_indices_to_poison):
        global_dataset_idx = target_subset.indices[local_idx]
        
        poisoned_image_np = np.transpose(poisoned_data[i], (2, 0, 1))
        new_image_tensor = torch.tensor(poisoned_image_np, dtype=torch.float32)
        new_label = int(np.argmax(poisoned_labels[i]))

        if hasattr(underlying_dataset, 'data') and hasattr(underlying_dataset, 'targets'):
            underlying_dataset.data[global_dataset_idx] = new_image_tensor
            underlying_dataset.targets[global_dataset_idx] = new_label
        elif hasattr(underlying_dataset, 'samples'):
            underlying_dataset.samples[global_dataset_idx] = (new_image_tensor, new_label)
        else:
            underlying_dataset[global_dataset_idx] = (new_image_tensor, new_label)

        poisoned_images_for_eval.append(new_image_tensor)

    true_labels_tensor = torch.tensor(true_labels_of_poisoned_samples, dtype=torch.long)
    evaluation_dataset = TensorDataset(torch.stack(poisoned_images_for_eval), true_labels_tensor)

    return clients_subsets, evaluation_dataset
