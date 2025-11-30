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
        class_indices[int(label)].append(idx)

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

import torch
import numpy as np
from torch.utils.data import TensorDataset
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd


def create_poisoned_data(clients_subsets, init_params_dict):

    poisoned_clients_subsets, unlearning_eval_dataset = poisoning_data(
        clients_subsets, init_params_dict
    )

    if len(unlearning_eval_dataset) == 0:
        # If no data was poisoned, the attack evaluation set is also empty
        attack_eval_dataset = TensorDataset(torch.empty(0), torch.empty(0))
    else:
        # Create the dataset for evaluating attack success
        poisoned_images = unlearning_eval_dataset.tensors[0]
        poison_target_label = init_params_dict.get("target_label", 9)
        target_labels = torch.full((len(poisoned_images),), poison_target_label, dtype=torch.long)
        attack_eval_dataset = TensorDataset(poisoned_images, target_labels)
        
    return poisoned_clients_subsets, attack_eval_dataset, unlearning_eval_dataset


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

    poisoned_data_np, poisoned_labels_np = backdoor.poison(
        data_to_poison, y=example_target, broadcast=True
    )

    poisoned_images_for_eval = []

    all_images = [underlying_dataset[i][0] for i in range(len(underlying_dataset))]
    all_labels = [underlying_dataset[i][1] for i in range(len(underlying_dataset))]

    for i, local_idx in enumerate(local_indices_to_poison):
        global_dataset_idx = target_subset.indices[local_idx]
        
        new_image_tensor = torch.tensor(
            np.transpose(poisoned_data_np[i], (2, 0, 1)), dtype=torch.float
        )

        new_label = int(np.argmax(poisoned_labels_np[i]))

        # Visualize the new image tensor
        # import matplotlib.pyplot as plt
        # plt.imshow(new_image_tensor.permute(1, 2, 0))
        # plt.title(f"Poisoned Label: {new_label}")
        # plt.show()

        all_images[global_dataset_idx] = new_image_tensor
        all_labels[global_dataset_idx] = new_label

        poisoned_images_for_eval.append(new_image_tensor)

    for client_subset in clients_subsets:
        client_subset.dataset = TensorDataset(torch.stack(all_images), torch.tensor(all_labels))

    true_labels_tensor = torch.tensor(true_labels_of_poisoned_samples, dtype=torch.long)
    unlearning_eval_dataset = TensorDataset(torch.stack(poisoned_images_for_eval), true_labels_tensor)

    return clients_subsets, unlearning_eval_dataset
