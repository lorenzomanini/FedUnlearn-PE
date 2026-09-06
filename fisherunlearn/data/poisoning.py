import numpy as np
import torch
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from torch.utils.data import TensorDataset


def create_poisoned_data(clients_subsets, init_params_dict):
    poisoned_clients_subsets, unlearning_eval_dataset = poisoning_data(
        clients_subsets, init_params_dict
    )
    if len(unlearning_eval_dataset) == 0:
        attack_eval_dataset = TensorDataset(torch.empty(0), torch.empty(0))
    else:
        poisoned_images = unlearning_eval_dataset.tensors[0]
        poison_target_label = init_params_dict.get("target_label", 9)
        target_labels = torch.full(
            (len(poisoned_images),), poison_target_label, dtype=torch.long
        )
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

    data_to_poison = np.array(
        [
            np.transpose(
                underlying_dataset[target_subset.indices[local_idx]][0].numpy(),
                (1, 2, 0),
            )
            for local_idx in local_indices_to_poison
        ]
    )
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
        all_images[global_dataset_idx] = new_image_tensor
        all_labels[global_dataset_idx] = new_label
        poisoned_images_for_eval.append(new_image_tensor)

    for client_subset in clients_subsets:
        client_subset.dataset = TensorDataset(
            torch.stack(all_images), torch.tensor(all_labels)
        )

    true_labels_tensor = torch.tensor(
        true_labels_of_poisoned_samples, dtype=torch.long
    )
    unlearning_eval_dataset = TensorDataset(
        torch.stack(poisoned_images_for_eval), true_labels_tensor
    )
    return clients_subsets, unlearning_eval_dataset
