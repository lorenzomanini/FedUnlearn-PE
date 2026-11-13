from numbers import Integral
from pathlib import Path
import pickle

import numpy as np
import torch

from fisherunlearn.data.partition import split_dataset_by_class_distribution


def _use_cifar100_superclasses(dataset, train):
    """Keep torchvision's image order and expose the official coarse labels."""
    folder = Path(dataset.root) / dataset.base_folder
    with (folder / ("train" if train else "test")).open("rb") as stream:
        labels = pickle.load(stream, encoding="latin1")
    with (folder / "meta").open("rb") as stream:
        metadata = pickle.load(stream, encoding="latin1")
    fine = np.asarray(labels["fine_labels"])
    coarse = np.asarray(labels["coarse_labels"])
    fine_classes = metadata["fine_label_names"]
    coarse_classes = metadata["coarse_label_names"]
    if (
        fine.shape != (len(dataset),)
        or coarse.shape != fine.shape
        or not np.array_equal(fine, np.asarray(dataset.targets))
        or list(dataset.classes) != fine_classes
        or len(fine_classes) != 100
        or len(coarse_classes) != 20
        or not np.array_equal(np.unique(fine), np.arange(100))
        or not np.array_equal(np.unique(coarse), np.arange(20))
    ):
        raise ValueError("CIFAR-100 coarse labels must align with all 100 fine classes")
    fine_to_coarse = []
    for label in range(100):
        parents = np.unique(coarse[fine == label])
        if len(parents) != 1:
            raise ValueError("Each CIFAR-100 fine class must have exactly one superclass")
        fine_to_coarse.append(int(parents[0]))
    if not np.all(np.bincount(fine_to_coarse, minlength=20) == 5):
        raise ValueError("Each CIFAR-100 superclass must contain five fine classes")
    dataset.fine_targets = fine.tolist()
    dataset.fine_classes = list(fine_classes)
    dataset.fine_class_to_idx = {name: index for index, name in enumerate(fine_classes)}
    dataset.fine_to_coarse = fine_to_coarse
    dataset.targets = coarse.tolist()
    dataset.classes = list(coarse_classes)
    dataset.coarse_classes = list(coarse_classes)
    dataset.class_to_idx = {name: index for index, name in enumerate(coarse_classes)}


def get_datasets(init_params_dict):
    dataset_name = init_params_dict["dataset_name"]
    model_name = init_params_dict["model_name"]
    if dataset_name == "breast_cancer":
        from BreastCancerDataset import BreastCancerDataset

        train_dataset = BreastCancerDataset("./data", split="train")
        test_dataset = BreastCancerDataset("./data", split="test")
        init_params_dict["num_classes"] = len(train_dataset.classes)
        init_params_dict["input_dim"] = len(train_dataset.input_columns)
    elif dataset_name == "mnist":
        from torchvision import transforms
        from torchvision.datasets import MNIST

        if model_name == "simple_cnn":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        else:
            raise ValueError("Unsupported model name for MNIST dataset")
        train_dataset = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        if model_name == "simple_cnn":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5), (0.5)),
                ]
            )
        elif model_name == "resnet18":
            transform = transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported model name for CIFAR10 dataset")
        train_dataset = CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name in ("cifar100", "cifar100_superclass"):
        from torchvision import transforms
        from torchvision.datasets import CIFAR100

        if model_name == "simple_cnn":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5), (0.5)),
                ]
            )
        elif model_name == "resnet18":
            transform = transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported model name for CIFAR100 dataset")
        train_dataset = CIFAR100(
            root="./data", train=True, download=init_params_dict.get("download", True),
            transform=transform
        )
        test_dataset = CIFAR100(
            root="./data", train=False, download=init_params_dict.get("download", True),
            transform=transform
        )
        if dataset_name == "cifar100_superclass":
            _use_cifar100_superclasses(train_dataset, train=True)
            _use_cifar100_superclasses(test_dataset, train=False)
            if (
                train_dataset.fine_to_coarse != test_dataset.fine_to_coarse
                or train_dataset.classes != test_dataset.classes
                or train_dataset.fine_classes != test_dataset.fine_classes
            ):
                raise ValueError("CIFAR-100 train and test superclass metadata must agree")
            init_params_dict["num_classes"] = 20
    elif dataset_name == "FashionMNIST":
        from torchvision import transforms
        from torchvision.datasets import FashionMNIST

        if model_name == "simple_cnn":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5), (0.5)),
                ]
            )
        elif model_name == "resnet18":
            transform = transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported model name for FashionMNIST dataset")
        train_dataset = FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "cartelli":
        from torchvision import transforms
        from torchvision.datasets import GTSRB

        transform = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3337, 0.3064, 0.3171],
                    std=[0.2672, 0.2564, 0.2629],
                ),
            ]
        )
        train_dataset = GTSRB(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = GTSRB(
            root="./data", split="test", download=True, transform=transform
        )
    else:
        raise ValueError("Unsupported dataset name")
    return train_dataset, test_dataset


def preferential_class_distribution(num_clients, num_classes):
    if num_clients > num_classes:
        raise ValueError(
            "Number of clients must be less than or equal to number of classes for preferential_class distribution"
        )
    num_common_classes = num_classes - num_clients
    p_common = 1 / (num_common_classes + num_clients)
    p_preferred = p_common * num_clients
    class_distribution = np.zeros((num_clients, num_classes))
    for i in range(num_clients):
        for j in range(num_common_classes):
            class_distribution[i, j] = p_common
        class_distribution[i, num_common_classes + i] = p_preferred
    return class_distribution


def get_clients_subsets(dataset, init_params_dict):
    num_clients = init_params_dict["num_clients"]
    num_classes = init_params_dict["num_classes"]
    distribution_type = init_params_dict["distribution_type"]
    if distribution_type == "fine_class":
        if not hasattr(dataset, "fine_targets") or not hasattr(dataset, "fine_classes"):
            raise ValueError("fine_class distribution requires CIFAR-100 superclass labels")
        if isinstance(num_clients, bool) or not isinstance(num_clients, Integral) or num_clients < 2:
            raise ValueError("fine_class distribution requires at least two clients")
        target = init_params_dict["target_fine_class"]
        if (
            isinstance(target, bool) or not isinstance(target, Integral)
            or not 0 <= target < len(dataset.fine_classes)
        ):
            raise ValueError("target_fine_class must be an integer fine-class ID")
        fine = np.asarray(dataset.fine_targets)
        if fine.shape != (len(dataset),):
            raise ValueError("fine_targets must align with the dataset")
        target_indices = np.flatnonzero(fine == target)
        retained_indices = np.flatnonzero(fine != target)
        if not len(target_indices) or len(retained_indices) < num_clients - 1:
            raise ValueError("fine_class distribution requires a nonempty target and retained clients")
        np.random.shuffle(retained_indices)
        indices = [target_indices, *np.array_split(retained_indices, num_clients - 1)]
        return [torch.utils.data.Subset(dataset, part.tolist()) for part in indices]
    elif distribution_type == "preferential_class":
        return split_dataset_by_class_distribution(
            dataset, preferential_class_distribution(num_clients, num_classes)
        )
    elif distribution_type == "preferential_mix":
        # Strength 0 is the stratified uniform split and 1 is preferential_class.
        # With balanced labels every client keeps the same size at every strength.
        strength = init_params_dict["preferential_strength"]
        if isinstance(strength, bool) or not 0 <= strength <= 1:
            raise ValueError("preferential_strength must be between 0 and 1")
        uniform = np.ones((num_clients, num_classes)) / num_classes
        class_distribution = (
            strength * preferential_class_distribution(num_clients, num_classes)
            + (1 - strength) * uniform
        )
        return split_dataset_by_class_distribution(dataset, class_distribution)
    elif distribution_type == "categorical":
        if num_clients != num_classes:
            raise ValueError(
                "Number of clients must be equal to number of classes for purely categorical distribution"
            )
        return split_dataset_by_class_distribution(dataset, np.identity(num_classes))
    elif distribution_type == "uniform":
        class_distribution = np.ones((num_clients, num_classes)) / num_classes
        return split_dataset_by_class_distribution(dataset, class_distribution)
    elif distribution_type == "dirichlet":
        alpha = init_params_dict.get("dirichlet_alpha", 1)
        class_distribution = np.random.dirichlet(
            [alpha] * num_classes, num_clients
        )
        return split_dataset_by_class_distribution(dataset, class_distribution)
    elif distribution_type == "random":
        lengths = [1 / num_clients] * num_clients
        return torch.utils.data.random_split(dataset, lengths)
    elif distribution_type == "BC_targeted":
        from BreastCancerDataset import BreastCancerDataset, split_by_age

        if not isinstance(dataset, BreastCancerDataset):
            raise ValueError(
                "BC_targeted distribution can only be used with BreastCancerDataset"
            )
        return split_by_age(dataset)
    else:
        raise ValueError("Unsupported distribution type")
