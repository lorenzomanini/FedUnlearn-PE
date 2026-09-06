import numpy as np
import torch

from fisherunlearn.data.partition import split_dataset_by_class_distribution


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
    elif dataset_name == "cifar100":
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
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
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


def get_clients_subsets(dataset, init_params_dict):
    num_clients = init_params_dict["num_clients"]
    num_classes = init_params_dict["num_classes"]
    distribution_type = init_params_dict["distribution_type"]
    if distribution_type == "preferential_class":
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
