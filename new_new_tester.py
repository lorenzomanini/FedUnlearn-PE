from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets, random_split_subset, create_poisoned_data, poisoning_data
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters, mia_attack
from fisherunlearn import UnlearnNet
from fisherunlearn import plot_information_parameters_tradeoff
from new_method import estimate_diag_commuting_backpack

import fisherunlearn

import os
import pickle
import random
import logging
import functools
import sys, traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

from torch.multiprocessing import Pool, Queue
torch.multiprocessing.set_start_method('spawn', force=True)

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


from typing import TypedDict, Literal

NUM_POWER_ITERS = int(os.environ.get("NUM_POWER_ITERS", "5"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 5000
TRAIN_BATCH_SIZE = 5000
INFO_BATCH_SIZE = 1000
MIA_BATCH_SIZE = 1000

fisherunlearn.set_device(DEVICE)
fisherunlearn.set_info_batch_size(INFO_BATCH_SIZE)
fisherunlearn.set_mia_batch_size(MIA_BATCH_SIZE)

def set_device(device):
    global DEVICE
    DEVICE = device
    fisherunlearn.set_device(device)

def set_batch_sizes(info_batch_size=INFO_BATCH_SIZE, mia_batch_size=MIA_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE, train_batch_size=TRAIN_BATCH_SIZE):
    global INFO_BATCH_SIZE, MIA_BATCH_SIZE, EVAL_BATCH_SIZE, TRAIN_BATCH_SIZE
    INFO_BATCH_SIZE = info_batch_size
    MIA_BATCH_SIZE = mia_batch_size
    EVAL_BATCH_SIZE = eval_batch_size
    TRAIN_BATCH_SIZE = train_batch_size
    fisherunlearn.set_info_batch_size(info_batch_size)
    fisherunlearn.set_mia_batch_size(mia_batch_size)


def compute_accuracy(model, dataset):
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    tqdm_bar = tqdm(total=len(dataloader), desc="Computing accuracy", unit="batch", leave=False)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tqdm_bar.update(1)
    
    tqdm_bar.close()
    model.cpu()

    return correct / total

def logit_margin(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    z: (B, C) logits
    y: (B,) int64 labels
    returns: (B,) margins z_y - max_{j!=y} z_j
    """
    # true class logit
    zy = z.gather(1, y.unsqueeze(1)).squeeze(1)  # (B,)

    # top-2 logits and their class indices
    top2_vals, top2_idx = z.topk(k=2, dim=1)     # (B,2), (B,2)

    # best "other" logit: if top1 is true class -> take top2 else top1
    top1_is_true = (top2_idx[:, 0] == y)
    z_other = torch.where(top1_is_true, top2_vals[:, 1], top2_vals[:, 0])  # (B,)

    return zy - z_other

def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    preds = []
    margins = []
    model.to(DEVICE)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            batch_preds = torch.argmax(output, dim=1)
            preds.append(batch_preds.cpu())
            batch_margins = logit_margin(output, labels)
            margins.append(batch_margins.cpu())
    model.cpu()
    return {
        "pred": torch.cat(preds).numpy(),
        "loss": torch.cat(margins).numpy()
    }


class InitParamsDict(TypedDict):
    test_name: str
    dataset_name: Literal['mnist', 'cifar10', 'FashionMNIST', 'cifar100']
    num_clients: int
    num_classes: int
    distribution_type: Literal['preferential_class', 'uniform', 'dirichlet', 'random', 'categorical']
    model_name: Literal['simple_cnn', 'resnet18']
    loss_name: Literal['cross_entropy', 'mse']
    trainer_name: Literal['sgd']
    train_epochs: int
    target_client: int
    num_tests: int
    info_use_converter: bool
    use_FIM: bool
    hessian_method: Literal['diag_hessian', 'diag_ggn', 'diag_ggn_mc']
    poison: bool
    local_epochs: int
    participation_rate: float
    learning_rate: float
    momentum: float

class TestParamsDict(TypedDict):
    subtest: int
    unlearning_method: Literal['information', 'parameters']
    unlearning_percentage: float
    retrain_epochs: int
    tests: list[Literal['test_accuracy', 'target_accuracy', 'clients_accuracies', 'class_accuracies', 'mia', 'categorical_accuracies', 'poison_accuracy']]
    mia_classifier_types: list[Literal['nn', 'logistic', 'svm']]
    whitelist: list[str]
    blacklist: list[str]


class Test():
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, 
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None):

        # Initialize parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.complete_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        self.clients_subsets = clients_subsets

        self.init_params_dict = init_params_dict
        self.target_client = init_params_dict['target_client']
        self.target_subset = self.clients_subsets[self.target_client]
        self.non_target_subsets = [subset for i, subset in enumerate(self.clients_subsets) if i != self.target_client]
        
        batch_size = 128
        self.client_loader = DataLoader(self.clients_subsets[self.target_client], batch_size, shuffle=True)
        self.total_loader = DataLoader(self.clients_subsets[0].dataset, batch_size, shuffle=True)

        self.poisoned_backdoor_dataset = poisoned_backdoor_dataset
        self.clean_backdoor_dataset = clean_backdoor_dataset

        self.model_class = model_class
        self.num_total_params = sum(p.numel() for p in model_class().parameters())
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        train_epochs = init_params_dict['train_epochs']

        logging.info("Preparing subsets for training...")
        eval_split_ratio = 0.1
        shadow_out_subsets = []
        eval_subsets = []
        for subset in self.non_target_subsets:
            n_eval = int(len(subset) * eval_split_ratio)
            n_train = len(subset) - n_eval
            train, eval = random_split_subset(subset, [n_train, n_eval])
            shadow_out_subsets.append(train)
            eval_subsets.append(eval)

        self.retrain_subsets = shadow_out_subsets
        self.eval_subsets = eval_subsets

        train_subsets = shadow_out_subsets + [self.target_subset]

        shadow_in_subsets = [Subset(self.complete_dataset, subset.indices) for subset in train_subsets]
        shadow_in_subsets.append(Subset(self.complete_dataset, list(range(len(self.train_dataset), len(self.complete_dataset)))))  # add test dataset as last subset

        logging.info("Training trained model...") 
        self.trained_model = self.trainer_function(
            self.model_class(), self.loss_class(), train_subsets, eval_subsets,
            train_epochs
        )

        logging.info("Training shadow_out model...") 
        shadow_out_model = self.trainer_function(
            self.model_class(), self.loss_class(), shadow_out_subsets, eval_subsets,
            train_epochs
        )

        logging.info("Training shadow_in model...")
        shadow_in_model = self.trainer_function(
            self.model_class(), self.loss_class(), shadow_in_subsets, eval_subsets,
            train_epochs
        )

        logging.info("Computing initial evaluation results...")

        self.init_eval_test_results = {
            "trained": evaluate_model(self.trained_model, self.test_dataset),
            "shadow_out": evaluate_model(shadow_out_model, self.test_dataset),
            "shadow_in": evaluate_model(shadow_in_model, self.test_dataset)
        }
        self.init_eval_train_results = {
            "trained": evaluate_model(self.trained_model, self.train_dataset),
            "shadow_out": evaluate_model(shadow_out_model, self.train_dataset),
            "shadow_in": evaluate_model(shadow_in_model, self.train_dataset)
        }

        logging.info("Computing client information...")
        self.client_information = estimate_diag_commuting_backpack(self.trained_model.to(DEVICE), self.total_loader, self.client_loader, loss_class(), 10, DEVICE, NUM_POWER_ITERS )["diag_by_name"]

    def plot_information_parameters_tradeoff(self, method='information', whitelist=None, blacklist=None):
        """Plot the information-vs-parameters tradeoff for this test's client information."""
        plot_information_parameters_tradeoff(self.client_information, method, whitelist=whitelist, blacklist=blacklist)

    def run_test(self, test_params_dict):

        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        logging.info(f"Unlearning: Method={unlearning_method}, Percentage={unlearning_percentage}, RetrainEpochs={retrain_epochs}")

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)
        num_reset_params = 0
        for indices_tensor in informative_params.values():
            if indices_tensor is not None and indices_tensor.numel() > 0: 
                 num_reset_params += indices_tensor.shape[0]

        if num_reset_params != 0:
            reset_model = self.model_class()
            reset_state_dict = reset_parameters(self.trained_model, informative_params)
            reset_model.load_state_dict(reset_state_dict)

            retrainer = UnlearnNet(reset_model, informative_params) 
            self.trainer_function(retrainer, self.loss_class(), self.retrain_subsets, self.eval_subsets, retrain_epochs)
            retrained_model = self.model_class()
            retrained_model.load_state_dict(retrainer.get_retrained_params())

            reset_params_percentage = num_reset_params / self.num_total_params * 100
            random_params = find_informative_params(self.client_information, 'random', reset_params_percentage, whitelist, blacklist)

            random_reset_model = self.model_class()
            random_reset_state_dict = reset_parameters(self.trained_model.cpu(), random_params)
            random_reset_model.load_state_dict(random_reset_state_dict)

            random_retrainer = UnlearnNet(random_reset_model, random_params)
            self.trainer_function(random_retrainer, self.loss_class(), self.retrain_subsets, self.eval_subsets, retrain_epochs)
            random_retrained_model = self.model_class()
            random_retrained_model.load_state_dict(random_retrainer.get_retrained_params())
        else:
            logging.warning("No parameters to reset, skipping unlearning and retraining.")
            reset_model = self.trained_model
            retrained_model = self.trained_model
            random_reset_model = self.trained_model
            random_retrained_model = self.trained_model
            reset_params_percentage = 0.0

        # Execute tests
        extra_results = {}

        extra_results['num_total_params'] = self.num_total_params
        extra_results['num_reset_params'] = num_reset_params
        extra_results['reset_params_percentage'] = reset_params_percentage

        logging.info(f"Percentage of reset parameters: {reset_params_percentage:.4f}% ({num_reset_params}/{self.num_total_params})")
        

        eval_test_results = {
            "reset": evaluate_model(reset_model, self.test_dataset),
            "retrained": evaluate_model(retrained_model, self.test_dataset),
            "random_reset": evaluate_model(random_reset_model, self.test_dataset),
            "random_retrained": evaluate_model(random_retrained_model, self.test_dataset)
        }
        eval_train_results = {
            "reset": evaluate_model(reset_model, self.train_dataset),
            "retrained": evaluate_model(retrained_model, self.train_dataset),
            "random_reset": evaluate_model(random_reset_model, self.train_dataset),
            "random_retrained": evaluate_model(random_retrained_model, self.train_dataset)
        }

        return eval_test_results, eval_train_results, extra_results
    
def get_datasets(init_params_dict):

    dataset_name = init_params_dict['dataset_name']
    model_name = init_params_dict['model_name']
    if dataset_name == 'breast_cancer':
        from BreastCancerDataset import BreastCancerDataset

        train_dataset = BreastCancerDataset('./data', split='train')
        test_dataset = BreastCancerDataset('./data', split='test')
        input_dim = len(train_dataset.input_columns)
        num_classes = len(train_dataset.classes)
        init_params_dict['num_classes'] = num_classes
        init_params_dict['input_dim'] = input_dim


    elif dataset_name == 'mnist':
        from torchvision.datasets import MNIST
        from torchvision import transforms

        if model_name == 'simple_cnn':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        else:
            raise ValueError("Unsupported model name for MNIST dataset")

        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        if model_name == 'simple_cnn':
            transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5), (0.5), (0.5))])
        elif model_name == 'resnet18':
            transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError("Unsupported model name for CIFAR10 dataset")

        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        from torchvision.datasets import CIFAR100
        from torchvision import transforms

        if model_name == 'simple_cnn':
            transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5), (0.5), (0.5))])
        elif model_name == 'resnet18':
            transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError("Unsupported model name for CIFAR100 dataset")

        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "FashionMNIST":
        from torchvision.datasets import FashionMNIST
        from torchvision import transforms

        if model_name == 'simple_cnn':
            transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5), (0.5), (0.5))])
        elif model_name == 'resnet18':
            transform = transforms.Compose([
                transforms.Resize(64),                      
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),                      
                transforms.Normalize(                      
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        else:
            raise ValueError("Unsupported model name for FashionMNIST dataset")

        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "cartelli":
        from torchvision.datasets import GTSRB
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((48, 48)), # Resize to a fixed size, e.g., 48x48
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171], # Mean for GTSRB pre-calculated
                std=[0.2672, 0.2564, 0.2629]   # Std for GTSRB pre-calculated
            )
        ])
        train_dataset = GTSRB(root='./data', split='train', download=True, transform=transform)
        test_dataset = GTSRB(root='./data', split='test', download=True, transform=transform)

    else:
        raise ValueError("Unsupported dataset name")

    return train_dataset, test_dataset

def get_clients_subsets(dataset, init_params_dict):
    num_clients = init_params_dict['num_clients']
    num_classes = init_params_dict['num_classes']
    distribution_type = init_params_dict['distribution_type']

    if distribution_type == 'preferential_class':
        if num_clients > num_classes:
            raise ValueError("Number of clients must be less than or equal to number of classes for preferential_class distribution")

        num_common_classes = num_classes - num_clients
        p_common = 1 / (num_common_classes + num_clients)
        p_preferred = p_common * num_clients
        class_distribution = np.zeros((num_clients, num_classes))
        for i in range(num_clients):
            for j in range(num_common_classes):
                class_distribution[i, j] = p_common
            class_distribution[i, num_common_classes+i] = p_preferred
        return split_dataset_by_class_distribution(dataset, class_distribution)
    
    elif distribution_type == 'categorical':
        if num_clients != num_classes:
            raise ValueError("Number of clients must be equal to number of classes for purely categorical distribution")
        
        return split_dataset_by_class_distribution(dataset, np.identity(num_classes))
    
    elif distribution_type == 'uniform':
        class_distribution = np.ones((num_clients, num_classes)) / num_classes
        return split_dataset_by_class_distribution(dataset, class_distribution)

    elif distribution_type == 'dirichlet':
        alpha = init_params_dict.get('dirichlet_alpha', 1)
        class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
        return split_dataset_by_class_distribution(dataset, class_distribution)

    elif distribution_type == 'random':
        lengths = [1 / num_clients] * num_clients
        return torch.utils.data.random_split(dataset, lengths)

    elif distribution_type == 'BC_targeted':
        from BreastCancerDataset import BreastCancerDataset, split_by_age
        if not isinstance(dataset, BreastCancerDataset):
            raise ValueError("BC_targeted distribution can only be used with BreastCancerDataset")
        return split_by_age(dataset)
    else:
        raise ValueError("Unsupported distribution type")

class FLNet(nn.Sequential):
            def __init__(self):
                super(FLNet, self).__init__(
                    nn.Conv2d(1, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                    )

class FLNet2(nn.Sequential):
    def __init__(self):
        super(FLNet2, self).__init__(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                    

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 43)
        )

class FeedForwardNN(nn.Sequential):
    def __init__(self, input_size, num_classes):
        dim_step = input_size // 4
        hidden_sizes = [input_size - dim_step, input_size - 2 * dim_step, input_size - 3 * dim_step]

        super(FeedForwardNN, self).__init__(
            # nn.Linear(input_size, input_size),
            # nn.ReLU(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            # nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            # nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[2], hidden_sizes[2]),
            # nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes)
        )

def create_resnet(init_params_dict):
    num_classes = init_params_dict['num_classes']
    model = resnet18(num_classes=num_classes) 
    if init_params_dict['dataset_name'] == 'mnist':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    return model

def get_model_class(init_params_dict):
    model_name = init_params_dict['model_name']
    if model_name == 'simple_cnn':
        init_params_dict['info_use_converter'] = False
        return FLNet 
    elif model_name == 'resnet18':
        init_params_dict['info_use_converter'] = True
        return functools.partial(create_resnet, init_params_dict=init_params_dict.copy())
    elif model_name == 'complex_cnn':
        init_params_dict['info_use_converter'] = False
        return FLNet2
    elif model_name == 'feedforward_nn':
        input_dim = init_params_dict['input_dim']
        num_classes = init_params_dict['num_classes']
        init_params_dict['info_use_converter'] = False
        return functools.partial(FeedForwardNN, input_size=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}") 

def get_loss_class(init_params_dict):
    loss_name = init_params_dict['loss_name']
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss
    elif loss_name == 'mse':
        return nn.MSELoss
    else:
        raise ValueError("Unsupported loss name")
    
def simple_trainer(model, loss_fn, train_subsets, val_subsets, epochs, init_params_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_dataset = concatenate_subsets(train_subsets)
    val_dataset = concatenate_subsets(val_subsets)

    dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, EVAL_BATCH_SIZE, shuffle=False)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_params_dict['learning_rate'], momentum=init_params_dict['momentum'])

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", leave=False):     
        loss = None
        loss_accum = 0.0
        n_batches = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            n_batches += 1
        val_loss_accum = 0.0
        val_n_batches = 0
        for val_inputs, val_targets in val_dataloader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_targets)
                val_loss_accum += val_loss.item()
                val_n_batches += 1
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss_accum / n_batches}, Val Loss: {val_loss_accum / val_n_batches}")

    model.eval()
    return model.cpu()

def fedavg_trainer(model, loss_fn, subsets, epochs, comm_tracker=None, init_params_dict=None):
    """
    Federated Averaging Trainer.
    
    Args:
        model: Global model
        loss_fn: Loss function
        subsets: List of client datasets
        epochs: Number of global communication rounds
        comm_tracker: CommunicationTracker instance
        init_params_dict: Dictionary containing 'local_epochs', 'participation_rate', 'lr'
    """
    import copy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    local_epochs = init_params_dict.get('local_epochs', 1) if init_params_dict else 1
    participation_rate = init_params_dict.get('participation_rate', 1.0) if init_params_dict else 1.0
    lr = init_params_dict.get('lr', 1e-3) if init_params_dict else 1e-3
    
    num_clients = len(subsets)
    num_participating = max(1, int(participation_rate * num_clients))
    
    model.to(device)
    
    for round in tqdm(range(epochs), desc="FedAvg Rounds", unit="round", leave=False):
        # 1. Select participating clients
        participating_idxs = np.random.choice(num_clients, num_participating, replace=False)
        
        # 2. Record communication
        if comm_tracker:
            comm_tracker.record_round(participating_idxs)
            
        # 3. Client Training
        local_weights = []
        local_sizes = []
        
        global_state = model.state_dict()
        
        for client_idx in participating_idxs:
            # Init local model with global weights
            client_model = copy.deepcopy(model) # Optimization: reuse instance if possible, but deepcopy is safer
            client_model.load_state_dict(global_state)
            client_model.train()
            
            # Local Optimizer
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr, weight_decay=1e-2)
            
            # Local Data
            train_loader = DataLoader(subsets[client_idx], batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            
            # Local Epochs
            for _ in range(local_epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            local_weights.append(client_model.state_dict())
            local_sizes.append(len(subsets[client_idx]))
            
        # 4. Aggregation (FedAvg)
        total_size = sum(local_sizes)
        avg_weights = copy.deepcopy(local_weights[0])
        
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * local_sizes[0]
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key] * local_sizes[i]
            avg_weights[key] = torch.div(avg_weights[key], total_size)
            
        model.load_state_dict(avg_weights)
        
    return model.cpu()

def get_trainer_function(init_params_dict):
    trainer_name = init_params_dict['trainer_name']
    if trainer_name == 'sgd':
        if 'learning_rate' not in init_params_dict:
            init_params_dict['learning_rate'] = 0.01
        if 'momentum' not in init_params_dict:
            init_params_dict['momentum'] = 0.9
        return functools.partial(simple_trainer, init_params_dict=init_params_dict)
    elif trainer_name == 'fedavg':
        return functools.partial(fedavg_trainer, init_params_dict=init_params_dict)
    else:
        raise ValueError(f"Unsupported trainer name: {trainer_name}")

def init_worker(device_queue):
    logging.getLogger().setLevel(logging.INFO)
    device = device_queue.get()
    set_device(device)

def run_tests_iter(iter, arg):
    test_path = arg['test_path']
    train_dataset = arg['train_dataset']
    test_dataset = arg['test_dataset']
    clients_subsets = arg['clients_subsets']
    model_class = arg['model_class']
    loss_class = arg['loss_class']
    trainer_function = arg['trainer_function']
    init_params_dict = arg['init_params_dict']
    test_params_dicts = arg['test_params_dicts']
    poisoned_backdoor_dataset = arg['poisoned_backdoor_dataset']
    clean_backdoor_dataset = arg['clean_backdoor_dataset']
    plot = arg.get('plot', False)

    test_iter_path = os.path.join(test_path, f"test_{iter}")
    os.makedirs(test_iter_path)

    log_file_handler = logging.FileHandler(os.path.join(test_iter_path, f"test_{iter}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)
    logging.info(f"--- Starting Test Iteration {iter} ---")

    logging.info(f"Using device: {DEVICE}")

    test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict, poisoned_backdoor_dataset, clean_backdoor_dataset)

    if plot:
        test_instance.plot_information_parameters_tradeoff()

    with open(os.path.join(test_iter_path, "initial_eval_test_results.pkl"), 'wb') as f:
        pickle.dump(test_instance.init_eval_test_results, f)
    with open(os.path.join(test_iter_path, "initial_eval_train_results.pkl"), 'wb') as f:
        pickle.dump(test_instance.init_eval_train_results, f)

    acc_eval_test_results = []
    acc_eval_train_results = []
    acc_extra_results = []
    errors = []
    for i, test_params_dict in enumerate(tqdm(test_params_dicts, desc=f"Unlearning tests", leave=False)):
        try:
            eval_test_result, eval_train_result, test_extra_result = test_instance.run_test(test_params_dict)
            acc_eval_test_results.append(eval_test_result)
            acc_eval_train_results.append(eval_train_result)
            acc_extra_results.append(test_extra_result)
        except Exception as e:
            logging.error(f"Error in test {i} of iteration {iter}: {str(e)}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            logging.error(f"Traceback:\n{traceback_str}")
            errors.append(i)
            acc_eval_test_results.append({'error': str(e)})
            acc_eval_train_results.append({'error': str(e)})
            acc_extra_results.append({'error': str(e)})

    valid_test_results = [r for r in acc_eval_test_results if 'error' not in r]
    valid_train_results = [r for r in acc_eval_train_results if 'error' not in r]
    valid_extra_results = [r for r in acc_extra_results if 'error' not in r]

    eval_test_results = {}
    if valid_test_results:
        for key in valid_test_results[0].keys():
            eval_test_results[f"{key}__pred"] = np.stack(
                [r[key]["pred"] for r in valid_test_results]
            )
            eval_test_results[f"{key}__loss"] = np.stack(
                [r[key]["loss"] for r in valid_test_results]
            )

    with open(os.path.join(test_iter_path, "eval_test_results.npz"), 'wb') as f:
        np.savez(f, **eval_test_results)

    eval_train_results = {}
    if valid_train_results:
        for key in valid_train_results[0].keys():
            eval_train_results[f"{key}__pred"] = np.stack(
                [r[key]["pred"] for r in valid_train_results]
            )
            eval_train_results[f"{key}__loss"] = np.stack(
                [r[key]["loss"] for r in valid_train_results]
            )

    with open(os.path.join(test_iter_path, "eval_train_results.npz"), 'wb') as f:
        np.savez(f, **eval_train_results)

    extra_results = {}
    if valid_extra_results:
        for key in valid_extra_results[0].keys():
            extra_results[key] = [r[key] for r in valid_extra_results]

    with open(os.path.join(test_iter_path, "extra_results.pkl"), 'wb') as f:
        pickle.dump(extra_results, f)

    logging.info(f"--- Finished Test Iteration {iter} ---")
    logging.getLogger().removeHandler(log_file_handler)
    log_file_handler.close()
    return errors


def run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1, devices=None, save_models=False, plot=False):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_name = init_params_dict['test_name']
    logging.info(f"Starting test suite: {test_name}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info(f"Created base save directory: {save_path}")
    test_path = os.path.join(save_path, test_name)
    if os.path.exists(test_path):
        orig_path = test_path
        i = 1
        test_path = f"{orig_path} ({i})"
        while os.path.exists(test_path): i += 1; test_path = f"{orig_path} ({i})"
        logging.warning(f"Test directory '{orig_path}' already exists.")
    os.makedirs(test_path)
    logging.info(f"Created test suite directory: {test_path}")

    log_file_handler = logging.FileHandler(os.path.join(test_path, f"{test_name}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)

    logging.info("Initial Configuration:")
    for key, value in init_params_dict.items():
        logging.info(f"  {key}: {value}")
    logging.info("-" * 30)

    init_params_dict_path = os.path.join(test_path, "init_params.pkl")
    test_params_dicts_path = os.path.join(test_path, "test_params.pkl")

    with open(init_params_dict_path, 'wb') as f: pickle.dump(init_params_dict, f)
    with open(test_params_dicts_path, 'wb') as f: pickle.dump(test_params_dicts, f)

    num_tests = init_params_dict['num_tests']

    train_dataset, test_dataset = get_datasets(init_params_dict)
    clients_subsets = get_clients_subsets(train_dataset, init_params_dict)
    model_class = get_model_class(init_params_dict)
    loss_class = get_loss_class(init_params_dict) 
    trainer_function = get_trainer_function(init_params_dict)

    poisoned_backdoor_dataset = None
    clean_backdoor_dataset = None
    
    if init_params_dict.get('poison', False):
        logging.info("Poisoning is enabled. Applying backdoor attack...")
        clients_subsets, poisoned_backdoor_dataset, clean_backdoor_dataset = create_poisoned_data(clients_subsets, init_params_dict)
        logging.info("Poisoning complete.")
    else:
        logging.info("Poisoning is disabled.")

    client_indices = [subset.indices for subset in clients_subsets]
    with open(os.path.join(test_path, "clients_indices.pkl"), 'wb') as f:
        pickle.dump(client_indices, f)
    
    labels = {
        'train': [label for _, label in train_dataset],
        'test': [label for _, label in test_dataset]
    }
    with open(os.path.join(test_path, "labels.pkl"), 'wb') as f:
        pickle.dump(labels, f)

    arg = {
        'test_path': test_path,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'clients_subsets': clients_subsets,
        'model_class': model_class,
        'loss_class': loss_class,
        'trainer_function': trainer_function,
        'init_params_dict': init_params_dict,
        'test_params_dicts': test_params_dicts,
        'poisoned_backdoor_dataset': poisoned_backdoor_dataset,
        'clean_backdoor_dataset': clean_backdoor_dataset,
        'save_models' : save_models,
        'plot': plot
    }

    if num_workers == 1:
        with logging_redirect_tqdm():
            for i in tqdm(range(num_tests), desc="Running repeated tests"):
                logging.getLogger().removeHandler(log_file_handler)
                errors = run_tests_iter(i, arg)    
                logging.getLogger().addHandler(log_file_handler)
                if len(errors) > 0:
                    logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")
    else:
        logging.info(f"Using {num_workers} workers for parallel processing.")
        if devices is None:
            logging.info(f"No devices provided, using default device {DEVICE} for all workers.")
            devices = [DEVICE] * num_workers
        elif len(devices) != num_workers:
            logging.error(f"Number of devices provided ({len(devices)}) does not match number of workers ({num_workers}). Using default device {DEVICE} for all workers.")
            devices = [DEVICE] * num_workers
        else:
            logging.info(f"Using provided devices: {devices}")

        device_queue = Queue(num_workers)
        for device in devices:
            device_queue.put(device)
                
        logging.getLogger().removeHandler(log_file_handler)
        os.environ['TQDM_DISABLE'] = '1'
        
        with Pool(num_workers, initializer=init_worker, initargs=(device_queue,)) as pool:
            iters_errors = pool.starmap(run_tests_iter, [(i, arg) for i in range(num_tests)])

        os.environ['TQDM_DISABLE'] = '0'
        logging.getLogger().addHandler(log_file_handler)

        for i, errors in enumerate(iters_errors):
            if len(errors) > 0:
                logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")

    logging.info(f"Test suite '{test_name}' completed")

    if plot:
        plot_experiment_results(test_path)


# ─── Analysis & plotting helpers ──────────────────────────────────────────────

def _merge_initial_results(accumulated_results):
    merged_results = {}
    for key in accumulated_results[0].keys():
        merged_results[key] = [[accumulated_results[j][key] for j in range(len(accumulated_results))]]
    return merged_results

def _merge_results(accumulated_results):
    merged_results = {}
    for key in accumulated_results[0].keys():
        merged_results[key] = []
        first_entry = accumulated_results[0][key]
        if isinstance(first_entry, dict) and "pred" in first_entry:
            num_tests_inner = first_entry["pred"].shape[0]
            for i in range(num_tests_inner):
                merged_results[key].append(
                    [{"pred": accumulated_results[j][key]["pred"][i],
                      "loss": accumulated_results[j][key]["loss"][i]}
                     for j in range(len(accumulated_results))]
                )
        else:
            for i in range(len(first_entry)):
                merged_results[key].append(
                    [accumulated_results[j][key][i] for j in range(len(accumulated_results))]
                )
    return merged_results

def _unpack_eval_results(npz_dict):
    if any("__" in key for key in npz_dict.keys()):
        unpacked = {}
        for key, value in npz_dict.items():
            if "__" not in key:
                continue
            model_key, suffix = key.split("__", 1)
            if suffix not in {"pred", "loss"}:
                continue
            unpacked.setdefault(model_key, {})[suffix] = value
        if unpacked:
            return unpacked
    return npz_dict

def _get_predictions(outputs):
    if isinstance(outputs, dict) and "pred" in outputs:
        return np.array(outputs["pred"])
    outputs = np.array(outputs)
    return outputs if outputs.ndim == 1 else np.argmax(outputs, axis=1)

def _compute_accuracies(merged_results, labels, subset=None):
    labels = np.array(labels)
    idx = slice(None) if subset is None else subset
    accuracies = {}
    for model_key in merged_results.keys():
        accuracies[model_key] = []
        for test_outputs in merged_results[model_key]:
            test_accuracies = []
            for outputs in test_outputs:
                predictions = _get_predictions(outputs)
                acc = np.mean(predictions[idx] == labels[idx])
                test_accuracies.append(acc)
            accuracies[model_key].append(test_accuracies)
    return accuracies

def _compute_shadow_losses(merged_initial_results, labels, subset=None):
    idx = slice(None) if subset is None else subset
    labels_tensor = torch.tensor(labels)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = {}
    for model_key in merged_initial_results.keys():
        losses_list = []
        for outputs in merged_initial_results[model_key][0]:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = torch.tensor(np.array(outputs["loss"])[idx])
            else:
                outputs_tensor = torch.tensor(outputs)
                loss = loss_fn(outputs_tensor[idx], labels_tensor[idx])
            losses_list.append(loss)
        losses[model_key] = torch.stack(losses_list, dim=0).numpy()
    return losses

def _compute_shadow_losses_dists(losses_members, losses_nonmembers, global_var=False):
    member_dists = {}
    nonmember_dists = {}
    for model_key in losses_members.keys():
        avg_member = np.mean(losses_members[model_key], axis=0)
        avg_nonmember = np.mean(losses_nonmembers[model_key], axis=0)
        if global_var:
            combined = np.concatenate([losses_members[model_key], losses_nonmembers[model_key]], axis=1)
            std = np.std(combined, ddof=1)
            std_member = std_nonmember = std
        else:
            std_member = np.std(losses_members[model_key], axis=0, ddof=1)
            std_nonmember = np.std(losses_nonmembers[model_key], axis=0, ddof=1)
        member_dists[model_key] = {'avg': avg_member, 'std': std_member}
        nonmember_dists[model_key] = {'avg': avg_nonmember, 'std': std_nonmember}
    return member_dists, nonmember_dists

def _log_gaussian_pdf(x, mu, sigma):
    return -np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

def _compute_lira_scores(losses_in, losses_out, merged_results, labels, subset=None):
    labels_tensor = torch.tensor(labels)
    idx = slice(None) if subset is None else subset
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    lira_scores = {}
    for model_key in merged_results.keys():
        lira_scores[model_key] = []
        for test_outputs in merged_results[model_key]:
            test_lira_scores = []
            for outputs in test_outputs:
                if isinstance(outputs, dict) and "loss" in outputs:
                    losses = np.array(outputs["loss"])[idx]
                else:
                    outputs_tensor = torch.tensor(outputs)
                    losses = loss_fn(outputs_tensor[idx], labels_tensor[idx]).numpy()
                lira = (_log_gaussian_pdf(losses, losses_in['avg'], losses_in['std'])
                        - _log_gaussian_pdf(losses, losses_out['avg'], losses_out['std']))
                test_lira_scores.append(lira)
            lira_scores[model_key].append(test_lira_scores)
    return lira_scores

def _compute_roc_curves(lira_member_scores, lira_nonmember_scores):
    from sklearn.metrics import roc_curve, auc as sk_auc
    roc_curves_dict = {}
    for model_key in lira_member_scores.keys():
        roc_curves_dict[model_key] = {'fpr': [], 'tpr': [], 'auc': []}
        for test_idx in range(len(lira_member_scores[model_key])):
            test_fprs, test_tprs, test_aucs = [], [], []
            for iter_idx in range(len(lira_member_scores[model_key][test_idx])):
                member_scores = lira_member_scores[model_key][test_idx][iter_idx]
                nonmember_scores = lira_nonmember_scores[model_key][test_idx][iter_idx]
                member_scores = member_scores[~np.isnan(member_scores)]
                nonmember_scores = nonmember_scores[~np.isnan(nonmember_scores)]
                all_scores = np.concatenate([member_scores, nonmember_scores])
                lira_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
                fpr, tpr, _ = roc_curve(lira_labels, all_scores)
                test_fprs.append(fpr)
                test_tprs.append(tpr)
                test_aucs.append(sk_auc(fpr, tpr))
            roc_curves_dict[model_key]['fpr'].append(test_fprs)
            roc_curves_dict[model_key]['tpr'].append(test_tprs)
            roc_curves_dict[model_key]['auc'].append(test_aucs)
    return roc_curves_dict

def _compute_tpr_at_fpr(roc_curve_data, target_fpr=0.01):
    tpr_at_fpr = {}
    for model_key in roc_curve_data.keys():
        tpr_at_fpr[model_key] = []
        for i in range(len(roc_curve_data[model_key]['fpr'])):
            test_tprs = []
            for j in range(len(roc_curve_data[model_key]['fpr'][i])):
                fpr = roc_curve_data[model_key]['fpr'][i][j]
                tpr = roc_curve_data[model_key]['tpr'][i][j]
                idx = np.where(fpr <= target_fpr)[0]
                test_tprs.append(tpr[idx[-1]] if len(idx) > 0 else 0.0)
            tpr_at_fpr[model_key].append(test_tprs)
    return tpr_at_fpr

def _summarize_baseline(values):
    values = np.array(values)
    return float(np.mean(values)), float(np.std(values))

def _add_baseline_band(ax, mean, std, label, color):
    ax.axhline(mean, color=color, linewidth=1.6, label=label)
    if std > 0:
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.12)

def _plot_metric_vs_unlearning(ax, x_values, metric_dict, label_map, ylabel, title, baselines=None):
    x_values = np.array(x_values)
    x_mean = x_values if x_values.ndim == 1 else np.mean(x_values, axis=1)
    for key, style in label_map.items():
        if key not in metric_dict:
            continue
        y = np.array(metric_dict[key])
        if y.ndim == 1:
            y_mean, y_std = y, np.zeros_like(y)
        else:
            if y.shape[0] != x_mean.shape[0] and y.shape[1] == x_mean.shape[0]:
                y = y.T
            y_mean = np.mean(y, axis=1)
            y_std = np.std(y, axis=1)
        sort_idx = np.argsort(x_mean)
        ax.errorbar(x_mean[sort_idx], y_mean[sort_idx], yerr=y_std[sort_idx],
                    label=style["label"], linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"), linewidth=1.6, markersize=5, capsize=3)
    if baselines:
        for baseline in baselines:
            _add_baseline_band(ax, baseline["mean"], baseline["std"], baseline["label"], baseline["color"])
    ax.set_title(title)
    ax.set_xlabel("Unlearning Percentage")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)


def plot_experiment_results(test_path, num_tests=None, target_fpr=0.001):
    """Load saved experiment results and produce diagnostic plots.

    Parameters
    ----------
    test_path : str
        Path to the experiment directory produced by run_repeated_tests.
    num_tests : int or None
        Number of test iterations to load. If None, inferred from saved init_params.pkl.
    target_fpr : float
        FPR threshold for TPR@FPR computation.
    """
    import matplotlib.pyplot as plt

    with open(os.path.join(test_path, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(test_path, "clients_indices.pkl"), "rb") as f:
        client_indices = pickle.load(f)

    saved_params = {}
    params_path = os.path.join(test_path, "init_params.pkl")
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            saved_params = pickle.load(f)
    if num_tests is None:
        num_tests = saved_params.get('num_tests', 1)

    target_client_idx = saved_params.get('target_client', 0)
    target_indices = client_indices[target_client_idx]

    acc_initial_eval_test, acc_initial_eval_train = [], []
    acc_tests_eval_test, acc_tests_eval_train, acc_tests_extra = [], [], []

    for i in range(num_tests):
        iter_path = os.path.join(test_path, f"test_{i}")
        with open(os.path.join(iter_path, "initial_eval_test_results.pkl"), "rb") as f:
            acc_initial_eval_test.append(pickle.load(f))
        with open(os.path.join(iter_path, "initial_eval_train_results.pkl"), "rb") as f:
            acc_initial_eval_train.append(pickle.load(f))
        with open(os.path.join(iter_path, "eval_test_results.npz"), "rb") as f:
            acc_tests_eval_test.append(_unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(iter_path, "eval_train_results.npz"), "rb") as f:
            acc_tests_eval_train.append(_unpack_eval_results(dict(np.load(f))))
        with open(os.path.join(iter_path, "extra_results.pkl"), "rb") as f:
            acc_tests_extra.append(pickle.load(f))

    initial_eval_test = _merge_initial_results(acc_initial_eval_test)
    initial_eval_train = _merge_initial_results(acc_initial_eval_train)
    unlearned_eval_test = _merge_results(acc_tests_eval_test)
    unlearned_eval_train = _merge_results(acc_tests_eval_train)
    unlearned_extra = _merge_results(acc_tests_extra)

    initial_test_accuracies = _compute_accuracies(initial_eval_test, labels["test"])
    unlearned_test_accuracies = _compute_accuracies(unlearned_eval_test, labels["test"])

    shadow_target_losses = _compute_shadow_losses(initial_eval_train, labels["train"], subset=target_indices)
    shadow_test_losses = _compute_shadow_losses(initial_eval_test, labels["test"])
    shadow_target_dists, shadow_test_dists = _compute_shadow_losses_dists(
        shadow_target_losses, shadow_test_losses, global_var=True)

    unlearned_lira_target = _compute_lira_scores(
        shadow_target_dists['shadow_in'], shadow_target_dists['shadow_out'],
        unlearned_eval_train, labels["train"], subset=target_indices)
    unlearned_lira_test = _compute_lira_scores(
        shadow_test_dists['shadow_in'], shadow_test_dists['shadow_out'],
        unlearned_eval_test, labels["test"])
    unlearned_roc_curves = _compute_roc_curves(unlearned_lira_target, unlearned_lira_test)

    initial_lira_target = _compute_lira_scores(
        shadow_target_dists['shadow_in'], shadow_target_dists['shadow_out'],
        initial_eval_train, labels["train"], subset=target_indices)
    initial_lira_test = _compute_lira_scores(
        shadow_test_dists['shadow_in'], shadow_test_dists['shadow_out'],
        initial_eval_test, labels["test"])
    initial_roc_curves = _compute_roc_curves(initial_lira_target, initial_lira_test)

    unlearned_tpr_at_fpr = _compute_tpr_at_fpr(unlearned_roc_curves, target_fpr=target_fpr)
    initial_tpr_at_fpr = _compute_tpr_at_fpr(initial_roc_curves, target_fpr=target_fpr)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: ROC curves for initial models
    plt.figure(figsize=(8, 6))
    for model_key, style in {
        "trained": {"label": "Trained", "linestyle": "-", "color": "tab:blue"},
        "shadow_out": {"label": "Shadow Out", "linestyle": "--", "color": "tab:orange"},
    }.items():
        fpr_list = initial_roc_curves[model_key]['fpr'][0]
        tpr_list = initial_roc_curves[model_key]['tpr'][0]
        for fpr, tpr in zip(fpr_list, tpr_list):
            plt.plot(fpr, tpr, linestyle=style["linestyle"], color=style["color"], alpha=0.3)
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fpr_list, tpr_list)], axis=0)
        plt.plot(mean_fpr, mean_tpr, linestyle=style["linestyle"], color=style["color"],
                 label=style["label"], linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Initial Models')
    plt.legend(frameon=False)
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

    accuracy_styles = {
        "reset":           {"label": "Reset",           "linestyle": "-",  "marker": "o"},
        "random_reset":    {"label": "Random Reset",    "linestyle": "--", "marker": "s"},
        "retrained":       {"label": "Retrained",       "linestyle": "-",  "marker": "D"},
        "random_retrained":{"label": "Random Retrained","linestyle": "--", "marker": "^"},
    }

    trained_acc_mean, trained_acc_std = _summarize_baseline(initial_test_accuracies["trained"][0])
    benchmark_acc_mean, benchmark_acc_std = _summarize_baseline(initial_test_accuracies["shadow_out"][0])

    # Plot 2: Test accuracy vs unlearning percentage
    _, ax = plt.subplots(figsize=(9, 5))
    _plot_metric_vs_unlearning(
        ax=ax, x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_test_accuracies, label_map=accuracy_styles,
        ylabel="Test Accuracy", title="Test Accuracy vs Unlearning Percentage",
        baselines=[
            {"mean": trained_acc_mean, "std": trained_acc_std,
             "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_acc_mean, "std": benchmark_acc_std,
             "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    trained_tpr_mean, trained_tpr_std = _summarize_baseline(initial_tpr_at_fpr["trained"][0])
    benchmark_tpr_mean, benchmark_tpr_std = _summarize_baseline(initial_tpr_at_fpr["shadow_out"][0])

    # Plot 3: TPR at FPR vs unlearning percentage
    _, ax = plt.subplots(figsize=(9, 5))
    _plot_metric_vs_unlearning(
        ax=ax, x_values=unlearned_extra["reset_params_percentage"],
        metric_dict=unlearned_tpr_at_fpr, label_map=accuracy_styles,
        ylabel=f"TPR at FPR={target_fpr*100:.2f}%",
        title=f"TPR at FPR={target_fpr*100:.2f}% vs Unlearning Percentage",
        baselines=[
            {"mean": trained_tpr_mean, "std": trained_tpr_std,
             "label": "Trained Baseline", "color": "tab:blue"},
            {"mean": benchmark_tpr_mean, "std": benchmark_tpr_std,
             "label": "Benchmark Baseline", "color": "tab:gray"},
        ],
    )
    plt.tight_layout()
    plt.show()

    print(f"Trained baseline TPR at FPR={target_fpr*100:.2f}%: "
          f"{trained_tpr_mean:.4f} ± {trained_tpr_std:.4f}")


if __name__ == "__main__":
    save_path = r'.\stat_tests\NEW_TESTER'

    num_tests = 1
    num_workers = 1
    set_batch_sizes(128,128,128,128)

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_random',

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 1,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': "diag_ggn_mc",
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }
    
    test_params_dict_0 = test_params_dict.copy()
    test_params_dict_0['subtest'] = 0
    test_params_dict_0['unlearning_method'] = 'information'

    percentages = np.arange(0, 10, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers, plot=True)
