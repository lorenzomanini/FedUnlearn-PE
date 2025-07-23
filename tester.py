from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets, create_poisoned_data, poisoning_data
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters, mia_attack
from fisherunlearn import UnlearnNet

import fisherunlearn

import os
import pickle
import random
import logging
import functools

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
INFO_BATCH_SIZE = 15
MIA_BATCH_SIZE = 128

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

class TestParamsDict(TypedDict):
    subtest: int
    unlearning_method: Literal['information', 'parameters']
    unlearning_percentage: float
    retrain_epochs: int
    tests: list[Literal['test_accuracy', 'target_accuracy', 'clients_accuracies', 'class_accuracies', 'mia', 'categorical_accuracies', 'poison_accuracy']]
    mia_classifier_types: list[Literal['nn', 'logistic', 'svm']]
    whitelist: list[str]
    blacklist: list[str]


class Test:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict={}, attack_eval_dataset=None, unlearning_eval_dataset=None):

        # Initialize parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clients_datasets = clients_subsets
        self.target_client = init_params_dict.get('target_client', 0)

        self.target_dataset = self.clients_datasets[self.target_client]
        self.benchmark_datasets = [subset for i, subset in enumerate(self.clients_datasets) if i != self.target_client]

        self.classes_datasets = split_dataset_by_class_distribution(self.test_dataset, np.identity(init_params_dict['num_classes']))
        self.categorical_test_datasets = [subset for i, subset in enumerate(self.classes_datasets) if i != self.target_client]

        self.attack_eval_dataset = attack_eval_dataset
        self.unlearning_eval_dataset = unlearning_eval_dataset

        self.model_class = model_class
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        self.train_epochs = init_params_dict['train_epochs']
        self.info_use_converter = init_params_dict.get('info_use_converter', True)


        self.trained_model = None
        self.benchmark_model = None
        self.client_information = None


        logging.info("Training model...") 
        self.trained_model = self.trainer_function(self.model_class(), self.loss_class(), self.clients_datasets, self.train_epochs)

        logging.info("Training benchmark model...") 
        self.benchmark_model = self.trainer_function(self.model_class(), self.loss_class(), self.benchmark_datasets, self.train_epochs)

        logging.info("Computing information...")
        hessian_method = init_params_dict.get('hessian_method', 'diag_ggn')
        if init_params_dict.get('use_FIM', False):
            logging.info("Using FIM for information computation...")
            self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_datasets, use_converter=self.info_use_converter, method=hessian_method, use_FIM=True)
        else:
            self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_datasets, use_converter=self.info_use_converter, method=hessian_method)


    def run_test(self, test_params_dict):

        # Apply unlearning method
        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        logging.info(f"Unlearning: Method={unlearning_method}, Percentage={unlearning_percentage}, RetrainEpochs={retrain_epochs}")

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)
        total_individual_reset_params = 0
        for name, indices_tensor in informative_params.items():
            if indices_tensor is not None and indices_tensor.numel() > 0: 
                 total_individual_reset_params += indices_tensor.shape[0]

        reset_model = self.model_class()
        reset_state_dict = reset_parameters(self.trained_model.cpu(), informative_params)
        reset_model.load_state_dict(reset_state_dict)

        retrainer = UnlearnNet(reset_model, informative_params) 
        self.trainer_function(retrainer, self.loss_class(), self.benchmark_datasets, retrain_epochs)
        retrained_model = self.model_class()
        retrained_model.load_state_dict(retrainer.get_retrained_params())


        # Execute tests
        result = {}

        result['total_individual_reset_params'] = total_individual_reset_params

        if 'test_accuracy' in test_params_dict['tests']:
            logging.info("Computing test accuracies...")

            try:
                result['trained_test_accuracy'] = self.trained_test_accuracy
                result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
            except AttributeError:
                self.trained_test_accuracy = compute_accuracy(self.trained_model, self.test_dataset)
                self.benchmark_test_accuracy = compute_accuracy(self.benchmark_model, self.test_dataset)
                result['trained_test_accuracy'] = self.trained_test_accuracy
                result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
            
            result['reset_test_accuracy'] = compute_accuracy(reset_model, self.test_dataset)
            result['retrained_test_accuracy'] = compute_accuracy(retrained_model, self.test_dataset)
        
        if 'target_accuracy' in test_params_dict['tests']:
            logging.info("Computing target accuracies...")
            try:
                result['trained_target_accuracy'] = self.trained_target_accuracy
                result['benchmark_target_accuracy'] = self.benchmark_target_accuracy
            except AttributeError:
                self.trained_target_accuracy = compute_accuracy(self.trained_model, self.target_dataset)
                self.benchmark_target_accuracy = compute_accuracy(self.benchmark_model, self.target_dataset)
                result['trained_target_accuracy'] = self.trained_target_accuracy
                result['benchmark_target_accuracy'] = self.benchmark_target_accuracy

            result['reset_target_accuracy'] = compute_accuracy(reset_model, self.target_dataset)
            result['retrained_target_accuracy'] = compute_accuracy(retrained_model, self.target_dataset)
        
        if 'clients_accuracies' in test_params_dict['tests']:
            logging.info("Computing clients accuracies...")
            try:
                result['trained_clients_accuracies'] = self.trained_clients_accuracies
                result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
            except AttributeError:
                self.trained_clients_accuracies = [compute_accuracy(self.trained_model, subset) for subset in self.clients_datasets]
                self.benchmark_clients_accuracies = [compute_accuracy(self.benchmark_model, subset) for subset in self.clients_datasets]
                result['trained_clients_accuracies'] = self.trained_clients_accuracies
                result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
            
            result['reset_clients_accuracies'] = [compute_accuracy(reset_model, subset) for subset in self.clients_datasets]
            result['retrained_clients_accuracies'] = [compute_accuracy(retrained_model, subset) for subset in self.clients_datasets]

        if 'class_accuracies' in test_params_dict['tests']:
            logging.info("Computing class accuracies...")
            try:
                result['trained_class_accuracies'] = self.trained_class_accuracies
                result['benchmark_class_accuracies'] = self.benchmark_class_accuracies
            except AttributeError:
                self.trained_class_accuracies = [compute_accuracy(self.trained_model, subset) for subset in self.classes_datasets]
                self.benchmark_class_accuracies = [compute_accuracy(self.benchmark_model, subset) for subset in self.classes_datasets]
                result['trained_class_accuracies'] = self.trained_class_accuracies
                result['benchmark_class_accuracies'] = self.benchmark_class_accuracies

            result['reset_class_accuracies'] = [compute_accuracy(reset_model, subset) for subset in self.classes_datasets]
            result['retrained_class_accuracies'] = [compute_accuracy(retrained_model, subset) for subset in self.classes_datasets]

        if 'categorical_accuracies' in test_params_dict['tests']:
            logging.info("Computing Categorical accuracies...")
            try:
                result['trained_cat_accuracy'] = self.trained_cat_accuracy
                result['benchmark_cat_accuracy'] = self.benchmark_cat_accuracy
            except AttributeError:
                self.trained_cat_accuracy = compute_accuracy(self.trained_model, self.categorical_test_datasets)
                self.benchmark_cat_accuracy = compute_accuracy(self.benchmark_model, self.categorical_test_datasets)
                result['trained_cat_accuracy'] = self.trained_cat_accuracy
                result['benchmark_cat_accuracy'] = self.benchmark_cat_accuracy
            
            result['reset_test_accuracy'] = compute_accuracy(reset_model, self.categorical_test_datasets)
            result['retrained_test_accuracy'] = compute_accuracy(retrained_model, self.categorical_test_datasets)
        
        if 'attack_success_rate' in test_params_dict['tests'] and self.attack_eval_dataset:
            logging.info("Computing Attack Success Rate (ASR)...")
            try:
                result['trained_asr'] = self.trained_asr_accuracy
                result['benchmark_asr'] = self.benchmark_asr_accuracy
            except AttributeError:
                self.trained_asr_accuracy = compute_accuracy(self.trained_model, self.attack_eval_dataset)
                self.benchmark_asr_accuracy = compute_accuracy(self.benchmark_model, self.attack_eval_dataset)
                result['trained_asr'] = self.trained_asr_accuracy
                result['benchmark_asr'] = self.benchmark_asr_accuracy
                
            result['reset_asr'] = compute_accuracy(reset_model, self.attack_eval_dataset)
            result['retrained_asr'] = compute_accuracy(retrained_model, self.attack_eval_dataset)
        
        if 'unlearning_accuracy' in test_params_dict['tests'] and self.unlearning_eval_dataset:
            logging.info("Computing Unlearning Accuracy on poisoned data...")
            try:
                result['trained_unlearning_accuracy'] = self.trained_unlearning_accuracy
                result['benchmark_asr_accuracy'] = self.benchmark_asr_accuracy
            except AttributeError:
                self.trained_unlearning_accuracy = compute_accuracy(self.trained_model, self.trained_unlearning_accuracy)
                self.benchmark_asr_accuracy = compute_accuracy(self.benchmark_model, self.unlearning_eval_dataset)
                result['trained_unlearning_accuracy'] = self.trained_unlearning_accuracy
                result['benchmark_asr_accuracy'] = self.benchmark_asr_accuracy
            result['reset_unlearning_accuracy'] = compute_accuracy(reset_model, self.unlearning_eval_dataset)
            result['retrained_unlearning_accuracy'] = compute_accuracy(retrained_model, self.unlearning_eval_dataset)

        if 'mia' in test_params_dict['tests']:
            logging.info("Running MIA...")
            for classifier_type in test_params_dict['mia_classifier_types']:
                logging.info(f"Classifier type: {classifier_type}")

                try:
                    result[f'trained_mia_{classifier_type}'] = self.trained_mia
                    result[f'benchmark_mia_{classifier_type}'] = self.benchmark_mia
                except AttributeError:
                    self.trained_mia = mia_attack(self.trained_model, self.target_dataset, self.test_dataset, classifier_type)
                    self.benchmark_mia = mia_attack(self.benchmark_model, self.target_dataset, self.test_dataset, classifier_type)
                    result[f'trained_mia_{classifier_type}'] = self.trained_mia
                    result[f'benchmark_mia_{classifier_type}'] = self.benchmark_mia

                result[f'reset_mia_{classifier_type}'] = mia_attack(reset_model, self.target_dataset, self.test_dataset, classifier_type)
                result[f'retrained_mia_{classifier_type}'] = mia_attack(retrained_model, self.target_dataset, self.test_dataset, classifier_type)

        return result
    
def get_datasets(init_params_dict):

    dataset_name = init_params_dict['dataset_name']
    model_name = init_params_dict['model_name']
    if dataset_name == 'mnist':
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
        alpha = 1
        class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
        return split_dataset_by_class_distribution(dataset, class_distribution)

    elif distribution_type == 'random':
        lengths = [1 / num_clients] * num_clients
        return torch.utils.data.random_split(dataset, lengths)

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
    
def simple_trainer(model, loss_fn, subsets, epochs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.train()

            dataloader = DataLoader(concatenate_subsets(subsets), TRAIN_BATCH_SIZE, shuffle=True)
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            for epoch in tqdm(range(epochs), desc="Training", unit="epoch", leave=False):
                loss = None
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets) 
                    loss.backward()
                    optimizer.step()
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

            model.eval()
            return model.cpu()

def get_trainer_function(init_params_dict):
    trainer_name = init_params_dict['trainer_name']
    if trainer_name == 'sgd':
        return simple_trainer
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
    attack_eval_dataset = arg['attack_eval_dataset']
    unlearning_eval_dataset = arg['unlearning_eval_dataset']

    test_iter_path = os.path.join(test_path, f"test_{iter}")
    os.makedirs(test_iter_path)

    log_file_handler = logging.FileHandler(os.path.join(test_iter_path, f"test_{iter}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)
    logging.info(f"--- Starting Test Iteration {iter} ---")

    logging.info(f"Using device: {DEVICE}")

    trained_model_path = os.path.join(test_iter_path, "trained_model.pth")
    benchmark_model_path = os.path.join(test_iter_path, "benchmark_model.pth")
    client_information_path = os.path.join(test_iter_path, "client_information.pkl")
    test_results_path = os.path.join(test_iter_path, "test_results.pkl")

    test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict, attack_eval_dataset, unlearning_eval_dataset)
    
    torch.save(test_instance.trained_model.cpu().state_dict(), trained_model_path)
    torch.save(test_instance.benchmark_model.cpu().state_dict(), benchmark_model_path)

    with open(client_information_path, 'wb') as f:
        pickle.dump(test_instance.client_information, f)

    iteration_results = []
    errors = []
    for i, test_params_dict in enumerate(tqdm(test_params_dicts, desc=f"Unlearning tests", leave=False)):
        try:
            test_result = test_instance.run_test(test_params_dict)
            iteration_results.append(test_result)
        except Exception as e:
            logging.error(f"Error in test iteration {iter}: {e}")
            iteration_results.append({'error': str(e)})
            errors.append(i)
        with open(test_results_path, 'wb') as f:
            pickle.dump(iteration_results, f)

    logging.info(f"--- Finished Test Iteration {iter} ---")
    logging.getLogger().removeHandler(log_file_handler)
    log_file_handler.close()
    return errors


def run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1, devices=None):

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

    attack_eval_dataset = None
    unlearning_eval_dataset = None
    
    if init_params_dict.get('poison', False):
        logging.info("Poisoning is enabled. Applying backdoor attack...")
        clients_subsets, attack_eval_dataset, unlearning_eval_dataset = create_poisoned_data(clients_subsets, init_params_dict)
        logging.info("Poisoning complete.")
    else:
        logging.info("Poisoning is disabled.")

    client_indices = [subset.indices for subset in clients_subsets]
    clients_indices_path = os.path.join(test_path, "clients_indices.pkl")
    with open(clients_indices_path, 'wb') as f:
        pickle.dump(client_indices, f)

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
        'attack_eval_dataset': attack_eval_dataset,
        'unlearning_eval_dataset': unlearning_eval_dataset
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


if __name__ == "__main__":
    init_params_dict : InitParamsDict = {
        'test_name': 'test_info',

        'dataset_name': 'mnist',
        'num_clients': 10,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'categorical',     # Distribution type

        'model_name': 'simple_cnn',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 4,                # Initial training epochs

        'use_FIM' : False,
        'info_use_converter': False,

        'target_client': 0,               # Client to unlearn
        'num_tests': 1,                   # Number of independent repetitions
        'hessian_method': 'diag_hessian',      # Hessian method
        'poison' : True,
    }

    test_params_dict : TestParamsDict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'target_accuracy', 'clients_accuracies', 'class_accuracies', 'categorical_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    percentages = np.arange(5, 5, 5)
    test_params_dicts = [test_params_dict.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts[i]['unlearning_percentage'] = percentage

    

    save_path = './stat_tests'

    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=2, devices=[torch.device("cpu"), torch.device("cuda")])
    run_repeated_tests(init_params_dict, test_params_dicts, save_path)
