from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters, mia_attack
from fisherunlearn import UnlearnNet

import fisherunlearn

import os
import pickle
import random
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

import numpy as np
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


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

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INFO_BATCH_SIZE = 15
    MIA_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 32
    TRAIN_BATCH_SIZE = 32

def compute_accuracy(model, dataset):
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    tqdm_bar = tqdm.tqdm(total=len(dataloader), desc="Computing accuracy", unit="batch", leave=False)
    
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


class Test:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict={}):

        # Initialize parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clients_datasets = clients_subsets
        self.target_client = init_params_dict.get('target_client', 0)

        self.target_dataset = self.clients_datasets[self.target_client]
        self.benchmark_datasets = [subset for i, subset in enumerate(self.clients_datasets) if i != self.target_client]

        self.classes_datasets = split_dataset_by_class_distribution(self.test_dataset, np.identity(init_params_dict['num_classes']))

        self.model_class = model_class
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        self.train_epochs = init_params_dict['train_epochs']
        self.info_use_converter = init_params_dict.get('info_use_converter', True)


        self.clients_indices = [subset.indices for subset in clients_subsets]
        self.trained_model = None
        self.benchmark_model = None
        self.client_information = None


        logging.info("Training model...") 
        self.trained_model = self.trainer_function(self.model_class(), self.loss_class(), self.clients_datasets, self.train_epochs)

        logging.info("Training benchmark model...") 
        self.benchmark_model = self.trainer_function(self.model_class(), self.loss_class(), self.benchmark_datasets, self.train_epochs)

        logging.info("Computing information...") 
        self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_datasets, use_converter=self.info_use_converter)


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

        if 'mia' in test_params_dict['tests']:
            logging.info("Running MIA...")
            classifier_type = test_params_dict['mia_classifier_type']

            try:
                result['trained_mia'] = self.trained_mia
                result['benchmark_mia'] = self.benchmark_mia
            except AttributeError:
                self.trained_mia = mia_attack(self.trained_model, self.target_dataset, self.test_dataset, classifier_type)
                self.benchmark_mia = mia_attack(self.benchmark_model, self.target_dataset, self.test_dataset, classifier_type)
                result['trained_mia'] = self.trained_mia
                result['benchmark_mia'] = self.benchmark_mia

            result['reset_mia'] = mia_attack(reset_model, self.target_dataset, self.test_dataset, classifier_type)
            result['retrained_mia'] = mia_attack(retrained_model, self.target_dataset, self.test_dataset, classifier_type)

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

def get_model_class(init_params_dict):
    model_name = init_params_dict['model_name']
    if model_name == 'simple_cnn':
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
        return FLNet
    elif model_name == 'resnet18':
        num_classes = init_params_dict['num_classes']
        # Ensure ResNet18 input layer matches dataset (e.g., MNIST needs adjustment)
        def create_resnet():
            model = resnet18(num_classes=num_classes)
            if init_params_dict['dataset_name'] == 'mnist':
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return model
        return create_resnet
    else:
        raise ValueError("Unsupported model name")

def get_loss_class(init_params_dict):
    loss_name = init_params_dict['loss_name']
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss
    elif loss_name == 'mse':
        return nn.MSELoss
    else:
        raise ValueError("Unsupported loss name")

def get_trainer_function(init_params_dict):
    trainer_name = init_params_dict['trainer_name']
    if trainer_name == 'sgd':
        def trainer(model, loss_fn, subsets, epochs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.train()

            dataloader = DataLoader(concatenate_subsets(subsets), TRAIN_BATCH_SIZE, shuffle=True)
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            for epoch in tqdm.tqdm(range(epochs), desc="Training", unit="epoch", leave=False):
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets) 
                    loss.backward()
                    optimizer.step()

            model.eval()
            return model.cpu()
        return trainer
    else:
        raise ValueError(f"Unsupported trainer name: {trainer_name}")

def run_repeated_tests(init_params_dict, test_params_dicts, save_path):

    test_name = init_params_dict['test_name']
    logging.info(f"Starting test suite: {test_name}")
    logging.info("Initial Configuration:")
    for key, value in init_params_dict.items():
        logging.info(f"  {key}: {value}")
    logging.info("-" * 30)
    
    seed_value = init_params_dict.get('seed', None)
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
        logging.info(f"Global seed set to {seed_value}")
    else:
        logging.warning("No seed specified in init_params_dict. Results may not be reproducible.")

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

    for i in tqdm.tqdm(range(num_tests), desc="Running repeated tests"):
        logging.info(f"--- Starting Test Iteration {i+1}/{num_tests} ---")

        test_iter_path = os.path.join(test_path, f"test_{i}")
        os.makedirs(test_iter_path)
        clients_indices_path = os.path.join(test_iter_path, "clients_indices.pkl")
        trained_model_path = os.path.join(test_iter_path, "trained_model.pth")
        benchmark_model_path = os.path.join(test_iter_path, "benchmark_model.pth")
        client_information_path = os.path.join(test_iter_path, "client_information.pkl")
        test_results_path = os.path.join(test_iter_path, "test_results.pkl")

        test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict)

        torch.save(test_instance.trained_model.cpu().state_dict(), trained_model_path)
        torch.save(test_instance.benchmark_model.cpu().state_dict(), benchmark_model_path)

        with open(clients_indices_path, 'wb') as f:
            pickle.dump(test_instance.clients_indices, f)
        with open(client_information_path, 'wb') as f:
            pickle.dump(test_instance.client_information, f)

        iteration_results = []
        for test_params_dict in tqdm.tqdm(test_params_dicts, desc=f"Unlearning tests", leave=False):
            test_result = test_instance.run_test(test_params_dict)
            iteration_results.append(test_result)

        with open(test_results_path, 'wb') as f:
            pickle.dump(iteration_results, f)



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = random.randint(0, 10000)

    init_params_dict = {
        'test_name': 'test_mnist_mia_final', # Changed name slightly
        'seed': seed,                       # Seed for reproducibility

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'random',     # Distribution type

        'model_name': 'simple_cnn',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 1,                # Initial training epochs

        'info_use_converter': False,      # Param for Fisher info calc

        'target_client': 0,               # Client to unlearn
        'num_tests': 2                   # Number of independent repetitions
    }

    test_params_dict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'target_accuracy', 'clients_accuracies', 'class_accuracies', 'mia'],
            'mia_classifier_type': 'nn',
            'retrain_epochs': 1
        }
    
    percentages = np.arange(5, 45, 5)
    test_params_dicts = [test_params_dict.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts[i]['unlearning_percentage'] = percentage

    save_path = './stat_tests'

    with logging_redirect_tqdm():
        run_repeated_tests(init_params_dict, test_params_dicts, save_path)
