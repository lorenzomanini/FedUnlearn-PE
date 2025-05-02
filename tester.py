from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters
from fisherunlearn import UnlearnNet

import os
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import numpy as np
import tqdm

def compute_accuracy(model, dataloader, device=None):
    if device is None:
        device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_ = device

    model.to(device_)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device_), labels.to(device_)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if device is None:
        model.cpu()

    return correct / total

def compute_dataloaders_accuracy(model, dataloaders):
    accuracies = []
    for dataloader in dataloaders:
        accuracy = compute_accuracy(model, dataloader)
        accuracies.append(accuracy)
    return accuracies

class Test:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict={}):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clients_subsets = clients_subsets
        self.target_client = init_params_dict.get('target_client', 0)
        
        self.model_class = model_class
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        self.num_classes = init_params_dict['num_classes']
        self.train_epochs = init_params_dict['train_epochs']

        self.info_batch_size = init_params_dict.get('info_batch_size', 15)
        self.info_use_converter = init_params_dict.get('info_use_converter', True)

        self.benchmark_datasets = list(self.clients_subsets)
        self.benchmark_datasets.pop(self.target_client)
        self.clients_indices = [subset.indices for subset in clients_subsets]

        # Attributes initialized in init_new_test()
        self.trained_model = None
        self.benchmark_model = None
        self.client_information = None

    def init_new_test(self):      
        # Train models
        self.trained_model = self.trainer_function(self.model_class(), self.loss_class(), self.clients_subsets, self.train_epochs)

        self.benchmark_model = self.trainer_function(self.model_class(), self.loss_class(), self.benchmark_datasets, self.train_epochs)

        # Compute information
        self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_subsets, batch_size=self.info_batch_size, use_converter=self.info_use_converter)

    def run_test(self, test_params_dict):
        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)

        reset_model = self.model_class()
        reset_model.load_state_dict(reset_parameters(self.trained_model, informative_params))

        retrainer = UnlearnNet(reset_model, informative_params)
        self.trainer_function(retrainer, self.loss_class(), self.benchmark_datasets, retrain_epochs)
        retrained_model = self.model_class()
        retrained_model.load_state_dict(retrainer.get_retrained_params())

        result = {}
        
        # Create dataloaders
        test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        target_dataloader = DataLoader(self.clients_subsets[self.target_client], batch_size=32, shuffle=False)
        clients_dataloaders = [DataLoader(subset, batch_size=32, shuffle=False) for subset in self.clients_subsets]

        classes_subsets = split_dataset_by_class_distribution(self.test_dataset, np.identity(self.num_classes))
        class_dataloaders = [DataLoader(subset, batch_size=32, shuffle=False) for subset in classes_subsets]

        # Compute accuracies
        try:
            result['trained_test_accuracy'] = self.trained_test_accuracy
            result['trained_target_accuracy'] = self.trained_target_accuracy
            result['trained_clients_accuracies'] = self.trained_clients_accuracies
            result['trained_class_accuracies'] = self.trained_class_accuracies
        except:
            result['trained_test_accuracy'] = self.trained_test_accuracy = compute_accuracy(self.trained_model, test_dataloader)
            result['trained_target_accuracy'] = self.trained_target_accuracy = compute_accuracy(self.trained_model, target_dataloader)
            result['trained_clients_accuracies'] = self.trained_clients_accuracies = compute_dataloaders_accuracy(self.trained_model, clients_dataloaders)
            result['trained_class_accuracies'] = self.trained_class_accuracies = compute_dataloaders_accuracy(self.trained_model, class_dataloaders)

        try:
            result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
            result['benchmark_target_accuracy'] = self.benchmark_target_accuracy
            result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
            result['benchmark_class_accuracies'] = self.benchmark_class_accuracies
        except:
            result['benchmark_test_accuracy'] = self.benchmark_test_accuracy = compute_accuracy(self.benchmark_model, test_dataloader)
            result['benchmark_target_accuracy'] = self.benchmark_target_accuracy = compute_accuracy(self.benchmark_model, target_dataloader)
            result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies = compute_dataloaders_accuracy(self.benchmark_model, clients_dataloaders)
            result['benchmark_class_accuracies'] = self.benchmark_class_accuracies = compute_dataloaders_accuracy(self.benchmark_model, class_dataloaders)  
        
        result['reset_test_accuracy'] =  compute_accuracy(reset_model, test_dataloader)
        result['reset_target_accuracy'] = compute_accuracy(reset_model, target_dataloader)
        result['reset_clients_accuracies'] = compute_dataloaders_accuracy(reset_model, clients_dataloaders)
        result['reset_class_accuracies'] = compute_dataloaders_accuracy(reset_model, class_dataloaders)

        result['retrained_test_accuracy'] = compute_accuracy(retrained_model, test_dataloader)
        result['retrained_target_accuracy'] = compute_accuracy(retrained_model, target_dataloader)
        result['retrained_clients_accuracies'] = compute_dataloaders_accuracy(retrained_model, clients_dataloaders)
        result['retrained_class_accuracies'] = compute_dataloaders_accuracy(retrained_model, class_dataloaders)

        return result


def get_datasets(init_params_dict):
    dataset_name = init_params_dict['dataset_name']
    if dataset_name == 'mnist':
        from torchvision.datasets import MNIST
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=data_transforms["train"])
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=data_transforms["test"])

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
        class_distribution = np.random.dirichlet(np.ones(num_classes), num_clients)
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
        return lambda: resnet18(num_classes=num_classes)
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

            dataloader = DataLoader(concatenate_subsets(subsets), batch_size=32, shuffle=True)
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            for epoch in tqdm.tqdm(range(epochs), desc="Training", unit="epoch"):
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
        raise ValueError("Unsupported trainer name")


def run_repeated_tests(init_params_dict, test_params_dicts, num_tests, save_path):
    
    test_name = init_params_dict['test_name']
    print(f"Running test: {test_name}")
    for key, value in init_params_dict.items():
        print(f"{key}: {value}")
    print()

    # Create a directory for the test
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_path = os.path.join(save_path, test_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    else:
        orig_path = test_path
        i = 1
        test_path = f"{orig_path} ({i})"
        while os.path.exists(test_path):
            i += 1
            test_path = f"{orig_path} ({i})"
        os.makedirs(test_path)
    
    init_params_dict_path = os.path.join(test_path, "init_params.pkl")
    test_params_dicts_path = os.path.join(test_path, "test_params.pkl")
    with open(init_params_dict_path, 'wb') as f:
        pickle.dump(init_params_dict, f)
    with open(test_params_dicts_path, 'wb') as f:
        pickle.dump(test_params_dicts, f)

    # Parse init_params_dict
    num_tests = init_params_dict['num_tests']

    train_dataset, test_dataset = get_datasets(init_params_dict)
    clients_subsets = get_clients_subsets(train_dataset, init_params_dict)

    model_class = get_model_class(init_params_dict)
    loss_class = get_loss_class(init_params_dict)
    trainer_function = get_trainer_function(init_params_dict)

    # Create a test instance
    test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict)

    for i in tqdm.tqdm(range(num_tests), desc="Running repeated tests"):
        test_iter_path = os.path.join(test_path, f"test_{i}")
        os.makedirs(test_iter_path)
        clients_indices_path = os.path.join(test_iter_path, "clients_indices.pkl")
        trained_model_path = os.path.join(test_iter_path, "trained_model.pth")
        benchmark_model_path = os.path.join(test_iter_path, "benchmark_model.pth")
        client_information_path = os.path.join(test_iter_path, "client_information.pkl")
        test_results_path = os.path.join(test_iter_path, "test_results.pkl")

        test_instance.init_new_test()

        torch.save(test_instance.trained_model.state_dict(), trained_model_path)
        torch.save(test_instance.benchmark_model.state_dict(), benchmark_model_path)
        with open(clients_indices_path, 'wb') as f:
            pickle.dump(test_instance.clients_indices, f)
        with open(client_information_path, 'wb') as f:
            pickle.dump(test_instance.client_information, f)
        
        test_results = []
        for test_params_dict in tqdm.tqdm(test_params_dicts, desc="Running tests"):
            test_result = test_instance.run_test(test_params_dict)
            test_results.append(test_result)

        with open(test_results_path, 'wb') as f:
            pickle.dump(test_results, f)

    

if __name__ == "__main__":
    # Example usage
    init_params_dict = {
        'test_name': 'test_mnist',
        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'dirichlet',
        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',
        'trainer_name': 'sgd',
        'train_epochs': 5,
        'info_batch_size': 10,
        'info_use_converter': False,
        'target_client': 0,
        'num_tests': 10
    }

    test_params_dict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'retrain_epochs': 2
        }
    
    percentages = np.arange(0, 80, 10)
    test_params_dicts = [test_params_dict.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts[i]['unlearning_percentage'] = percentage

    print(test_params_dicts)

    save_path = './stat_tests'
    run_repeated_tests(init_params_dict, test_params_dicts, init_params_dict['num_tests'], save_path)







            
        





        
        








