from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters
from fisherunlearn import UnlearnNet

import os
import pickle
import random
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

import numpy as np
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed: int):
    """Sets the seed for reproducibility in random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU (eventualmente qualora servisse per il server)
    logging.info(f"Global seed set to {seed}")


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
    return correct / total

def compute_dataloaders_accuracy(model, dataloaders, device=None):
    accuracies = []
    if device is None:
        device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_ = device
    model.to(device_)
    model.eval()
    for dataloader in dataloaders:
        accuracy = compute_accuracy(model, dataloader, device=device_)
        accuracies.append(accuracy)
    return accuracies

#Copied here instead of using the library only because I added some logging insider + the possibility to plot_ROC or save
def mia_attack(model, member_loader, nonmember_loader, device, classifier_type='logistic', plot_roc=False, save_plot_path=None):
    """
    Performs Membership Inference Attack.

    Args:
        model: The target PyTorch model.
        member_loader: DataLoader for member data.
        nonmember_loader: DataLoader for non-member data.
        device: The device to run computations on.
        classifier_type: Type of attack classifier ('logistic', 'svm', 'linear', 'nn').
        plot_roc: Whether to generate and save/show the ROC plot.
        save_plot_path: Path to save the ROC plot. If None and plot_roc is True, plt.show() is used.

    Returns:
        Tuple (auc, accuracy) of the attack.
    """
    # This function remains unchanged as per the request
    logging.info(f"Starting MIA attack using {classifier_type} classifier...")
    if classifier_type not in ['logistic', 'svm', 'linear', 'nn']:
        raise ValueError("Invalid classifier_type: choose 'logistic', 'svm', 'linear', or 'nn'.")

    def get_features(target_model, dataloader):
        target_model.eval()
        target_model.to(device)
        losses = []
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = target_model(x)
                loss = ce_loss(logits, y)
                losses.extend(loss.cpu().numpy())
        return np.array(losses).reshape(-1, 1)

    logging.debug("MIA: Extracting features for members...")
    X_member = get_features(model, member_loader)
    logging.debug("MIA: Extracting features for non-members...")
    X_nonmember = get_features(model, nonmember_loader)

    if len(X_member) == 0 or len(X_nonmember) == 0:
         logging.error("MIA failed: Member or non-member feature set is empty.")
         return 0.5, 0.5

    X = np.concatenate([X_member, X_nonmember])
    y = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])

    auc, acc = 0.5, 0.5
    y_true_plot, y_score_plot = y, np.zeros_like(y, dtype=float)

    try:
        if classifier_type in ['logistic', 'svm', 'linear']:
            logging.debug(f"MIA: Training {classifier_type} classifier...")
            if classifier_type == 'logistic':
                clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='liblinear')
            elif classifier_type == 'svm':
                clf = SVC(probability=True, random_state=42, gamma='auto')
            elif classifier_type == 'linear':
                clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, tol=1e-3)

            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_score = clf.predict_proba(X)[:, 1]
            y_true = y
            auc = roc_auc_score(y_true, y_score)
            acc = accuracy_score(y_true, y_pred)
            y_true_plot, y_score_plot = y_true, y_score

        elif classifier_type == 'nn':
            logging.debug("MIA: Training NN classifier...")
            class SimpleNN(nn.Module):
                def __init__(self, input_dim=1):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 16)
                    self.fc2 = nn.Linear(16, 1)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.sigmoid(self.fc2(x))
                    return x

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            if len(X_train) == 0 or len(X_test) == 0:
                 logging.warning("MIA NN: Train or test split is empty. Skipping NN training.")
                 return 0.5, 0.5

            input_dim = X_train.shape[1]
            model_nn = SimpleNN(input_dim).to(device)
            optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.005)
            criterion = nn.BCELoss()

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

            epochs = 100
            batch_size_nn = 32
            train_dataset_nn = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader_nn = torch.utils.data.DataLoader(train_dataset_nn, batch_size=batch_size_nn, shuffle=True)

            model_nn.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader_nn:
                    optimizer.zero_grad()
                    outputs = model_nn(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            model_nn.eval()
            with torch.no_grad():
                y_score = model_nn(X_test_tensor).cpu().numpy().flatten()
                y_pred = (y_score > 0.5).astype(int)

            if len(np.unique(y_test)) > 1:
                 auc = roc_auc_score(y_test, y_score)
            else:
                 auc = 0.5
            acc = accuracy_score(y_test, y_pred)
            y_true_plot, y_score_plot = y_test, y_score

        if plot_roc and len(np.unique(y_true_plot)) > 1:
            fpr, tpr, _ = roc_curve(y_true_plot, y_score_plot)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Membership Inference ROC ({classifier_type})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_plot_path:
                plt.savefig(save_plot_path)
                logging.info(f"MIA ROC curve saved to {save_plot_path}")
                plt.close()
            else:
                plt.close()
        elif plot_roc:
            logging.warning("MIA Plotting skipped: Only one class present in labels.")

    except Exception as e:
        logging.error(f"MIA failed during classifier training/evaluation: {e}", exc_info=True)
        return 0.5, 0.5

    logging.info(f"[MIA Result ({classifier_type})] AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    return auc, acc


class Test:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict={}):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clients_subsets = clients_subsets # List of Subsets for each client
        self.target_client = init_params_dict.get('target_client', 0)

        self.model_class = model_class
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        self.num_classes = init_params_dict['num_classes']
        self.train_epochs = init_params_dict['train_epochs']
        self.eval_batch_size = init_params_dict.get('eval_batch_size', 64)
        self.mia_batch_size = init_params_dict.get('mia_batch_size', self.eval_batch_size) # MIA batch size

        self.info_batch_size = init_params_dict.get('info_batch_size', 15)
        self.info_use_converter = init_params_dict.get('info_use_converter', True)

        self.run_mia = init_params_dict.get('run_mia', True)
        self.mia_classifier_type = init_params_dict.get('mia_classifier_type', 'logistic')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Test instance using device: {self.device}")

        self.benchmark_datasets = [subset for i, subset in enumerate(clients_subsets) if i != self.target_client]
        self.clients_indices = [subset.indices for subset in clients_subsets]

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False)
        self.target_dataloader = DataLoader(self.clients_subsets[self.target_client], batch_size=self.eval_batch_size, shuffle=False)
        self.mia_nonmember_loader = DataLoader(self.test_dataset, batch_size=self.mia_batch_size, shuffle=False)
        self.mia_member_loader = DataLoader(self.clients_subsets[self.target_client], batch_size=self.mia_batch_size, shuffle=True)

        self.clients_dataloaders = [DataLoader(subset, batch_size=self.eval_batch_size, shuffle=False) for subset in self.clients_subsets]
        classes_subsets = split_dataset_by_class_distribution(self.test_dataset, np.identity(self.num_classes))
        self.class_dataloaders = [DataLoader(subset, batch_size=self.eval_batch_size, shuffle=False) for subset in classes_subsets]

        self.trained_model = None
        self.benchmark_model = None
        self.client_information = None


        self.trained_test_accuracy = None
        self.trained_target_accuracy = None
        self.benchmark_test_accuracy = None
        self.trained_clients_accuracies = None 
        self.trained_class_accuracies = None 
        self.benchmark_target_accuracy = None
        self.benchmark_clients_accuracies = None
        self.benchmark_class_accuracies = None

    def _compute_and_cache_initial_accuracies(self):
        """Computes and caches accuracies for trained and benchmark models."""
        logging.info("Computing initial accuracies for trained model...")
        self.trained_model.to(self.device)
        self.trained_test_accuracy = compute_accuracy(self.trained_model, self.test_dataloader, self.device)
        self.trained_target_accuracy = compute_accuracy(self.trained_model, self.target_dataloader, self.device)
        self.trained_clients_accuracies = compute_dataloaders_accuracy(self.trained_model, self.clients_dataloaders, self.device)
        self.trained_class_accuracies = compute_dataloaders_accuracy(self.trained_model, self.class_dataloaders, self.device)
        logging.info(f"Trained Model - Test Acc: {self.trained_test_accuracy:.4f}, Target Acc: {self.trained_target_accuracy:.4f}")

        logging.info("Computing initial accuracies for benchmark model...")
        self.benchmark_model.to(self.device)
        self.benchmark_test_accuracy = compute_accuracy(self.benchmark_model, self.test_dataloader, self.device)
        self.benchmark_target_accuracy = compute_accuracy(self.benchmark_model, self.target_dataloader, self.device)
        self.benchmark_clients_accuracies = compute_dataloaders_accuracy(self.benchmark_model, self.clients_dataloaders, self.device)
        self.benchmark_class_accuracies = compute_dataloaders_accuracy(self.benchmark_model, self.class_dataloaders, self.device)
        logging.info(f"Benchmark Model - Test Acc: {self.benchmark_test_accuracy:.4f}, Target Acc (on unseen data): {self.benchmark_target_accuracy:.4f}")

    def init_new_test(self):
        logging.info("Initializing new test: Training models...") 
        # Train models
        self.trained_model = self.trainer_function(self.model_class(), self.loss_class(), self.clients_subsets, self.train_epochs, self.device)

        self.benchmark_model = self.trainer_function(self.model_class(), self.loss_class(), self.benchmark_datasets, self.train_epochs, self.device)

        # Compute information
        self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_subsets, batch_size=self.info_batch_size, use_converter=self.info_use_converter)

        self._compute_and_cache_initial_accuracies()

    def run_test(self, test_params_dict):
        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        logging.info(f"Running test: Method={unlearning_method}, Percentage={unlearning_percentage}, RetrainEpochs={retrain_epochs}")

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)

        total_individual_reset_params = 0
        for name, indices_tensor in informative_params.items():
            if indices_tensor is not None and indices_tensor.numel() > 0: 
                 total_individual_reset_params += indices_tensor.shape[0]

        logging.info(f"Total number of individual parameters identified for reset: {total_individual_reset_params}")

        reset_model = self.model_class().to(self.device)

        reset_state_dict = reset_parameters(self.trained_model.cpu(), informative_params)
        reset_model.load_state_dict(reset_state_dict)
        reset_model.to(self.device)

        retrainer = UnlearnNet(reset_model, informative_params) 
        self.trainer_function(retrainer, self.loss_class(), self.benchmark_datasets, retrain_epochs, self.device)
        retrained_model = self.model_class().to(self.device)
        retrained_model.load_state_dict(retrainer.get_retrained_params())

        result = {}

        result['total_individual_reset_params'] = total_individual_reset_params

        result['trained_test_accuracy'] = self.trained_test_accuracy
        result['trained_target_accuracy'] = self.trained_target_accuracy
        result['trained_clients_accuracies'] = self.trained_clients_accuracies
        result['trained_class_accuracies'] = self.trained_class_accuracies
        result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
        result['benchmark_target_accuracy'] = self.benchmark_target_accuracy
        result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
        result['benchmark_class_accuracies'] = self.benchmark_class_accuracies

        logging.info("Computing accuracies for reset model...")
        result['reset_test_accuracy'] = compute_accuracy(reset_model, self.test_dataloader, self.device)
        result['reset_target_accuracy'] = compute_accuracy(reset_model, self.target_dataloader, self.device)
        result['reset_clients_accuracies'] = compute_dataloaders_accuracy(reset_model, self.clients_dataloaders, self.device)
        result['reset_class_accuracies'] = compute_dataloaders_accuracy(reset_model, self.class_dataloaders, self.device)

        logging.info("Computing accuracies for retrained model...")
        result['retrained_test_accuracy'] = compute_accuracy(retrained_model, self.test_dataloader, self.device)
        result['retrained_target_accuracy'] = compute_accuracy(retrained_model, self.target_dataloader, self.device)
        result['retrained_clients_accuracies'] = compute_dataloaders_accuracy(retrained_model, self.clients_dataloaders, self.device)
        result['retrained_class_accuracies'] = compute_dataloaders_accuracy(retrained_model, self.class_dataloaders, self.device)

        if self.run_mia:
            logging.info(f"--- Running Membership Inference Attack ({self.mia_classifier_type}) ---")

            # Attack 1: Original Trained Model
            logging.info("MIA on: Original Trained Model")
            try:
                # Ensure trained_model is in eval mode for MIA
                self.trained_model.eval()
                trained_mia_auc, trained_mia_acc = mia_attack(
                    self.trained_model,
                    self.mia_member_loader,    # Target client data = members
                    self.mia_nonmember_loader, # Test data = non-members
                    self.device,
                    classifier_type=self.mia_classifier_type,
                    plot_roc=False # Disable plotting during run
                )
                result['trained_mia_auc'] = trained_mia_auc
                result['trained_mia_acc'] = trained_mia_acc
            except Exception as e:
                 logging.error(f"MIA failed for trained_model: {e}", exc_info=True)
                 result['trained_mia_auc'] = -1.0 # Indicate error
                 result['trained_mia_acc'] = -1.0

            # **** START: ADDED MIA FOR RESET MODEL ****
            # Attack 2: Reset Model (after reset, before retraining)
            logging.info("MIA on: Reset Model")
            try:
                # Ensure reset_model is in eval mode for MIA (it should be already, but explicit is good)
                reset_model.eval()
                reset_mia_auc, reset_mia_acc = mia_attack(
                    reset_model,              # Use the reset model
                    self.mia_member_loader,    # Target client data = members (test if still distinguishable)
                    self.mia_nonmember_loader, # Test data = non-members
                    self.device,
                    classifier_type=self.mia_classifier_type,
                    plot_roc=False # Disable plotting during run
                )
                result['reset_mia_auc'] = reset_mia_auc   # Use new keys
                result['reset_mia_acc'] = reset_mia_acc   # Use new keys
            except Exception as e:
                 logging.error(f"MIA failed for reset_model: {e}", exc_info=True)
                 result['reset_mia_auc'] = -1.0 # Indicate error
                 result['reset_mia_acc'] = -1.0
            # **** END: ADDED MIA FOR RESET MODEL ****


            # Attack 3: Retrained Model (after unlearning and retraining)
            logging.info("MIA on: Retrained Model")
            try:
                # Ensure retrained_model is in eval mode (it should be already)
                retrained_model.eval()
                retrained_mia_auc, retrained_mia_acc = mia_attack(
                    retrained_model,
                    self.mia_member_loader,    # Target client data = members
                    self.mia_nonmember_loader, # Test data = non-members
                    self.device,
                    classifier_type=self.mia_classifier_type,
                    plot_roc=False # Disable plotting during run
                )
                result['retrained_mia_auc'] = retrained_mia_auc
                result['retrained_mia_acc'] = retrained_mia_acc
            except Exception as e:
                 logging.error(f"MIA failed for retrained_model: {e}", exc_info=True)
                 result['retrained_mia_auc'] = -1.0 # Indicate error
                 result['retrained_mia_acc'] = -1.0

        else:
            logging.info("--- Skipping Membership Inference Attack ---")
            result['trained_mia_auc'] = -1.0
            result['trained_mia_acc'] = -1.0
            result['reset_mia_auc'] = -1.0   
            result['reset_mia_acc'] = -1.0    
            result['retrained_mia_auc'] = -1.0
            result['retrained_mia_acc'] = -1.0

        # Verify that MIA keys exist in the results dictionary before returning
        mia_keys_to_check = [
            'trained_mia_auc', 'trained_mia_acc',
            'reset_mia_auc', 'reset_mia_acc',    
            'retrained_mia_auc', 'retrained_mia_acc'
        ]
        missing_keys = [key for key in mia_keys_to_check if key not in result]
        if missing_keys:
            logging.error(f"CRITICAL CHECK FAILED: MIA keys {missing_keys} are missing from the result dictionary before returning!")
        else:
            logging.debug("MIA keys check passed. All expected MIA keys are present in the result dictionary.")

        logging.info(f"Finished test run. Retrained Test Acc: {result['retrained_test_accuracy']:.4f}, "
                     f"Trained MIA AUC: {result.get('trained_mia_auc', 'N/A'):.4f}, "
                     f"Reset MIA AUC: {result.get('reset_mia_auc', 'N/A'):.4f}, "
                     f"Retrained MIA AUC: {result.get('retrained_mia_auc', 'N/A'):.4f}")

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
        # Note: split_dataset_by_class_distribution with uniform might not be truly IID
        # if dataset class distribution isn't perfectly balanced.
        # random_split might be better for pure IID.
        class_distribution = np.ones((num_clients, num_classes)) / num_classes
        return split_dataset_by_class_distribution(dataset, class_distribution)

    elif distribution_type == 'dirichlet':
        alpha = init_params_dict.get('dirichlet_alpha', 1.0) # Default alpha=1.0
        class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
        return split_dataset_by_class_distribution(dataset, class_distribution)

    elif distribution_type == 'random':
        total_len = len(dataset)
        lengths = [1.0 / num_clients] * num_clients
        # Ensure lengths sum to total_len for random_split
        lengths_int = [int(l * total_len) for l in lengths]
        remainder = total_len - sum(lengths_int)
        for i in range(remainder): lengths_int[i] += 1
        return torch.utils.data.random_split(dataset, lengths_int)

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
            if init_params_dict.get('dataset_name') == 'mnist':
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
    lr = init_params_dict.get('lr', 0.01)
    momentum = init_params_dict.get('momentum', 0.9)
    train_batch_size = init_params_dict.get('train_batch_size', 32)
    logging.info(f"Using trainer: {trainer_name} with lr={lr}, momentum={momentum}, batch_size={train_batch_size}")

    if trainer_name == 'sgd':

        def trainer(model, loss_fn_instance, subsets, epochs, device):
            model.to(device)
            model.train()

            loss_fn = loss_fn_instance
            dataloader = DataLoader(concatenate_subsets(subsets), batch_size=train_batch_size, shuffle=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            for epoch in tqdm.tqdm(range(epochs), desc=f"Training on {device}", unit="epoch", leave=False):
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets) 
                    loss.backward()
                    optimizer.step()

            model.eval()
            return model
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
        set_seed(seed_value)
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
        logging.warning(f"Test directory '{orig_path}' already exists. Saving to '{test_path}'.")
    os.makedirs(test_path)
    logging.info(f"Created test suite directory: {test_path}")
    init_params_dict_path = os.path.join(test_path, "init_params.pkl")
    test_params_dicts_path = os.path.join(test_path, "test_params.pkl")
    try:
        with open(init_params_dict_path, 'wb') as f: pickle.dump(init_params_dict, f)
        with open(test_params_dicts_path, 'wb') as f: pickle.dump(test_params_dicts, f)
        logging.info("Saved configuration dictionaries.")
    except Exception as e: logging.error(f"Error saving configuration files: {e}")

    num_tests = init_params_dict['num_tests']
    try:
        train_dataset, test_dataset = get_datasets(init_params_dict)
        clients_subsets = get_clients_subsets(train_dataset, init_params_dict)
        model_class = get_model_class(init_params_dict)
        loss_class = get_loss_class(init_params_dict) 
        trainer_function = get_trainer_function(init_params_dict)
    except ValueError as e:
        logging.error(f"Error during setup: {e}")
        return

    test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict)

    all_iterations_results = []
    for i in tqdm.tqdm(range(num_tests), desc="Running repeated tests"):
        logging.info(f"--- Starting Test Iteration {i+1}/{num_tests} ---")
        iter_seed = seed_value + i if seed_value is not None else None
        if iter_seed is not None:
            logging.info(f"Setting seed for iteration {i+1} to {iter_seed}")
            set_seed(iter_seed)
            # Optional: Regenerate client subsets if they depend on iteration seed
            # clients_subsets = get_clients_subsets(train_dataset, init_params_dict)
            # test_instance.clients_subsets = clients_subsets
            # test_instance.benchmark_datasets = [subset for idx, subset in enumerate(clients_subsets) if idx != test_instance.target_client]
            # # Need to update dataloaders if subsets change
            # test_instance.target_dataloader = DataLoader(...)
            # test_instance.mia_member_loader = DataLoader(...)
            # test_instance.clients_dataloaders = [DataLoader(...) for ...]

        test_iter_path = os.path.join(test_path, f"test_{i}")
        try: os.makedirs(test_iter_path, exist_ok=True)
        except OSError as e: logging.error(f"Could not create directory {test_iter_path}: {e}"); continue

        clients_indices_path = os.path.join(test_iter_path, "clients_indices.pkl")
        trained_model_path = os.path.join(test_iter_path, "trained_model.pth")
        benchmark_model_path = os.path.join(test_iter_path, "benchmark_model.pth")
        client_information_path = os.path.join(test_iter_path, "client_information.pkl")
        test_results_path = os.path.join(test_iter_path, "test_results.pkl")

        try:
            test_instance.init_new_test()

            torch.save(test_instance.trained_model.cpu().state_dict(), trained_model_path)
            torch.save(test_instance.benchmark_model.cpu().state_dict(), benchmark_model_path)
            with open(clients_indices_path, 'wb') as f: pickle.dump(test_instance.clients_indices, f)
            try:
                client_info_to_save = test_instance.client_information
                if isinstance(client_info_to_save, dict):
                     client_info_to_save = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in client_info_to_save.items()}
                elif isinstance(client_info_to_save, list): 
                     client_info_to_save = [v.cpu() if isinstance(v, torch.Tensor) else v for v in client_info_to_save]


                with open(client_information_path, 'wb') as f: pickle.dump(client_info_to_save, f)
            except Exception as pickle_err:
                 logging.warning(f"Could not pickle client_information: {pickle_err}. Saving as None.")
                 with open(client_information_path, 'wb') as f: pickle.dump(None, f)

            logging.info(f"Saved artifacts for iteration {i+1} to {test_iter_path}")

            iteration_results = []
            for test_params_dict in tqdm.tqdm(test_params_dicts, desc=f"Iter {i+1} - Unlearning tests", leave=False):
                test_result = test_instance.run_test(test_params_dict)
                iteration_results.append(test_result)

            with open(test_results_path, 'wb') as f:
                pickle.dump(iteration_results, f)
            all_iterations_results.append(iteration_results)
            logging.info(f"Saved results for iteration {i+1}.")

        except Exception as e:
             logging.error(f"Error during test iteration {i+1}: {e}", exc_info=True)

        logging.info(f"--- Finished Test Iteration {i+1}/{num_tests} ---")

    logging.info(f"--- Test Suite '{test_name}' Completed ---")


if __name__ == "__main__":
    init_params_dict = {
        'test_name': 'test_mnist_mia_final', # Changed name slightly
        'seed': 42,                       # Seed for reproducibility
        'run_mia': True,                  # Enable/disable MIA runs
        'mia_classifier_type': 'nn',      # Attack classifier ('logistic', 'svm', 'nn')
        'mia_batch_size': 128,            # Batch size for MIA feature extraction

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'random',     # Distribution type
        #'dirichlet_alpha': 0.5,           # Alpha for Dirichlet if used

        'model_name': 'simple_cnn',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'lr': 0.01,                       # Learning rate
        'momentum': 0.9,                  # Momentum
        'train_batch_size': 64,           # Training batch size
        'eval_batch_size': 128,           # Evaluation batch size
        'train_epochs': 5,                # Initial training epochs

        'info_batch_size': 10,            # Batch size for Fisher info calc
        'info_use_converter': False,      # Param for Fisher info calc

        'target_client': 0,               # Client to unlearn
        'num_tests': 2                   # Number of independent repetitions
    }

    test_params_dict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'retrain_epochs': 1
        }
    
    percentages = np.arange(5, 45, 5)
    test_params_dicts = [test_params_dict.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts[i]['unlearning_percentage'] = percentage


    logging.info(f"Generated {len(test_params_dicts)} unlearning test configurations.")

    save_path = './stat_tests_mia_final'

    run_repeated_tests(init_params_dict, test_params_dicts, save_path)
