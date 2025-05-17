import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import DiagHessian, DiagGGNExact

import numpy as np
import matplotlib.pyplot as plt

import copy

from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

DEVICE = 'cpu'
INFO_BATCH_SIZE = 1
MIA_BATCH_SIZE = 1

def set_device(device):
    global DEVICE
    DEVICE = device

def set_info_batch_size(batch_size):
    global INFO_BATCH_SIZE
    INFO_BATCH_SIZE = batch_size

def set_mia_batch_size(batch_size):
    global MIA_BATCH_SIZE
    MIA_BATCH_SIZE = batch_size


def compute_diag_hessian(model, criterion, inputs, targets, device='cpu'):
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    with backpack(DiagHessian()):
        loss.backward()

    diag_hessian_params = {}
    for name, param in model.named_parameters():
        if hasattr(param, 'diag_h') and param.requires_grad:
            diag_hessian_params[name] = param.diag_h.clone().detach()
            # Cleanup to avoid leftover references
            del param.diag_h

    return diag_hessian_params

def compute_diag_ggn(model, criterion, inputs, targets, device='cpu'):
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    with backpack(DiagGGNExact()):
        loss.backward()

    diag_hessian_params = {}
    for name, param in model.named_parameters():
        if hasattr(param, 'diag_ggn_exact') and param.requires_grad:
            diag_hessian_params[name] = param.diag_ggn_exact.clone().detach()
            # Cleanup to avoid leftover references
            del param.diag_ggn_exact

    return diag_hessian_params

def compute_informations(model, criterion, dataloader_list, method='diag_ggn', use_converter=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model).to(device).eval()
    criterion = copy.deepcopy(criterion).to(device)
    
    model = extend(model, use_converter=use_converter)
    criterion = extend(criterion)

    num_clients = len(dataloader_list)
    clients_hessians = []

    num_batches = sum(len(loader) for loader in dataloader_list)
    tqdm_bar = tqdm(total=num_batches, desc="Computing clients information", unit="batch")    

    for loader in dataloader_list:
        client_hessian = {}
        for inputs, targets in loader:
            # Compute the diag Hessian for this batch
            if method == 'diag_hessian':
                diag_h = compute_diag_hessian(model, criterion, inputs, targets, device=device)
            elif method == 'diag_ggn':
                diag_h = compute_diag_ggn(model, criterion, inputs, targets, device=device)
            else:
                raise ValueError("Invalid method. Use 'diag_hessian' or 'diag_ggn'.")

            # Accumulate avarage over batches
            for name, value in diag_h.items():
                if name not in client_hessian:
                    client_hessian[name] = value/(len(loader)*num_clients)
                else:
                    client_hessian[name] += value/(len(loader)*num_clients)
                    
            tqdm_bar.update(1)

        clients_hessians.append(client_hessian)
    
    tqdm_bar.close()

    total_hessian = {}
    for name in clients_hessians[0].keys():
        total_hessian[name] = sum(client_hessian[name] for client_hessian in clients_hessians)

    
    clients_informations = []
    for client_idx in range(num_clients):
        client_info = {}
        for name in clients_hessians[client_idx].keys():
            layer_info = 0.5 * torch.pow(clients_hessians[client_idx][name]/total_hessian[name], 2)
            layer_info[total_hessian[name] == 0] = 0
            client_info[name] = layer_info.detach().cpu()
        clients_informations.append(client_info)
    
    return clients_informations


def compute_client_information(client_idx, model, criterion, datasets_list, method='diag_ggn', use_converter=True):

    global DEVICE
    global INFO_BATCH_SIZE

    model = copy.deepcopy(model).to(DEVICE).eval()
    criterion = copy.deepcopy(criterion).to(DEVICE)
    
    model = extend(model, use_converter=use_converter)
    criterion = extend(criterion)

    num_clients = len(datasets_list)
    target_client_hessian = {}
    total_hessian = {}

    dataloader_list = [DataLoader(dataset, INFO_BATCH_SIZE, shuffle=False) for dataset in datasets_list]

    num_batches = sum(len(loader) for loader in dataloader_list)
    tqdm_bar = tqdm(total=num_batches, desc="Computing clients information", unit="batch", leave=False)    

    for loader_idx, loader in enumerate(dataloader_list):
        for inputs, targets in loader:
            # Compute the diag Hessian for this batch
            if method == 'diag_hessian':
                diag_h = compute_diag_hessian(model, criterion, inputs, targets, device=DEVICE)
            elif method == 'diag_ggn':
                diag_h = compute_diag_ggn(model, criterion, inputs, targets, device=DEVICE)
            else:
                raise ValueError("Invalid method. Use 'diag_hessian' or 'diag_ggn'.")

            for name, value in diag_h.items():
                if name not in total_hessian:
                    total_hessian[name] = value/(len(loader)*num_clients)
                else:
                    total_hessian[name] += value/(len(loader)*num_clients)

            if loader_idx == client_idx:
                for name, value in diag_h.items():
                    if name not in target_client_hessian:
                        target_client_hessian[name] = value/(len(loader)*num_clients)
                    else:
                        target_client_hessian[name] += value/(len(loader)*num_clients)
            
            tqdm_bar.update(1)
    
    tqdm_bar.close()

    
    target_client_info = {}
    for name in target_client_hessian.keys():
        layer_info = 0.5 * torch.pow(target_client_hessian[name]/total_hessian[name], 2)
        layer_info[total_hessian[name] == 0] = 0
        target_client_info[name] = layer_info.detach().cpu()
    
    return target_client_info

def plot_information_parameters_tradeoff(information, method, whitelist=None, blacklist=None):
    percentages = np.arange(0,100,0.1)
    information_values = np.zeros(len(percentages))
    params_values = np.zeros(len(percentages))

    total_information = 0
    total_params = 0

    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue

        sorted_layer_info=np.sort(layer_info.flatten())[::-1]
        cumulative_sum = np.cumsum(sorted_layer_info)
        total_information += cumulative_sum[-1]
        total_params += len(sorted_layer_info)

        for i, percentage in enumerate(percentages):
            if method == 'parameters':
                threshold_idx = int(len(sorted_layer_info) / 100 * percentage)
                information_values[i] += cumulative_sum[threshold_idx]
                params_values[i] += threshold_idx
            elif method == 'information':
                threshold_idx = np.argmin(np.abs(cumulative_sum - cumulative_sum[-1] * percentage / 100))
                information_values[i] += cumulative_sum[threshold_idx]
                params_values[i] += threshold_idx
            else:
                raise ValueError("Invalid method. Use 'information' or 'parameters'.")
    
    for i in range(len(information_values)):
        information_values[i] = information_values[i] / total_information * 100
        params_values[i] = 100 - params_values[i] / total_params * 100
    
    plt.plot(percentages, information_values, label='Information erased')
    plt.plot(percentages, params_values, label='Remaining parameters')
    plt.xlabel(f'Layer {method} percentage resetted')
    plt.ylabel('Total percentage')
    plt.title('Information vs Parameters tradeoff')
    plt.legend()
    plt.grid()
    plt.show()

def find_informative_params(information, method, percentage, whitelist=None, blacklist=None, graph=False, tuple_out=False):
    informative_params = {}
    thresholds = {}

    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        if method == 'information':
            sorted_layer_info=np.sort(layer_info.flatten())[::-1]
            cumulative_sum = np.cumsum(sorted_layer_info)
            threshold_idx = np.argmin(np.abs(cumulative_sum - cumulative_sum[-1] * percentage / 100))
            thresholds[name] = sorted_layer_info[threshold_idx]
        elif method == 'parameters':
            sorted_layer_info=np.sort(layer_info.flatten())[::-1]
            threshold_idx = int(len(sorted_layer_info) / 100 * percentage)
            thresholds[name] = sorted_layer_info[threshold_idx]
        else:
            raise ValueError("Invalid method. Use 'information' or 'parameters'.")
    
        if graph:
            plt.figure(figsize=(10, 5))
            plt.title(name)
            plt.plot(sorted_layer_info)
            plt.axvline(threshold_idx, color='r', linestyle='--')
            plt.xlabel('Parameters')
            plt.ylabel('Information')
            plt.show()

    for name, layer_info in information.items():
        if whitelist is not None and name not in whitelist:
            continue
        if blacklist is not None and name in blacklist:
            continue
        if tuple_out:
            informative_params[name] = tuple(torch.argwhere(layer_info > thresholds[name]).t())
        else:
            informative_params[name] = torch.argwhere(layer_info > thresholds[name])

    return informative_params

def reset_parameters(model, informative_params):
    model_state = model.state_dict()
    resetted_params = {}

    for name in model_state.keys():
        if name in informative_params.keys():
            indices = tuple(informative_params[name].t())
            new_param = model_state[name].clone().detach()
            new_param[indices] = 0.0
            resetted_params[name] = new_param
        else:
            resetted_params[name] = model_state[name].clone().detach()
        
    return resetted_params
        


class UnlearnNet(nn.Module):
    """
    A module that wraps an existing model and selectively retrains individual 
    scalar elements (indices) of its parameters while keeping the rest fixed.
    """

    def __init__(self, base_model, informative_params):
        """
        Args:
            base_model (nn.Module): The original model whose parameters 
                                    we want to partially retrain.
            indices_to_retrain (List[torch.Tensor]): For each parameter of 
                                    'base_model', a tensor of indices indicating 
                                    which scalar values should be retrained.
        """
        super().__init__()

        # We store the base model inside a dictionary to allow
        # functional calls later without overshadowing state_dict keys.
        self.inner_model = {"model": copy.deepcopy(base_model)}

        # Create a copy of the base model's parameters as buffers, where
        # we zero out the positions that will be retrained.
        base_params = reset_parameters(base_model, informative_params)

        # Register these base parameters as buffers so they are not optimized
        for param_name, buf in base_params.items():
            buf_name = param_name.replace(".", "_")
            self.register_buffer(f"base_{buf_name}", buf)

        # Create the new learnable parameters for only the chosen indices
        retrain_params_dict = {}
        for param_name, param in base_model.named_parameters():
            if param_name not in informative_params.keys():
                continue
            if len(informative_params[param_name]) == 0:
                continue
            # We create a 1D tensor (one entry per retrained element)
            key = param_name.replace(".", "_")
            retrain_params_dict[key] = nn.Parameter(
                torch.zeros(len(informative_params[param_name]))
            )
        self.retrain_params = nn.ParameterDict(retrain_params_dict)

        # Build sparse masks to apply the learnable values at the correct indices
        sparse_masks = {}
        for param_name, param in base_model.named_parameters():
            if param_name not in informative_params.keys():
                continue
            if len(informative_params[param_name]) == 0:
                continue
            # 'retrain_indices' has shape (k, n_dims). Add a final dim to index positions in the retrain-param vector.
            k = len(informative_params[param_name])

            # Create an index column [0..k-1], then concatenate it with 'retrain_indices'.
            row_idx = torch.arange(k).unsqueeze(1)
            final_idx_matrix = torch.cat([informative_params[param_name], row_idx], dim=1)

            # A sparse_coo_tensor expects indices with shape (ndim, nnz). Transpose to (n_dims+1, k).
            indices_for_sparse = final_idx_matrix.t().contiguous()

            # Append k as the final dimension so each retrained element indexes differently.
            mask_shape = tuple(param.size()) + (k,)

            # Build the sparse mask with 1.0 at the retrained indices.
            key = f"mask_{param_name.replace('.', '_')}"
            sparse_masks[key] = torch.sparse_coo_tensor(
                indices_for_sparse,
                torch.ones(k, dtype=torch.float32),
                size=mask_shape
            )
        
        # Register these sparse masks as buffers
        for mask_name, mask in sparse_masks.items():
            self.register_buffer(mask_name, mask.coalesce())

    def contract_last_dim_with_vector(self, sp_tensor: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Contract the last dimension of a sparse tensor (shape [..., N]) with
        a dense vector of shape (N,), returning a sparse tensor of shape [...].

        This effectively applies elementwise multiplication with 'vec'
        across the last dimension of 'sp_tensor'.
        """

        # Extract indices (shape [ndim, nnz]) and values (shape [nnz])
        indices = sp_tensor.indices()
        values = sp_tensor.values()

        # Multiply each sparse value by the corresponding element in 'vec'
        # indices[-1] indicates which element in 'vec' to use per sparse entry
        new_values = values * vec[indices[-1]]

        # Create a new sparse_coo_tensor with one fewer dimension
        new_shape = sp_tensor.shape[:-1]
        new_indices = indices[:-1, :]  # drop the last dimension index row

        result_tensor = torch.sparse_coo_tensor(
            new_indices,
            new_values,
            size=new_shape,
            dtype=sp_tensor.dtype,
            device=sp_tensor.device
        )

        return result_tensor

    def forward(self, x):
        """
        Forward pass using a functional call to the base model. We reconstruct 
        final parameters by adding the base buffers and the contracted retrain 
        parameters at the relevant indices.
        """
        model = self.inner_model["model"]
        current_state = self.state_dict()

        # Rebuild parameter dict from buffers (base params)
        final_params = {}
        for param_name in model.state_dict().keys():
            buf_name = param_name.replace(".", "_")
            final_params[param_name] = current_state[f"base_{buf_name}"]

        # Add in the learnable values at specified indices
        for key, param_vector in self.retrain_params.items():
            mask_key = f"mask_{key}"
            base_key = f"base_{key}"
            original_param_name = key.replace("_", ".")

            # Convert sparse mask to shape that can be added to base param
            sparse_update = self.contract_last_dim_with_vector(
                current_state[mask_key], param_vector
            )

            # Add the sparse update to the base buffer
            final_params[original_param_name] = (
                current_state[base_key] + sparse_update
            )

        # Perform a functional forward pass with the reconstructed parameters
        return torch.func.functional_call(model, final_params, x)
    
    def get_retrained_params(self):
        """
        Returns the retrained parameters of the model.
        """
        model = self.inner_model["model"]
        current_state = self.state_dict()

        # Rebuild parameter dict from buffers (base params)
        final_params = {}
        for param_name in model.state_dict().keys():
            buf_name = param_name.replace(".", "_")
            final_params[param_name] = current_state[f"base_{buf_name}"]

        # Add in the learnable values at specified indices
        for key, param_vector in self.retrain_params.items():
            mask_key = f"mask_{key}"
            base_key = f"base_{key}"
            original_param_name = key.replace("_", ".")

            # Convert sparse mask to shape that can be added to base param
            sparse_update = self.contract_last_dim_with_vector(
                current_state[mask_key], param_vector
            )

            # Add the sparse update to the base buffer
            final_params[original_param_name] = (
                current_state[base_key] + sparse_update
            )
        
        detached_params = {}
        for key, value in final_params.items():
            detached_params[key] = value.cpu().clone().detach()
        return detached_params


def mia_attack(model, member_dataset, nonmember_dataset, classifier_type='logistic', plot=False):

    global DEVICE
    global MIA_BATCH_SIZE

    member_loader = DataLoader(member_dataset, MIA_BATCH_SIZE, shuffle=False)
    nonmember_loader = DataLoader(nonmember_dataset, MIA_BATCH_SIZE, shuffle=False)

    model.eval()
    
    if classifier_type not in ['logistic', 'svm', 'linear', 'nn']:
        raise ValueError("Invalid classifier_type: choose 'logistic', 'svm', 'linear', or 'nn'.")
    
    def get_features(model, dataloader):
        model.eval()
        model.to(DEVICE)
        confs, losses, entropies = [], [], []
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                conf, _ = probs.max(dim=1)
                loss = ce_loss(logits, y)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

                confs.extend(conf.cpu().numpy())
                losses.extend(loss.cpu().numpy())
                entropies.extend(entropy.cpu().numpy())
        return np.stack([losses], axis=1)
        #return np.stack([confs, losses, entropies], axis=1)

    X_member = get_features(model, member_loader)
    X_nonmember = get_features(model, nonmember_loader)
    X = np.concatenate([X_member, X_nonmember])
    y = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])

    if classifier_type in ['logistic', 'svm', 'linear']:
        if classifier_type == 'logistic':
            clf = LogisticRegression(max_iter=1000)
        elif classifier_type == 'svm':
            clf = SVC(probability=True)
        elif classifier_type == 'linear':
            clf = SGDClassifier(loss='log_loss', max_iter=1000)

        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_score = clf.predict_proba(X)[:, 1]
        y_true = y 
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)

    elif classifier_type == 'nn':
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 16)
                self.fc2 = nn.Linear(16, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.sigmoid(self.fc2(x))
                return x

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_nn = SimpleNN().to(DEVICE)
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

        model_nn.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model_nn(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate
        model_nn.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_score = model_nn(X_test_tensor).cpu().numpy().flatten()
            y_pred = (y_score > 0.5).astype(int)

        auc = roc_auc_score(y_test, y_score)
        acc = accuracy_score(y_test, y_pred)

    if plot:
        fpr, tpr, _ = roc_curve(y if classifier_type != 'nn' else y_test, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Membership Inference ROC Curve ({classifier_type})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"[MIA: {classifier_type}] AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    return {'roc_auc': auc, 'accuracy': acc}
