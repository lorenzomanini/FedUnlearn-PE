import torch
from torch import nn
from backpack import backpack, extend
from backpack.extensions import DiagHessian

import numpy as np
import matplotlib.pyplot as plt

import copy
from tqdm import tqdm


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

def compute_informations(model, criterion, dataloader_list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model).to(device)
    criterion = copy.deepcopy(criterion).to(device)
    model = extend(model)
    criterion = extend(criterion)

    num_clients = len(dataloader_list)
    clients_hessians = []

    num_batches = sum(len(loader) for loader in dataloader_list)
    tqdm_bar = tqdm(total=num_batches, desc="Computing clients information", unit="batch")    

    for loader in dataloader_list:
        client_hessian = {}
        for inputs, targets in loader:
            # Compute the diag Hessian for this batch
            diag_h = compute_diag_hessian(model, criterion, inputs, targets, device=device)

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

def compute_client_information(client_idx, model, criterion, dataloader_list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model).to(device)
    criterion = copy.deepcopy(criterion).to(device)
    model = extend(model)
    criterion = extend(criterion)

    num_clients = len(dataloader_list)
    target_client_hessian = {}
    total_hessian = {}

    num_batches = sum(len(loader) for loader in dataloader_list)
    tqdm_bar = tqdm(total=num_batches, desc="Computing clients information", unit="batch")    

    for loader_idx, loader in enumerate(dataloader_list):
        for inputs, targets in loader:
            # Compute the diag Hessian for this batch
            diag_h = compute_diag_hessian(model, criterion, inputs, targets, device=device)

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

def find_informative_params(information, method='parameters', info_percentage=None, param_percentage=None, graph=False):
    informative_params = {}
    thresholds = {}

    for name, layer_info in information.items():
        if method == 'information':
            raise NotImplementedError("Information method is not implemented yet.")
        elif method == 'parameters':
            sorted_layer_info=np.sort(layer_info.flatten())[::-1]
            threshold_idx = int(len(sorted_layer_info) // 100 * param_percentage)
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
        informative_params[name] = torch.argwhere(layer_info >= thresholds[name])

    return informative_params

def reset_parameters(model, informative_params):
    model_state = model.state_dict()
    resetted_params = {}

    for name in informative_params.keys():
        new_param = model_state[name].clone().detach()
        new_param[tuple(informative_params[name].t())] = 0.0
        resetted_params[name] = new_param
        
    return resetted_params
        


class UnlearnNet(nn.Module):
    """
    A module that wraps an existing model and selectively retrains individual 
    scalar elements (indices) of its parameters while keeping the rest fixed.
    """

    def __init__(self, base_model, indices_to_retrain):
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
        self.inner_model = {"model": base_model}

        # Move any index tensors to CPU and store them.
        self.indices_to_retrain = [idx.cpu() for idx in indices_to_retrain]

        # Create a copy of the base model's parameters as buffers, where
        # we zero out the positions that will be retrained.
        base_params = {}
        for i, (param_name, param) in enumerate(base_model.named_parameters()):
            # Detach a clone of the original parameter
            cloned_param = param.clone().detach()
            # Zero-out the entries we plan to retrain
            if len(self.indices_to_retrain[i]) > 0:
                cloned_param[tuple(self.indices_to_retrain[i].t())] = 0
            base_params[param_name] = cloned_param

        # Register these base parameters as buffers so they are not optimized
        for param_name, buf in base_params.items():
            buf_name = param_name.replace(".", "_")
            self.register_buffer(f"base_{buf_name}", buf)

        # Create the new learnable parameters for only the chosen indices
        retrain_params_dict = {}
        for i, (param_name, param) in enumerate(base_model.named_parameters()):
            if len(self.indices_to_retrain[i]) == 0:
                continue
            # We create a 1D tensor (one entry per retrained element)
            key = param_name.replace(".", "_")
            retrain_params_dict[key] = nn.Parameter(
                torch.zeros(len(self.indices_to_retrain[i]))
            )
        self.retrain_params = nn.ParameterDict(retrain_params_dict)

        # Build sparse masks to apply the learnable values at the correct indices
        sparse_masks = {}
        for i, (param_name, param) in enumerate(base_model.named_parameters()):
            if len(self.indices_to_retrain[i]) == 0:
                continue
            # 'retrain_indices' has shape (k, n_dims). Add a final dim to index positions in the retrain-param vector.
            retrain_indices = indices_to_retrain[i]
            k = retrain_indices.size(0)

            # Create an index column [0..k-1], then concatenate it with 'retrain_indices'.
            row_idx = torch.arange(k).unsqueeze(1)
            final_idx_matrix = torch.cat([retrain_indices, row_idx], dim=1)

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
            detached_params[key] = value.cpu().detach()
        return detached_params
    