import copy

import torch
from torch import nn


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
    """Retrain selected scalar parameters while keeping all others fixed."""

    def __init__(self, base_model, informative_params):
        super().__init__()
        self.inner_model = {"model": copy.deepcopy(base_model)}
        base_params = reset_parameters(base_model, informative_params)
        for param_name, buf in base_params.items():
            buf_name = param_name.replace(".", "_")
            self.register_buffer(f"base_{buf_name}", buf)

        retrain_params_dict = {}
        for param_name, param in base_model.named_parameters():
            if param_name not in informative_params.keys():
                continue
            if len(informative_params[param_name]) == 0:
                continue
            key = param_name.replace(".", "_")
            retrain_params_dict[key] = nn.Parameter(
                torch.zeros(len(informative_params[param_name]))
            )
        self.retrain_params = nn.ParameterDict(retrain_params_dict)

        sparse_masks = {}
        for param_name, param in base_model.named_parameters():
            if param_name not in informative_params.keys():
                continue
            if len(informative_params[param_name]) == 0:
                continue
            k = len(informative_params[param_name])
            row_idx = torch.arange(k).unsqueeze(1)
            final_idx_matrix = torch.cat(
                [informative_params[param_name], row_idx], dim=1
            )
            indices_for_sparse = final_idx_matrix.t().contiguous()
            mask_shape = tuple(param.size()) + (k,)
            key = f"mask_{param_name.replace('.', '_')}"
            sparse_masks[key] = torch.sparse_coo_tensor(
                indices_for_sparse,
                torch.ones(k, dtype=torch.float32),
                size=mask_shape,
            )
        for mask_name, mask in sparse_masks.items():
            self.register_buffer(mask_name, mask.coalesce())

    def contract_last_dim_with_vector(
        self, sp_tensor: torch.Tensor, vec: torch.Tensor
    ) -> torch.Tensor:
        indices = sp_tensor.indices()
        values = sp_tensor.values()
        new_values = values * vec[indices[-1]]
        new_shape = sp_tensor.shape[:-1]
        new_indices = indices[:-1, :]
        return torch.sparse_coo_tensor(
            new_indices,
            new_values,
            size=new_shape,
            dtype=sp_tensor.dtype,
            device=sp_tensor.device,
        )

    def _reconstructed_state(self):
        model = self.inner_model["model"]
        current_state = self.state_dict()
        final_params = {}
        for param_name in model.state_dict().keys():
            buf_name = param_name.replace(".", "_")
            final_params[param_name] = current_state[f"base_{buf_name}"]
        for key, param_vector in self.retrain_params.items():
            mask_key = f"mask_{key}"
            base_key = f"base_{key}"
            original_param_name = key.replace("_", ".")
            sparse_update = self.contract_last_dim_with_vector(
                current_state[mask_key], param_vector
            )
            final_params[original_param_name] = current_state[base_key] + sparse_update
        return final_params

    def forward(self, x):
        return torch.func.functional_call(
            self.inner_model["model"], self._reconstructed_state(), x
        )

    def get_retrained_params(self):
        return {
            key: value.cpu().clone().detach()
            for key, value in self._reconstructed_state().items()
        }
