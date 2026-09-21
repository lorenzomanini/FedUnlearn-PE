"""Reset and retrain selected scalar coordinates without sparse mask tensors."""

import copy

import torch
from torch import nn


def _flat_indices(tensor, coordinates):
    if isinstance(coordinates, tuple):
        coordinates = (
            torch.stack(coordinates, dim=1)
            if coordinates
            else torch.empty((0, tensor.ndim), dtype=torch.long)
        )
    coordinates = torch.as_tensor(coordinates, dtype=torch.long, device=tensor.device)
    if coordinates.numel() == 0 and coordinates.ndim != 2:
        coordinates = coordinates.reshape(0, tensor.ndim)
    if coordinates.ndim != 2 or coordinates.shape[1] != tensor.ndim:
        raise ValueError("Selected coordinates must have shape (count, tensor.ndim).")
    flat = torch.zeros(coordinates.shape[0], dtype=torch.long, device=tensor.device)
    for axis, size in enumerate(tensor.shape):
        if ((coordinates[:, axis] < 0) | (coordinates[:, axis] >= size)).any():
            raise ValueError("Selected coordinate is out of bounds.")
        flat = flat * size + coordinates[:, axis]
    if flat.unique().numel() != flat.numel():
        raise ValueError("Selected coordinates must not contain duplicates.")
    return flat


def reset_parameters(model, informative_params):
    """Return an independent state dict with the selected coordinates zeroed."""
    reset_state = {}
    for name, tensor in model.state_dict().items():
        new_tensor = tensor.detach().clone(memory_format=torch.contiguous_format)
        if name in informative_params:
            indices = _flat_indices(new_tensor, informative_params[name])
            new_tensor.reshape(-1).index_fill_(0, indices, 0)
        reset_state[name] = new_tensor
    return reset_state


class UnlearnNet(nn.Module):
    """Retrain selected scalar parameters while keeping all others fixed.

    Only selected values are registered as trainable parameters. A differentiable
    indexed scatter reconstructs each affected tensor, avoiding sparse COO tensor
    creation and sparse-to-dense backward operations on every training batch.
    """

    def __init__(self, base_model, informative_params):
        super().__init__()
        # The template must not register a second, fully trainable parameter set.
        self.inner_model = {"model": copy.deepcopy(base_model)}
        self.inner_model["model"].requires_grad_(False)
        self._base_names = {}
        self._selected_names = {}
        used_keys = set()
        for param_name, tensor in reset_parameters(base_model, informative_params).items():
            key = param_name.replace(".", "_")
            while key in used_keys:
                key += "_"
            used_keys.add(key)
            self._base_names[param_name] = key
            self.register_buffer(f"base_{key}", tensor)

        retrain_params = {}
        for param_name, param in base_model.named_parameters():
            if param_name not in informative_params:
                continue
            indices = _flat_indices(param, informative_params[param_name])
            if not indices.numel():
                continue
            key = self._base_names[param_name]
            self._selected_names[param_name] = key
            self.register_buffer(f"indices_{key}", indices)
            retrain_params[key] = nn.Parameter(param.new_zeros(indices.numel()))
        self.retrain_params = nn.ParameterDict(retrain_params)
        self.train(base_model.training)

    def train(self, mode=True):
        super().train(mode)
        self.inner_model["model"].train(mode)
        return self

    def _reconstructed_state(self):
        final_params = {
            name: getattr(self, f"base_{key}")
            for name, key in self._base_names.items()
        }
        for name, key in self._selected_names.items():
            base = final_params[name]
            final_params[name] = base.reshape(-1).scatter(
                0, getattr(self, f"indices_{key}"), self.retrain_params[key]
            ).reshape_as(base)
        return final_params

    def forward(self, *args, **kwargs):
        return torch.func.functional_call(
            self.inner_model["model"], self._reconstructed_state(), args, kwargs
        )

    def get_retrained_params(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self._reconstructed_state().items()
        }
