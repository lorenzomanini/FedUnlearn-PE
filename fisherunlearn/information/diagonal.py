import copy

import torch
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .. import runtime


def compute_client_information(
    client_idx,
    model,
    criterion,
    datasets_list,
    method="diag_ggn",
    stochastic_correction=False,
    use_converter=True,
    gamma_stochastic=1.0,
    momentum=None,
    learning_rate=None,
):
    backpack_extension = None
    backpack_parameter = None

    if method == "diag_hessian":
        backpack_extension = DiagHessian()
        backpack_parameter = "diag_h"
    elif method == "diag_ggn":
        backpack_extension = DiagGGNExact()
        backpack_parameter = "diag_ggn_exact"
    elif method == "diag_ggn_mc":
        backpack_extension = DiagGGNMC()
        backpack_parameter = "diag_ggn_mc"
    elif method == "zeros":
        stochastic_correction = False
    else:
        raise ValueError(
            "Invalid method. Use 'diag_hessian', 'diag_ggn' or 'diag_ggn_mc'."
        )

    if gamma_stochastic == 0:
        stochastic_correction = False

    model = copy.deepcopy(model).to(runtime.DEVICE).eval()
    criterion = copy.deepcopy(criterion).to(runtime.DEVICE)
    model = extend(model, use_converter=use_converter)
    criterion = extend(criterion)

    target_hessian = {}
    total_hessian = {}
    target_gradients = {}
    total_gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            target_hessian[name] = torch.zeros_like(param)
            total_hessian[name] = torch.zeros_like(param)
            if stochastic_correction:
                target_gradients[name] = []
                total_gradients[name] = []

    dataloader_list = [
        DataLoader(dataset, batch_size=runtime.INFO_BATCH_SIZE, shuffle=False)
        for dataset in datasets_list
    ]
    num_batches = sum(len(loader) for loader in dataloader_list)
    tqdm_bar = tqdm(
        total=num_batches,
        desc="Computing clients information",
        unit="batch",
        leave=False,
    )

    if method != "zeros":
        for loader_idx, loader in enumerate(dataloader_list):
            for inputs, targets in loader:
                diag_h = {}
                grad = {}
                inputs = inputs.to(runtime.DEVICE)
                targets = targets.to(runtime.DEVICE)
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                with backpack(backpack_extension):
                    loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        diag_h = (
                            getattr(param, backpack_parameter).clone().detach()
                            * len(inputs)
                        )
                        total_hessian[name] += diag_h
                        if loader_idx == client_idx:
                            target_hessian[name] += diag_h
                        if stochastic_correction:
                            grad = param.grad.clone().detach() * len(inputs)
                            total_gradients[name].append(grad)
                            if loader_idx == client_idx:
                                target_gradients[name].append(grad)
                tqdm_bar.update(1)
    tqdm_bar.close()

    target_client_info = {}
    for name in target_hessian.keys():
        if stochastic_correction:
            target_grad = torch.stack(target_gradients[name], dim=0)
            total_grad = torch.stack(total_gradients[name], dim=0)
            target_var = torch.var(target_grad, dim=0, unbiased=True)
            total_var = torch.var(total_grad, dim=0, unbiased=True)
            factor = (gamma_stochastic * learning_rate) / (
                runtime.INFO_BATCH_SIZE * momentum
            )
            correction = target_var / (1 / factor + total_var)
            layer_info = torch.pow(
                target_hessian[name] / total_hessian[name] - correction, 2
            )
        else:
            layer_info = torch.pow(target_hessian[name] / total_hessian[name], 2)
        layer_info[total_hessian[name] <= 0] = 0
        layer_info[target_hessian[name] <= 0] = 0
        target_client_info[name] = layer_info.detach().cpu()
    return target_client_info
