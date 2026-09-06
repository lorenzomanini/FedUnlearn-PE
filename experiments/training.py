import copy
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fisherunlearn.data.partition import concatenate_subsets


def legacy_simple_trainer(
    model,
    loss_fn,
    subsets,
    epochs,
    comm_tracker=None,
    train_batch_size=5000,
    eval_batch_size=5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    dataset = concatenate_subsets(subsets)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))],
    )
    dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, eval_batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", leave=False):
        if comm_tracker is not None:
            comm_tracker.record_round()
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
        logging.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss_accum / n_batches}, Val Loss: {val_loss_accum / val_n_batches}"
        )
    model.eval()
    return model.cpu()


def revised_simple_trainer(
    model,
    loss_fn,
    train_subsets,
    val_subsets,
    epochs,
    init_params_dict,
    train_batch_size=5000,
    eval_batch_size=5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = concatenate_subsets(train_subsets)
    dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    val_dataloader = None
    if val_subsets:
        val_dataset = concatenate_subsets(val_subsets)
        val_dataloader = DataLoader(val_dataset, eval_batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=init_params_dict["learning_rate"],
        momentum=init_params_dict["momentum"],
    )
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", leave=False):
        model.train()
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
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = loss_fn(val_outputs, val_targets)
                    val_loss_accum += val_loss.item()
                    val_n_batches += 1
        val_text = val_loss_accum / val_n_batches if val_n_batches else "n/a"
        logging.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss_accum / n_batches}, Val Loss: {val_text}"
        )
    model.eval()
    return model.cpu()


def fedavg_trainer(
    model,
    loss_fn,
    subsets,
    epochs,
    comm_tracker=None,
    init_params_dict=None,
    train_batch_size=5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = (
        init_params_dict.get("local_epochs", 1) if init_params_dict else 1
    )
    participation_rate = (
        init_params_dict.get("participation_rate", 1.0)
        if init_params_dict
        else 1.0
    )
    lr = init_params_dict.get("lr", 1e-3) if init_params_dict else 1e-3
    num_clients = len(subsets)
    num_participating = max(1, int(participation_rate * num_clients))
    model.to(device)
    for round in tqdm(
        range(epochs), desc="FedAvg Rounds", unit="round", leave=False
    ):
        participating_idxs = np.random.choice(
            num_clients, num_participating, replace=False
        )
        if comm_tracker:
            comm_tracker.record_round(participating_idxs)
        local_weights = []
        local_sizes = []
        global_state = model.state_dict()
        for client_idx in participating_idxs:
            client_model = copy.deepcopy(model)
            client_model.load_state_dict(global_state)
            client_model.train()
            optimizer = torch.optim.AdamW(
                client_model.parameters(), lr=lr, weight_decay=1e-2
            )
            train_loader = DataLoader(
                subsets[client_idx], batch_size=train_batch_size, shuffle=True
            )
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

        total_size = sum(local_sizes)
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * local_sizes[0]
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key] * local_sizes[i]
            avg_weights[key] = torch.div(avg_weights[key], total_size)
        model.load_state_dict(avg_weights)
    return model.cpu()
