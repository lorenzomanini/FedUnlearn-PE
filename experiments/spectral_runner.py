from analysis.results import plot_experiment_results
from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets, random_split_subset, create_poisoned_data, poisoning_data
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters, mia_attack
from fisherunlearn import UnlearnNet
from fisherunlearn import plot_information_parameters_tradeoff
from fisherunlearn.information.spectral_wip import estimate_diag_commuting_backpack

import fisherunlearn

import os
import pickle
import random
import logging
import functools
import sys, traceback

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

NUM_POWER_ITERS = int(os.environ.get("NUM_POWER_ITERS", "5"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 5000
TRAIN_BATCH_SIZE = 5000
INFO_BATCH_SIZE = 1000
MIA_BATCH_SIZE = 1000

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


from experiments.config import RevisedInitParamsDict as InitParamsDict, TestParamsDict
from experiments.datasets import get_clients_subsets, get_datasets
from experiments.models import (
    FLNet,
    FLNet2,
    FeedForwardNN,
    create_resnet,
    get_loss_class,
    get_model_class,
)
from experiments import evaluation as _evaluation
from experiments import training as _training
from experiments import persistence

logit_margin = _evaluation.logit_margin
logit_confidence = _evaluation.logit_confidence


def compute_accuracy(model, dataset):
    return _evaluation.compute_accuracy(model, dataset, DEVICE, EVAL_BATCH_SIZE)


def evaluate_model(model, dataset):
    return _evaluation.evaluate_model(model, dataset, DEVICE, EVAL_BATCH_SIZE)


def evaluate_lira(model, dataset):
    return _evaluation.evaluate_lira(model, dataset, DEVICE, EVAL_BATCH_SIZE)


def simple_trainer(model, loss_fn, train_subsets, val_subsets, epochs, init_params_dict):
    return _training.revised_simple_trainer(
        model,
        loss_fn,
        train_subsets,
        val_subsets,
        epochs,
        init_params_dict,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )


def fedavg_trainer(model, loss_fn, subsets, epochs, comm_tracker=None, init_params_dict=None):
    return _training.fedavg_trainer(
        model,
        loss_fn,
        subsets,
        epochs,
        comm_tracker=comm_tracker,
        init_params_dict=init_params_dict,
        train_batch_size=TRAIN_BATCH_SIZE,
    )


def get_trainer_function(init_params_dict):
    trainer_name = init_params_dict['trainer_name']
    if trainer_name == 'sgd':
        if 'learning_rate' not in init_params_dict:
            init_params_dict['learning_rate'] = 0.01
        if 'momentum' not in init_params_dict:
            init_params_dict['momentum'] = 0.9
        return functools.partial(simple_trainer, init_params_dict=init_params_dict)
    elif trainer_name == 'fedavg':
        return functools.partial(fedavg_trainer, init_params_dict=init_params_dict)
    else:
        raise ValueError(f"Unsupported trainer name: {trainer_name}")








from experiments import runner as _shared_runner


class Test(_shared_runner.SpectralTest):
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None):
        super().__init__(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                         init_params_dict, poisoned_backdoor_dataset, clean_backdoor_dataset,
                         _num_power_iters=NUM_POWER_ITERS)
    







    



def init_worker(device_queue):
    logging.getLogger().setLevel(logging.INFO)
    device = device_queue.get()
    set_device(device)

def run_tests_iter(iter, arg):
    return _shared_runner._run_tests_iter(
        iter, arg, Test, DEVICE, filter_error_results=True, plot=arg.get('plot', False)
    )


def run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1, devices=None, save_models=False, plot=False):
    return _shared_runner._run_repeated_tests(
        init_params_dict, test_params_dicts, save_path, num_workers, devices, save_models,
        run_tests_iter, init_worker, DEVICE, get_datasets, get_clients_subsets,
        get_model_class, get_loss_class, get_trainer_function, create_poisoned_data,
        plot=plot, plot_results_function=plot_experiment_results,
        evaluate_lira_function=evaluate_lira,
    )


# ─── Analysis & plotting helpers ──────────────────────────────────────────────


















if __name__ == "__main__":
    save_path = r'.\stat_tests\NEW_TESTER'

    num_tests = 1
    num_workers = 1
    set_batch_sizes(128,128,128,128)

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_random',

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 1,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': "diag_ggn_mc",
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }
    
    test_params_dict_0 = test_params_dict.copy()
    test_params_dict_0['subtest'] = 0
    test_params_dict_0['unlearning_method'] = 'information'

    percentages = np.arange(0, 10, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers, plot=True)
