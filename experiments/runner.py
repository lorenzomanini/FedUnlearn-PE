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


def _balanced_membership_mask(num_shadow_models, num_candidates, rng):
    if num_shadow_models < 4 or num_shadow_models % 2:
        raise ValueError("Online LiRA requires an even number of at least four shadow models.")
    if num_candidates < 2 or num_candidates % 2:
        raise ValueError("Online LiRA requires an even, non-empty candidate set.")

    mask = np.zeros((num_shadow_models, num_candidates), dtype=bool)
    for column in range(0, num_candidates, 2):
        rows = rng.permutation(num_shadow_models)
        mask[rows[: num_shadow_models // 2], column] = True
        mask[rows[num_shadow_models // 2 :], column + 1] = True

    # Random 2x2 edge switches preserve every row and column total while
    # removing the artificial one-to-one anti-correlation from construction.
    for _ in range(32 * num_candidates):
        row_a, row_b = rng.integers(num_shadow_models, size=2)
        column_a, column_b = rng.integers(num_candidates, size=2)
        if row_a == row_b or column_a == column_b:
            continue
        if (
            mask[row_a, column_a] != mask[row_a, column_b]
            and mask[row_a, column_a] != mask[row_b, column_a]
            and mask[row_a, column_a] == mask[row_b, column_b]
        ):
            mask[row_a, column_a] = ~mask[row_a, column_a]
            mask[row_a, column_b] = ~mask[row_a, column_b]
            mask[row_b, column_a] = ~mask[row_b, column_a]
            mask[row_b, column_b] = ~mask[row_b, column_b]
    return mask


def _prepare_online_lira(
    train_dataset,
    test_dataset,
    clients_subsets,
    model_class,
    loss_class,
    trainer_function,
    init_params_dict,
    evaluate_lira_function,
):
    """Train a reusable online-LiRA bank with per-record IN/OUT membership.

    Candidate membership is balanced across shadows. All other training records
    are sampled independently from the common pool, conditioned on the shadow
    having the same training-set size as the attacked model.
    """
    if init_params_dict.get("trainer_name") != "sgd":
        raise NotImplementedError(
            "Paper-faithful record-level LiRA currently requires the centralized SGD trainer."
        )

    num_shadow_models = int(init_params_dict["num_shadow_models"])
    seed = int(init_params_dict["lira_seed"])
    rng = np.random.default_rng(seed)
    target_client = init_params_dict["target_client"]
    target_indices = np.asarray(clients_subsets[target_client].indices, dtype=np.int64)

    fixed_train_indices = {}
    fixed_eval_indices = {}
    for client_index, subset in enumerate(clients_subsets):
        if client_index == target_client:
            continue
        indices = rng.permutation(np.asarray(subset.indices, dtype=np.int64))
        num_eval = int(len(indices) * 0.1)
        num_train = len(indices) - num_eval
        fixed_train_indices[client_index] = indices[:num_train].tolist()
        fixed_eval_indices[client_index] = indices[num_train:].tolist()

    held_out = np.asarray(
        [index for indices in fixed_eval_indices.values() for index in indices],
        dtype=np.int64,
    )
    shadow_train_dataset = clients_subsets[0].dataset
    test_indices = np.arange(len(test_dataset), dtype=np.int64) + len(shadow_train_dataset)
    nonmember_pool = np.concatenate([held_out, test_indices])
    num_targets = min(len(target_indices), len(nonmember_pool))
    if num_targets < 2:
        raise ValueError("Online LiRA requires at least two known members and non-members.")

    selected_targets = rng.choice(target_indices, num_targets, replace=False)
    selected_nonmembers = rng.choice(nonmember_pool, num_targets, replace=False)
    selected_held_out = set(
        selected_nonmembers[selected_nonmembers < len(shadow_train_dataset)].tolist()
    )
    for client_index, indices in fixed_eval_indices.items():
        fixed_eval_indices[client_index] = [
            index for index in indices if index not in selected_held_out
        ]

    retained = np.asarray(
        [index for indices in fixed_train_indices.values() for index in indices],
        dtype=np.int64,
    )
    candidate_indices = np.concatenate([selected_targets, selected_nonmembers])
    membership = _balanced_membership_mask(
        num_shadow_models, len(candidate_indices), rng
    )
    complete_dataset = torch.utils.data.ConcatDataset([shadow_train_dataset, test_dataset])
    candidate_dataset = Subset(complete_dataset, candidate_indices.tolist())
    candidate_index_set = set(candidate_indices.tolist())
    auxiliary_pool = np.asarray(
        [
            index
            for index in range(len(complete_dataset))
            if index not in candidate_index_set
        ],
        dtype=np.int64,
    )
    target_training_size = len(retained) + len(target_indices)
    auxiliary_size = target_training_size - membership.shape[1] // 2
    if auxiliary_size < 0 or auxiliary_size > len(auxiliary_pool):
        raise ValueError("The available data cannot match the target training-set size.")

    shadow_scores = []
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        for shadow_index, row in enumerate(membership):
            torch.manual_seed(seed + shadow_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + shadow_index)
            auxiliary_indices = rng.choice(
                auxiliary_pool, auxiliary_size, replace=False
            )
            shadow_indices = np.concatenate(
                [auxiliary_indices, candidate_indices[row]]
            )
            shadow_model = trainer_function(
                model_class(),
                loss_class(),
                [Subset(complete_dataset, shadow_indices.tolist())],
                [],
                init_params_dict["train_epochs"],
            )
            shadow_scores.append(evaluate_lira_function(shadow_model, candidate_dataset))

    bank = {
        "scores": np.stack(shadow_scores).astype(np.float32),
        "shadow_membership": membership,
        "candidate_membership": np.concatenate(
            [np.ones(num_targets, dtype=bool), np.zeros(num_targets, dtype=bool)]
        ),
        "candidate_complete_indices": candidate_indices,
        "num_shadow_models": np.asarray(num_shadow_models),
        "target_training_size": np.asarray(target_training_size),
        "seed": np.asarray(seed),
        "statistic": np.asarray("stable_logit_confidence_v1"),
        "sampling": np.asarray("balanced_candidates_random_auxiliary_v1"),
    }
    context = {
        "candidate_dataset": candidate_dataset,
        "fixed_train_indices": fixed_train_indices,
        "fixed_eval_indices": fixed_eval_indices,
    }
    return bank, context








class _RevisedTest:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, 
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None,
                 _information_method="diagonal", _num_power_iters=None):

        # Initialize parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.complete_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        self.clients_subsets = clients_subsets

        self.init_params_dict = init_params_dict
        self.target_client = init_params_dict['target_client']
        self.target_subset = self.clients_subsets[self.target_client]
        self.non_target_subsets = [subset for i, subset in enumerate(self.clients_subsets) if i != self.target_client]

        if _information_method == "spectral_wip":
            batch_size = 128
            self.client_loader = DataLoader(self.clients_subsets[self.target_client], batch_size, shuffle=True)
            self.total_loader = DataLoader(self.clients_subsets[0].dataset, batch_size, shuffle=True)
        
        self.poisoned_backdoor_dataset = poisoned_backdoor_dataset
        self.clean_backdoor_dataset = clean_backdoor_dataset

        self.model_class = model_class
        self.num_total_params = sum(p.numel() for p in model_class().parameters())
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        train_epochs = init_params_dict['train_epochs']

        logging.info("Preparing subsets for training...")
        shadow_out_subsets = []
        eval_subsets = []
        fixed_train = init_params_dict.get("_lira_fixed_train_indices")
        fixed_eval = init_params_dict.get("_lira_fixed_eval_indices")
        for client_index, subset in enumerate(self.clients_subsets):
            if client_index == self.target_client:
                continue
            if fixed_train is None:
                n_eval = int(len(subset) * 0.1)
                n_train = len(subset) - n_eval
                train, eval = random_split_subset(subset, [n_train, n_eval])
            else:
                train = Subset(subset.dataset, fixed_train[client_index])
                eval = Subset(subset.dataset, fixed_eval[client_index])
            shadow_out_subsets.append(train)
            if len(eval):
                eval_subsets.append(eval)

        self.retrain_subsets = shadow_out_subsets
        self.eval_subsets = eval_subsets

        train_subsets = shadow_out_subsets + [self.target_subset]

        logging.info("Training trained model...") 
        self.trained_model = self.trainer_function(
            self.model_class(), self.loss_class(), train_subsets, eval_subsets,
            train_epochs
        )

        if _information_method == "diagonal":
            logging.info("Computing information...")
            self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_subsets, stochastic_correction=init_params_dict.get('stochastic_correction', False), use_converter=init_params_dict.get('info_use_converter', True), method=init_params_dict['hessian_method'], learning_rate=init_params_dict['learning_rate'], momentum=init_params_dict['momentum'])

        logging.info("Training gold-standard client-removal model...")
        self.gold_retrain_model = self.trainer_function(
            self.model_class(), self.loss_class(), shadow_out_subsets, eval_subsets,
            train_epochs
        )
        # Artifact compatibility only: record-level LiRA OUT populations come
        # from False entries in the saved shadow-membership mask, not this model.
        self.shadow_out_model = self.gold_retrain_model

        logging.info("Computing initial evaluation results...")

        self.init_eval_test_results = {
            "trained": evaluate_model(self.trained_model, self.test_dataset),
            "shadow_out": evaluate_model(self.shadow_out_model, self.test_dataset),
        }
        self.init_eval_train_results = {
            "trained": evaluate_model(self.trained_model, self.train_dataset),
            "shadow_out": evaluate_model(self.shadow_out_model, self.train_dataset),
        }

        if _information_method == "spectral_wip":
            logging.info("Computing client information...")
            self.client_information = estimate_diag_commuting_backpack(self.trained_model.to(DEVICE), self.total_loader, self.client_loader, loss_class(), 10, DEVICE, _num_power_iters)["diag_by_name"]

    def configure_lira(self, candidate_dataset, evaluate_lira_function=evaluate_lira):
        self.lira_candidate_dataset = candidate_dataset
        self.evaluate_lira_function = evaluate_lira_function
        self.init_lira_results = {
            "trained": evaluate_lira_function(self.trained_model, candidate_dataset),
            "shadow_out": evaluate_lira_function(self.shadow_out_model, candidate_dataset),
        }

    def run_test(self, test_params_dict):

        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        logging.info(f"Unlearning: Method={unlearning_method}, Percentage={unlearning_percentage}, RetrainEpochs={retrain_epochs}")

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)
        num_reset_params = 0
        for indices_tensor in informative_params.values():
            if indices_tensor is not None and indices_tensor.numel() > 0: 
                 num_reset_params += indices_tensor.shape[0]

        if num_reset_params != 0:
            reset_model = self.model_class()
            reset_state_dict = reset_parameters(self.trained_model, informative_params)
            reset_model.load_state_dict(reset_state_dict)

            retrainer = UnlearnNet(reset_model, informative_params) 
            self.trainer_function(retrainer, self.loss_class(), self.retrain_subsets, self.eval_subsets, retrain_epochs)
            retrained_model = self.model_class()
            retrained_model.load_state_dict(retrainer.get_retrained_params())

            reset_params_percentage = num_reset_params / self.num_total_params * 100
            random_params = find_informative_params(self.client_information, 'random', reset_params_percentage, whitelist, blacklist)

            random_reset_model = self.model_class()
            random_reset_state_dict = reset_parameters(self.trained_model.cpu(), random_params)
            random_reset_model.load_state_dict(random_reset_state_dict)

            random_retrainer = UnlearnNet(random_reset_model, random_params)
            self.trainer_function(random_retrainer, self.loss_class(), self.retrain_subsets, self.eval_subsets, retrain_epochs)
            random_retrained_model = self.model_class()
            random_retrained_model.load_state_dict(random_retrainer.get_retrained_params())
        else:
            logging.warning("No parameters to reset, skipping unlearning and retraining.")
            reset_model = self.trained_model
            retrained_model = self.trained_model
            random_reset_model = self.trained_model
            random_retrained_model = self.trained_model
            reset_params_percentage = 0.0

        # Execute tests
        extra_results = {}

        extra_results['num_total_params'] = self.num_total_params
        extra_results['num_reset_params'] = num_reset_params
        extra_results['reset_params_percentage'] = reset_params_percentage

        logging.info(f"Percentage of reset parameters: {reset_params_percentage:.4f}% ({num_reset_params}/{self.num_total_params})")
        

        eval_test_results = {
            "reset": evaluate_model(reset_model, self.test_dataset),
            "retrained": evaluate_model(retrained_model, self.test_dataset),
            "random_reset": evaluate_model(random_reset_model, self.test_dataset),
            "random_retrained": evaluate_model(random_retrained_model, self.test_dataset)
        }
        eval_train_results = {
            "reset": evaluate_model(reset_model, self.train_dataset),
            "retrained": evaluate_model(retrained_model, self.train_dataset),
            "random_reset": evaluate_model(random_reset_model, self.train_dataset),
            "random_retrained": evaluate_model(random_retrained_model, self.train_dataset)
        }

        if hasattr(self, "lira_candidate_dataset"):
            self.last_lira_results = {
                "reset": self.evaluate_lira_function(reset_model, self.lira_candidate_dataset),
                "retrained": self.evaluate_lira_function(retrained_model, self.lira_candidate_dataset),
                "random_reset": self.evaluate_lira_function(random_reset_model, self.lira_candidate_dataset),
                "random_retrained": self.evaluate_lira_function(random_retrained_model, self.lira_candidate_dataset),
            }

        return eval_test_results, eval_train_results, extra_results


class Test(_RevisedTest):
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None):
        super().__init__(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                         init_params_dict, poisoned_backdoor_dataset, clean_backdoor_dataset)


class SpectralTest(_RevisedTest):
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None,
                 _num_power_iters=5):
        super().__init__(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function,
                         init_params_dict, poisoned_backdoor_dataset, clean_backdoor_dataset,
                         _information_method="spectral_wip", _num_power_iters=_num_power_iters)

    def plot_information_parameters_tradeoff(self, method='information', whitelist=None, blacklist=None):
        plot_information_parameters_tradeoff(self.client_information, method, whitelist=whitelist, blacklist=blacklist)
    







    



def init_worker(device_queue):
    logging.getLogger().setLevel(logging.INFO)
    device = device_queue.get()
    set_device(device)

def _run_tests_iter(iter, arg, test_class, device, filter_error_results=False, plot=False):
    test_path = arg['test_path']
    train_dataset = arg['train_dataset']
    test_dataset = arg['test_dataset']
    clients_subsets = arg['clients_subsets']
    model_class = arg['model_class']
    loss_class = arg['loss_class']
    trainer_function = arg['trainer_function']
    init_params_dict = arg['init_params_dict']
    test_params_dicts = arg['test_params_dicts']
    poisoned_backdoor_dataset = arg['poisoned_backdoor_dataset']
    clean_backdoor_dataset = arg['clean_backdoor_dataset']

    test_iter_path = os.path.join(test_path, f"test_{iter}")
    os.makedirs(test_iter_path)

    log_file_handler = logging.FileHandler(os.path.join(test_iter_path, f"test_{iter}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)
    logging.info(f"--- Starting Test Iteration {iter} ---")

    logging.info(f"Using device: {device}")

    run_init_params = init_params_dict.copy()
    lira_context = arg.get("lira_context")
    if lira_context is not None:
        run_init_params["_lira_fixed_train_indices"] = lira_context["fixed_train_indices"]
        run_init_params["_lira_fixed_eval_indices"] = lira_context["fixed_eval_indices"]
    test_instance = test_class(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, run_init_params, poisoned_backdoor_dataset, clean_backdoor_dataset)

    if lira_context is not None:
        test_instance.configure_lira(
            lira_context["candidate_dataset"], arg["evaluate_lira_function"]
        )

    if plot:
        test_instance.plot_information_parameters_tradeoff()

    persistence.dump_pickle(test_iter_path, persistence.INITIAL_TEST_RESULTS, test_instance.init_eval_test_results)
    persistence.dump_pickle(test_iter_path, persistence.INITIAL_TRAIN_RESULTS, test_instance.init_eval_train_results)
    if lira_context is not None:
        persistence.dump_pickle(
            test_iter_path, persistence.INITIAL_LIRA_RESULTS, test_instance.init_lira_results
        )

    acc_eval_test_results = []
    acc_eval_train_results = []
    acc_lira_results = []
    acc_extra_results = []
    errors = []
    for i, test_params_dict in enumerate(tqdm(test_params_dicts, desc=f"Unlearning tests", leave=False)):
        try:
            eval_test_result, eval_train_result, test_extra_result = test_instance.run_test(test_params_dict)
            acc_eval_test_results.append(eval_test_result)
            acc_eval_train_results.append(eval_train_result)
            acc_extra_results.append(test_extra_result)
            if lira_context is not None:
                acc_lira_results.append(test_instance.last_lira_results)
        except Exception as e:
            logging.error(f"Error in test {i} of iteration {iter}: {str(e)}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            logging.error(f"Traceback:\n{traceback_str}")
            errors.append(i)
            acc_eval_test_results.append({'error': str(e)})
            acc_extra_results.append({'error': str(e)})
            if filter_error_results:
                acc_eval_train_results.append({'error': str(e)})

    if filter_error_results:
        acc_eval_test_results = [result for result in acc_eval_test_results if 'error' not in result]
        acc_eval_train_results = [result for result in acc_eval_train_results if 'error' not in result]
        acc_extra_results = [result for result in acc_extra_results if 'error' not in result]
        eval_test_results = persistence.pack_revised_results(acc_eval_test_results) if acc_eval_test_results else {}
        eval_train_results = persistence.pack_revised_results(acc_eval_train_results) if acc_eval_train_results else {}
        extra_results = persistence.pack_result_lists(acc_extra_results) if acc_extra_results else {}
    else:
        eval_test_results = persistence.pack_revised_results(acc_eval_test_results)
        eval_train_results = persistence.pack_revised_results(acc_eval_train_results)
        extra_results = persistence.pack_result_lists(acc_extra_results)

    persistence.dump_npz(test_iter_path, persistence.EVAL_TEST_RESULTS, eval_test_results)
    persistence.dump_npz(test_iter_path, persistence.EVAL_TRAIN_RESULTS, eval_train_results)
    persistence.dump_pickle(test_iter_path, persistence.EXTRA_RESULTS, extra_results)
    if lira_context is not None and acc_lira_results:
        persistence.dump_npz(
            test_iter_path,
            persistence.EVAL_LIRA_RESULTS,
            persistence.pack_score_results(acc_lira_results),
        )

    logging.info(f"--- Finished Test Iteration {iter} ---")
    logging.getLogger().removeHandler(log_file_handler)
    log_file_handler.close()
    return errors


def run_tests_iter(iter, arg):
    return _run_tests_iter(iter, arg, Test, DEVICE)


def _run_repeated_tests(
    init_params_dict, test_params_dicts, save_path, num_workers, devices, save_models,
    run_tests_iter_function, init_worker_function, device, get_datasets_function,
    get_clients_subsets_function, get_model_class_function, get_loss_class_function,
    get_trainer_function_function, create_poisoned_data_function,
    plot=None, plot_results_function=None, evaluate_lira_function=evaluate_lira,
):

    init_params_dict = init_params_dict.copy()
    init_params_dict.setdefault("num_shadow_models", 64)
    init_params_dict.setdefault("lira_seed", 2026)
    init_params_dict.setdefault("lira_global_variance", True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_name = init_params_dict['test_name']
    logging.info(f"Starting test suite: {test_name}")

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

    log_file_handler = logging.FileHandler(os.path.join(test_path, f"{test_name}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)

    logging.info("Initial Configuration:")
    for key, value in init_params_dict.items():
        logging.info(f"  {key}: {value}")
    logging.info("-" * 30)

    persistence.dump_pickle(test_path, persistence.INIT_PARAMS, init_params_dict)
    persistence.dump_pickle(test_path, persistence.TEST_PARAMS, test_params_dicts)

    num_tests = init_params_dict['num_tests']

    train_dataset, test_dataset = get_datasets_function(init_params_dict)
    clients_subsets = get_clients_subsets_function(train_dataset, init_params_dict)
    model_class = get_model_class_function(init_params_dict)
    loss_class = get_loss_class_function(init_params_dict)
    trainer_function = get_trainer_function_function(init_params_dict)

    poisoned_backdoor_dataset = None
    clean_backdoor_dataset = None
    
    if init_params_dict.get('poison', False):
        logging.info("Poisoning is enabled. Applying backdoor attack...")
        clients_subsets, poisoned_backdoor_dataset, clean_backdoor_dataset = create_poisoned_data_function(clients_subsets, init_params_dict)
        logging.info("Poisoning complete.")
    else:
        logging.info("Poisoning is disabled.")

    client_indices = [subset.indices for subset in clients_subsets]
    persistence.dump_pickle(test_path, persistence.CLIENT_INDICES, client_indices)

    lira_context = None
    if init_params_dict["num_shadow_models"]:
        logging.info(
            "Training %s reusable online-LiRA shadow models...",
            init_params_dict["num_shadow_models"],
        )
        lira_bank, lira_context = _prepare_online_lira(
            train_dataset,
            test_dataset,
            clients_subsets,
            model_class,
            loss_class,
            trainer_function,
            init_params_dict,
            evaluate_lira_function,
        )
        persistence.dump_npz(test_path, persistence.LIRA_SHADOW_BANK, lira_bank)
    
    labels = {
        'train': [label for _, label in train_dataset],
        'test': [label for _, label in test_dataset]
    }
    persistence.dump_pickle(test_path, persistence.LABELS, labels)

    arg = {
        'test_path': test_path,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'clients_subsets': clients_subsets,
        'model_class': model_class,
        'loss_class': loss_class,
        'trainer_function': trainer_function,
        'init_params_dict': init_params_dict,
        'test_params_dicts': test_params_dicts,
        'poisoned_backdoor_dataset': poisoned_backdoor_dataset,
        'clean_backdoor_dataset': clean_backdoor_dataset,
        'save_models' : save_models,
        'lira_context': lira_context,
        'evaluate_lira_function': evaluate_lira_function,
    }
    if plot is not None:
        arg['plot'] = plot

    if num_workers == 1:
        with logging_redirect_tqdm():
            for i in tqdm(range(num_tests), desc="Running repeated tests"):
                logging.getLogger().removeHandler(log_file_handler)
                errors = run_tests_iter_function(i, arg)
                logging.getLogger().addHandler(log_file_handler)
                if len(errors) > 0:
                    logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")
    else:
        logging.info(f"Using {num_workers} workers for parallel processing.")
        if devices is None:
            logging.info(f"No devices provided, using default device {device} for all workers.")
            devices = [device] * num_workers
        elif len(devices) != num_workers:
            logging.error(f"Number of devices provided ({len(devices)}) does not match number of workers ({num_workers}). Using default device {device} for all workers.")
            devices = [device] * num_workers
        else:
            logging.info(f"Using provided devices: {devices}")

        device_queue = Queue(num_workers)
        for device in devices:
            device_queue.put(device)
                
        logging.getLogger().removeHandler(log_file_handler)
        os.environ['TQDM_DISABLE'] = '1'
        
        with Pool(num_workers, initializer=init_worker_function, initargs=(device_queue,)) as pool:
            iters_errors = pool.starmap(run_tests_iter_function, [(i, arg) for i in range(num_tests)])

        os.environ['TQDM_DISABLE'] = '0'
        logging.getLogger().addHandler(log_file_handler)

        for i, errors in enumerate(iters_errors):
            if len(errors) > 0:
                logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")

    logging.info(f"Test suite '{test_name}' completed")

    if plot:
        plot_results_function(test_path)


def run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1, devices=None, save_models=False):
    return _run_repeated_tests(
        init_params_dict, test_params_dicts, save_path, num_workers, devices, save_models,
        run_tests_iter, init_worker, DEVICE, get_datasets, get_clients_subsets,
        get_model_class, get_loss_class, get_trainer_function, create_poisoned_data,
    )


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

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)
