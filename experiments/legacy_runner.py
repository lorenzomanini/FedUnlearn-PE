from fisherunlearn.clients_utils import split_dataset_by_class_distribution, concatenate_subsets, create_poisoned_data, poisoning_data
from fisherunlearn import compute_client_information, find_informative_params, reset_parameters, mia_attack
from fisherunlearn import UnlearnNet

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
from experiments.communication import CommunicationTracker
from experiments import persistence

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


from experiments.config import LegacyInitParamsDict as InitParamsDict, TestParamsDict
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


def compute_accuracy(model, dataset):
    return _evaluation.compute_accuracy(model, dataset, DEVICE, EVAL_BATCH_SIZE)


class PerformanceEvaluator(_evaluation.PerformanceEvaluator):
    def __init__(self, model, dataset):
        super().__init__(model, dataset, DEVICE, EVAL_BATCH_SIZE)


def simple_trainer(model, loss_fn, subsets, epochs, comm_tracker=None):
    return _training.legacy_simple_trainer(
        model,
        loss_fn,
        subsets,
        epochs,
        comm_tracker=comm_tracker,
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
        return simple_trainer
    elif trainer_name == 'fedavg':
        return functools.partial(fedavg_trainer, init_params_dict=init_params_dict)
    else:
        raise ValueError(f"Unsupported trainer name: {trainer_name}")







class Test:
    def __init__(self, train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, 
                 init_params_dict={}, poisoned_backdoor_dataset=None, clean_backdoor_dataset=None):

        # Initialize parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clients_subsets = clients_subsets
        self.target_client = init_params_dict.get('target_client', 0)

        self.target_subset = self.clients_subsets[self.target_client]
        self.benchmark_subsets = [subset for i, subset in enumerate(self.clients_subsets) if i != self.target_client]
        
        self.poisoned_backdoor_dataset = poisoned_backdoor_dataset
        self.clean_backdoor_dataset = clean_backdoor_dataset

        self.model_class = model_class
        self.loss_class = loss_class
        self.trainer_function = trainer_function

        self.train_epochs = init_params_dict['train_epochs']
        self.info_use_converter = init_params_dict.get('info_use_converter', True)


        self.trained_model = None
        self.benchmark_model = None
        self.client_information = None

        # Initialize communication trackers
        self.initial_training_comm_tracker = CommunicationTracker(len(self.clients_subsets), self.model_class())
        self.benchmark_training_comm_tracker = CommunicationTracker(len(self.benchmark_subsets), self.model_class())
        # Tracker for unlearning phase ONLY (Hessian exchange)
        self.unlearning_comm_tracker = CommunicationTracker(len(self.clients_subsets), self.model_class())

        logging.info("Training model...") 
        self.trained_model = self.trainer_function(
            self.model_class(), self.loss_class(), self.clients_subsets, 
            self.train_epochs, comm_tracker=self.initial_training_comm_tracker
        )
        self.num_total_params = sum(p.numel() for p in self.trained_model.parameters())

        logging.info("Training benchmark model...") 
        self.benchmark_model = self.trainer_function(
            self.model_class(), self.loss_class(), self.benchmark_subsets, 
            self.train_epochs, comm_tracker=self.benchmark_training_comm_tracker
        )

        logging.info("Computing information...")
        hessian_method = init_params_dict.get('hessian_method', 'diag_ggn')
        stochastic_correction = init_params_dict.get('stochastic_correction', False)
        self.client_information = compute_client_information(self.target_client, self.trained_model, self.loss_class(), self.clients_subsets, stochastic_correction=stochastic_correction, use_converter=self.info_use_converter, method=hessian_method)
        
        # Record Hessian computation communication on the UNLEARNING tracker
        # (Downlink: Model, Uplink: Diagonal Hessian per client)
        # Diagonal Hessian has the same size as the model parameters.
        self.unlearning_comm_tracker.record_round()
        
        self.init_train_PE_dict = {}
        tqdm_bar = tqdm(total=2, desc="Initializing Train Dataset Performance Evaluators", unit="PE", leave=False)
        self.init_train_PE_dict['trained'] = PerformanceEvaluator(self.trained_model, self.train_dataset)
        tqdm_bar.update(1)
        self.init_train_PE_dict['benchmark'] = PerformanceEvaluator(self.benchmark_model, self.train_dataset)
        tqdm_bar.close()

        self.init_test_PE_dict = {}
        tqdm_bar = tqdm(total=2, desc="Initializing Test Dataset Performance Evaluators", unit="PE", leave=False)
        self.init_test_PE_dict['trained'] = PerformanceEvaluator(self.trained_model, self.test_dataset)
        tqdm_bar.update(1)
        self.init_test_PE_dict['benchmark'] = PerformanceEvaluator(self.benchmark_model, self.test_dataset)
        tqdm_bar.close()


    def run_test(self, test_params_dict):

        unlearning_method = test_params_dict['unlearning_method']
        unlearning_percentage = test_params_dict['unlearning_percentage']
        retrain_epochs = test_params_dict['retrain_epochs']
        whitelist = test_params_dict.get('whitelist', None)
        blacklist = test_params_dict.get('blacklist', None)

        logging.info(f"Unlearning: Method={unlearning_method}, Percentage={unlearning_percentage}, RetrainEpochs={retrain_epochs}")

        informative_params = find_informative_params(self.client_information, unlearning_method, unlearning_percentage, whitelist, blacklist)
        num_reset_params = 0
        for name, indices_tensor in informative_params.items():
            if indices_tensor is not None and indices_tensor.numel() > 0: 
                 num_reset_params += indices_tensor.shape[0]

        # Initialize communication trackers for retraining
        retrain_comm_tracker = CommunicationTracker(len(self.benchmark_subsets), self.model_class())
        random_retrain_comm_tracker = CommunicationTracker(len(self.benchmark_subsets), self.model_class())

        if num_reset_params != 0:
            reset_model = self.model_class()
            reset_state_dict = reset_parameters(self.trained_model.cpu(), informative_params)
            reset_model.load_state_dict(reset_state_dict)

            # Record the broadcast of the reset model to clients (Server -> Clients)
            # This is the "receive the new model with the reset parameters" part.
            self.unlearning_comm_tracker.record_downlink()

            retrainer = UnlearnNet(reset_model, informative_params) 
            self.trainer_function(retrainer, self.loss_class(), self.benchmark_subsets, retrain_epochs, comm_tracker=retrain_comm_tracker)
            retrained_model = self.model_class()
            retrained_model.load_state_dict(retrainer.get_retrained_params())

            reset_params_percentage = num_reset_params / self.num_total_params * 100
            random_params = find_informative_params(self.client_information, 'random', reset_params_percentage, whitelist, blacklist)

            random_reset_model = self.model_class()
            random_reset_state_dict = reset_parameters(self.trained_model.cpu(), random_params)
            random_reset_model.load_state_dict(random_reset_state_dict)

            random_retrainer = UnlearnNet(random_reset_model, random_params)
            self.trainer_function(random_retrainer, self.loss_class(), self.benchmark_subsets, retrain_epochs, comm_tracker=random_retrain_comm_tracker)
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
        result = {}

        result['num_total_params'] = self.num_total_params
        result['num_reset_params'] = num_reset_params
        result['reset_params_percentage'] = reset_params_percentage

        # Add communication metrics to results
        initial_comm = self.initial_training_comm_tracker.get_metrics()
        benchmark_comm = self.benchmark_training_comm_tracker.get_metrics()
        unlearning_hessian_comm = self.unlearning_comm_tracker.get_metrics()
        retrain_comm = retrain_comm_tracker.get_metrics()
        random_retrain_comm = random_retrain_comm_tracker.get_metrics()
        
        result['communication_metrics'] = {
            'initial_training': initial_comm,
            'benchmark_training': benchmark_comm,
            'unlearning_hessian': unlearning_hessian_comm,
            'unlearning_retraining': retrain_comm,
            'random_baseline_retraining': random_retrain_comm,
        }
        
        # Summary communication metrics for easy access (initial training + unlearning)
        result['total_communication_rounds'] = initial_comm['communication_rounds'] + unlearning_hessian_comm['communication_rounds']
        result['total_communication_bytes'] = initial_comm['total_communication_bytes'] + unlearning_hessian_comm['total_communication_bytes']
        result['benchmark_communication_rounds'] = benchmark_comm['communication_rounds']
        result['benchmark_communication_bytes'] = benchmark_comm['total_communication_bytes']
        
        # Per-phase breakdown
        result['initial_training_rounds'] = initial_comm['communication_rounds']
        result['initial_training_bytes'] = initial_comm['total_communication_bytes']
        
        # UNLEARNING ONLY metrics (Hessian exchange + Reset Model Broadcast)
        # Note: We explicitly EXCLUDE the retraining cost here as requested.
        result['unlearning_hessian_rounds'] = unlearning_hessian_comm['communication_rounds']
        result['unlearning_hessian_bytes'] = unlearning_hessian_comm['total_communication_bytes']
        result['unlearning_retraining_rounds'] = retrain_comm['communication_rounds']
        result['unlearning_retraining_bytes'] = retrain_comm['total_communication_bytes']
        
        result['unlearning_total_rounds'] = unlearning_hessian_comm['communication_rounds']
        result['unlearning_total_bytes'] = unlearning_hessian_comm['total_communication_bytes']

        train_PE_dict = {}
        tqdm_bar = tqdm(total=4, desc="Initializing Train Dataset Performance Evaluators", unit="PE", leave=False)
        train_PE_dict['reset'] = PerformanceEvaluator(reset_model, self.train_dataset)
        tqdm_bar.update(1)
        train_PE_dict['retrained'] = PerformanceEvaluator(retrained_model, self.train_dataset)
        tqdm_bar.update(1)
        train_PE_dict['random_reset'] = PerformanceEvaluator(random_reset_model, self.train_dataset)
        tqdm_bar.update(1)
        train_PE_dict['random_retrained'] = PerformanceEvaluator(random_retrained_model, self.train_dataset)
        tqdm_bar.close()

        test_PE_dict = {}
        tqdm_bar = tqdm(total=4, desc="Initializing Test Dataset Performance Evaluators", unit="PE", leave=False)
        test_PE_dict['reset'] = PerformanceEvaluator(reset_model, self.test_dataset)
        tqdm_bar.update(1)
        test_PE_dict['retrained'] = PerformanceEvaluator(retrained_model, self.test_dataset)
        tqdm_bar.update(1)
        test_PE_dict['random_reset'] = PerformanceEvaluator(random_reset_model, self.test_dataset)
        tqdm_bar.update(1)
        test_PE_dict['random_retrained'] = PerformanceEvaluator(random_retrained_model, self.test_dataset)
        tqdm_bar.close()

        if 'test_accuracy' in test_params_dict['tests']:
            logging.info("Computing test accuracies...")

            try:
                result['trained_test_accuracy'] = self.trained_test_accuracy
                result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
            except AttributeError:
                self.trained_test_accuracy = self.init_test_PE_dict['trained'].get_accuracy()
                self.benchmark_test_accuracy = self.init_test_PE_dict['benchmark'].get_accuracy()
                result['trained_test_accuracy'] = self.trained_test_accuracy
                result['benchmark_test_accuracy'] = self.benchmark_test_accuracy
            
            for key, pe in test_PE_dict.items():
                result[f'{key}_test_accuracy'] = pe.get_accuracy()


        if 'target_accuracy' in test_params_dict['tests']:
            logging.info("Computing target accuracies...")
            try:
                result['trained_target_accuracy'] = self.trained_target_accuracy
                result['benchmark_target_accuracy'] = self.benchmark_target_accuracy
            except AttributeError:
                self.trained_target_accuracy = self.init_train_PE_dict['trained'].get_accuracy(self.target_subset)
                self.benchmark_target_accuracy = self.init_train_PE_dict['benchmark'].get_accuracy(self.target_subset)
                result['trained_target_accuracy'] = self.trained_target_accuracy
                result['benchmark_target_accuracy'] = self.benchmark_target_accuracy

            for key, pe in train_PE_dict.items():
                result[f'{key}_target_accuracy'] = pe.get_accuracy(self.target_subset)

        if 'clients_accuracies' in test_params_dict['tests']:
            logging.info("Computing clients accuracies...")
            try:
                result['trained_clients_accuracies'] = self.trained_clients_accuracies
                result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
            except AttributeError:
                self.trained_clients_accuracies = [self.init_train_PE_dict['trained'].get_accuracy(subset) for subset in self.clients_subsets]
                self.benchmark_clients_accuracies = [self.init_train_PE_dict['benchmark'].get_accuracy(subset) for subset in self.clients_subsets]
                result['trained_clients_accuracies'] = self.trained_clients_accuracies
                result['benchmark_clients_accuracies'] = self.benchmark_clients_accuracies
            
            for key, pe in train_PE_dict.items():
                result[f'{key}_clients_accuracies'] = [pe.get_accuracy(subset) for subset in self.clients_subsets]

        if 'class_accuracies' in test_params_dict['tests']:
            logging.info("Computing class accuracies...")
            try:
                result['trained_class_accuracies'] = self.trained_class_accuracies
                result['benchmark_class_accuracies'] = self.benchmark_class_accuracies
            except AttributeError:
                self.trained_class_accuracies = [self.init_train_PE_dict['trained'].get_accuracy(subset) for subset in self.classes_subsets]
                self.benchmark_class_accuracies = [self.init_train_PE_dict['benchmark'].get_accuracy(subset) for subset in self.classes_subsets]
                result['trained_class_accuracies'] = self.trained_class_accuracies
                result['benchmark_class_accuracies'] = self.benchmark_class_accuracies

            for key, pe in train_PE_dict.items():
                result[f'{key}_class_accuracies'] = [pe.get_accuracy(subset) for subset in self.classes_subsets]

        if 'LiRA' in test_params_dict['tests']:
            raise NotImplementedError(
                "The legacy single-benchmark score is not LiRA. Use the revised "
                "runner, which records explicit per-example shadow-IN and "
                "shadow-OUT populations."
            )


        if 'poisoned_backdoor_accuracy' in test_params_dict['tests'] and self.poisoned_backdoor_dataset:
            logging.info("Computing Poisoned Backdoor Accuracy...")
            try:
                result['trained_poisoned_backdoor_accuracy'] = self.trained_poisoned_backdoor_accuracy
                result['benchmark_poisoned_backdoor_accuracy'] = self.benchmark_poisoned_backdoor_accuracy
            except AttributeError:
                self.trained_poisoned_backdoor_accuracy = compute_accuracy(self.trained_model, self.poisoned_backdoor_dataset)
                self.benchmark_poisoned_backdoor_accuracy = compute_accuracy(self.benchmark_model, self.poisoned_backdoor_dataset)
                result['trained_poisoned_backdoor_accuracy'] = self.trained_poisoned_backdoor_accuracy
                result['benchmark_poisoned_backdoor_accuracy'] = self.benchmark_poisoned_backdoor_accuracy

            result['reset_poisoned_backdoor_accuracy'] = compute_accuracy(reset_model, self.poisoned_backdoor_dataset)
            result['retrained_poisoned_backdoor_accuracy'] = compute_accuracy(retrained_model, self.poisoned_backdoor_dataset)
            result['random_reset_poisoned_backdoor_accuracy'] = compute_accuracy(random_reset_model, self.poisoned_backdoor_dataset)
            result['random_retrained_poisoned_backdoor_accuracy'] = compute_accuracy(random_retrained_model, self.poisoned_backdoor_dataset)

        if 'clean_backdoor_accuracy' in test_params_dict['tests'] and self.clean_backdoor_dataset:
            logging.info("Computing Clean Backdoor Accuracy...")
            try:
                result['trained_clean_backdoor_accuracy'] = self.trained_clean_backdoor_accuracy
                result['benchmark_clean_backdoor_accuracy'] = self.benchmark_clean_backdoor_accuracy
            except AttributeError:
                self.trained_clean_backdoor_accuracy = compute_accuracy(self.trained_model, self.clean_backdoor_dataset)
                self.benchmark_clean_backdoor_accuracy = compute_accuracy(self.benchmark_model, self.clean_backdoor_dataset)
                result['trained_clean_backdoor_accuracy'] = self.trained_clean_backdoor_accuracy
                result['benchmark_clean_backdoor_accuracy'] = self.benchmark_clean_backdoor_accuracy
            result['reset_clean_backdoor_accuracy'] = compute_accuracy(reset_model, self.clean_backdoor_dataset)
            result['retrained_clean_backdoor_accuracy'] = compute_accuracy(retrained_model, self.clean_backdoor_dataset)
            result['random_reset_clean_backdoor_accuracy'] = compute_accuracy(random_reset_model, self.clean_backdoor_dataset)
            result['random_retrained_clean_backdoor_accuracy'] = compute_accuracy(random_retrained_model, self.clean_backdoor_dataset)

        if 'mia' in test_params_dict['tests']:
            logging.info("Running MIA...")
            for classifier_type in test_params_dict['mia_classifier_types']:
                logging.info(f"Classifier type: {classifier_type}")

                try:
                    result[f'trained_mia_{classifier_type}'] = self.trained_mia
                    result[f'benchmark_mia_{classifier_type}'] = self.benchmark_mia
                except AttributeError:
                    self.trained_mia = mia_attack(self.trained_model, self.target_subset, self.test_dataset, classifier_type)
                    self.benchmark_mia = mia_attack(self.benchmark_model, self.target_subset, self.test_dataset, classifier_type)
                    if self.trained_mia["accuracy"] == self.benchmark_mia["accuracy"]:
                        logging.warning("Trained and Benchmark MIA have the same accuracy")
                    result[f'trained_mia_{classifier_type}'] = self.trained_mia
                    result[f'benchmark_mia_{classifier_type}'] = self.benchmark_mia

                result[f'reset_mia_{classifier_type}'] = mia_attack(reset_model, self.target_subset, self.test_dataset, classifier_type)
                result[f'retrained_mia_{classifier_type}'] = mia_attack(retrained_model, self.target_subset, self.test_dataset, classifier_type)
                result[f'random_reset_mia_{classifier_type}'] = mia_attack(random_reset_model, self.target_subset, self.test_dataset, classifier_type)
                result[f'random_retrained_mia_{classifier_type}'] = mia_attack(random_retrained_model, self.target_subset, self.test_dataset, classifier_type)

        return result
    







    



def init_worker(device_queue):
    logging.getLogger().setLevel(logging.INFO)
    device = device_queue.get()
    set_device(device)

def run_tests_iter(iter, arg):
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
    save_models = arg['save_models']

    test_iter_path = os.path.join(test_path, f"test_{iter}")
    os.makedirs(test_iter_path)

    log_file_handler = logging.FileHandler(os.path.join(test_iter_path, f"test_{iter}.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)
    logging.info(f"--- Starting Test Iteration {iter} ---")

    logging.info(f"Using device: {DEVICE}")

    test_instance = Test(train_dataset, test_dataset, clients_subsets, model_class, loss_class, trainer_function, init_params_dict, poisoned_backdoor_dataset, clean_backdoor_dataset)
    
    if save_models:
        persistence.save_legacy_models(test_iter_path, test_instance)

    iteration_results = []
    errors = []
    for i, test_params_dict in enumerate(tqdm(test_params_dicts, desc=f"Unlearning tests", leave=False)):
        try:
            test_result = test_instance.run_test(test_params_dict)
            iteration_results.append(test_result)
        except Exception as e:
            logging.error(f"Error in test {i} of iteration {iter}: {str(e)}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            logging.error(f"Traceback:\n{traceback_str}")
            errors.append(i)
            iteration_results.append({'error': str(e)})
            
        persistence.dump_pickle(test_iter_path, persistence.LEGACY_RESULTS, iteration_results)

    logging.info(f"--- Finished Test Iteration {iter} ---")
    logging.getLogger().removeHandler(log_file_handler)
    log_file_handler.close()
    return errors


def run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1, devices=None, save_models=False):

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

    train_dataset, test_dataset = get_datasets(init_params_dict)
    clients_subsets = get_clients_subsets(train_dataset, init_params_dict)
    model_class = get_model_class(init_params_dict)
    loss_class = get_loss_class(init_params_dict) 
    trainer_function = get_trainer_function(init_params_dict)

    poisoned_backdoor_dataset = None
    clean_backdoor_dataset = None
    
    if init_params_dict.get('poison', False):
        logging.info("Poisoning is enabled. Applying backdoor attack...")
        clients_subsets, poisoned_backdoor_dataset, clean_backdoor_dataset = create_poisoned_data(clients_subsets, init_params_dict)
        logging.info("Poisoning complete.")
    else:
        logging.info("Poisoning is disabled.")

    client_indices = [subset.indices for subset in clients_subsets]
    persistence.dump_pickle(test_path, persistence.CLIENT_INDICES, client_indices)

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
        'save_models' : save_models
    }

    if num_workers == 1:
        with logging_redirect_tqdm():
            for i in tqdm(range(num_tests), desc="Running repeated tests"):
                logging.getLogger().removeHandler(log_file_handler)
                errors = run_tests_iter(i, arg)    
                logging.getLogger().addHandler(log_file_handler)
                if len(errors) > 0:
                    logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")
    else:
        logging.info(f"Using {num_workers} workers for parallel processing.")
        if devices is None:
            logging.info(f"No devices provided, using default device {DEVICE} for all workers.")
            devices = [DEVICE] * num_workers
        elif len(devices) != num_workers:
            logging.error(f"Number of devices provided ({len(devices)}) does not match number of workers ({num_workers}). Using default device {DEVICE} for all workers.")
            devices = [DEVICE] * num_workers
        else:
            logging.info(f"Using provided devices: {devices}")

        device_queue = Queue(num_workers)
        for device in devices:
            device_queue.put(device)
                
        logging.getLogger().removeHandler(log_file_handler)
        os.environ['TQDM_DISABLE'] = '1'
        
        with Pool(num_workers, initializer=init_worker, initargs=(device_queue,)) as pool:
            iters_errors = pool.starmap(run_tests_iter, [(i, arg) for i in range(num_tests)])

        os.environ['TQDM_DISABLE'] = '0'
        logging.getLogger().addHandler(log_file_handler)

        for i, errors in enumerate(iters_errors):
            if len(errors) > 0:
                logging.error(f"Test iteration {i} encountered errors at the following test runs: {str(errors)}")

    logging.info(f"Test suite '{test_name}' completed")


if __name__ == "__main__":
    save_path = '.\stat_tests\CHECK ONLY'

    num_tests = 1
    num_workers = 1

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR10_FedAvg_Test',

        'dataset_name': 'cifar10',
        'num_clients': 5,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'fedavg',
        'train_epochs': 20,
        'local_epochs': 2,
        'participation_rate': 1.0,
        'lr': 1e-3,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': 'diag_ggn_mc'
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy'],
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
