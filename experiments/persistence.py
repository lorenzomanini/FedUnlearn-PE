import os
import pickle

import numpy as np
import torch


INIT_PARAMS = "init_params.pkl"
TEST_PARAMS = "test_params.pkl"
CLIENT_INDICES = "clients_indices.pkl"
LABELS = "labels.pkl"
LEGACY_RESULTS = "test_results.pkl"
TRAINED_MODEL = "trained_model.pth"
BENCHMARK_MODEL = "benchmark_model.pth"
CLIENT_INFORMATION = "client_information.pkl"
INITIAL_TEST_RESULTS = "initial_eval_test_results.pkl"
INITIAL_TRAIN_RESULTS = "initial_eval_train_results.pkl"
EVAL_TEST_RESULTS = "eval_test_results.npz"
EVAL_TRAIN_RESULTS = "eval_train_results.npz"
EXTRA_RESULTS = "extra_results.pkl"
LIRA_SHADOW_BANK = "lira_shadow_bank.npz"
INITIAL_LIRA_RESULTS = "initial_lira_results.pkl"
EVAL_LIRA_RESULTS = "eval_lira_results.npz"


def dump_pickle(directory, basename, value):
    with open(os.path.join(directory, basename), "wb") as stream:
        pickle.dump(value, stream)


def load_pickle(directory, basename):
    with open(os.path.join(directory, basename), "rb") as stream:
        return pickle.load(stream)


def dump_npz(directory, basename, values):
    with open(os.path.join(directory, basename), "wb") as stream:
        np.savez(stream, **values)


def load_npz(directory, basename):
    with open(os.path.join(directory, basename), "rb") as stream:
        return dict(np.load(stream))


def pack_revised_results(results):
    packed = {}
    for key in results[0].keys():
        packed[f"{key}__pred"] = np.stack([result[key]["pred"] for result in results])
        packed[f"{key}__loss"] = np.stack([result[key]["loss"] for result in results])
    return packed


def pack_result_lists(results):
    return {
        key: [result[key] for result in results]
        for key in results[0].keys()
    }


def pack_score_results(results):
    return {
        key: np.stack([result[key] for result in results])
        for key in results[0]
    }


def save_legacy_models(directory, test_instance):
    torch.save(test_instance.trained_model.cpu().state_dict(), os.path.join(directory, TRAINED_MODEL))
    torch.save(test_instance.benchmark_model.cpu().state_dict(), os.path.join(directory, BENCHMARK_MODEL))
    dump_pickle(directory, CLIENT_INFORMATION, test_instance.client_information)
