import numpy as np
import os
import new_new_tester as tester
from new_new_tester import run_repeated_tests, InitParamsDict, TestParamsDict

def generate_params_ranges(test_params_dict):
    test_params_dict_0 = test_params_dict.copy()
    test_params_dict_0['subtest'] = 0
    test_params_dict_0['unlearning_method'] = 'information'

    percentages = np.linspace(0, 100, num=10).tolist()
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    return test_params_dicts_0

if __name__ == "__main__":

    save_path = f'./stat_tests/EXPERIMENTS/power_iters_{os.environ.get("NUM_POWER_ITERS", "5")}/'

    num_tests = 20
    num_workers = 1

    tester.set_batch_sizes(128, 128, 128, 128)

    # MNIST preferential — matches experiments.ipynb exactly

    init_params_dict: InitParamsDict = {
        'test_name': 'MNIST_pref',

        'dataset_name': 'mnist',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 6,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict: TestParamsDict = {
        'retrain_epochs': 1,
    }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # MNIST random

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_random',

        'dataset_name': 'mnist',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 6,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR10 random

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR_random',

        'dataset_name': 'cifar10',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR10 preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR_pref',

        'dataset_name': 'cifar10',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # FMNIST random

    init_params_dict : InitParamsDict = {
        'test_name': 'FMNIST_random',

        'dataset_name': 'FashionMNIST',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # FMNIST preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'FMNIST_pref',

        'dataset_name': 'FashionMNIST',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR100 random

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR100_random',

        'dataset_name': 'cifar100',
        'num_clients': 10,
        'num_classes': 100,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR100 preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR100_pref',

        'dataset_name': 'cifar100',
        'num_clients': 10,
        'num_classes': 100,
        'distribution_type': 'preferential_class',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # POISON ATTACK

    init_params_dict: InitParamsDict = {
        'test_name': 'POISON_PAPER',

        'dataset_name': 'cifar10',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 5,
        'learning_rate': 0.01,
        'momentum': 0.9,

        'target_client': 0,
        'num_tests': num_tests,

        'poison': True,
        'target_label': 9,
    }

    test_params_dict: TestParamsDict = {
            'tests': ['poisoned_backdoor_accuracy', 'clean_backdoor_accuracy'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)
