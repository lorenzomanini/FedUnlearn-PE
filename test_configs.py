import numpy as np
from tester import run_repeated_tests, InitParamsDict, TestParamsDict

save_path = '.\stat_tests'

num_tests = 20

# MNIST random

init_params_dict : InitParamsDict = {
    'test_name': 'MNIST_random_PAPER',

    'dataset_name': 'mnist',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'random',

    'model_name': 'simple_cnn',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 5,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    } 
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# MNIST preferential

init_params_dict : InitParamsDict = {
    'test_name': 'MNIST_pref_PAPER',

    'dataset_name': 'mnist',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'preferential_class',

    'model_name': 'simple_cnn',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 5,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    } 
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# CIFAR10 random

init_params_dict : InitParamsDict = {
    'test_name': 'CIFAR_random_PAPER',

    'dataset_name': 'cifar10',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'random',

    'model_name': 'resnet18',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 40,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    } 
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# CIFAR10 preferential

init_params_dict : InitParamsDict = {
    'test_name': 'CIFAR_pref_PAPER',

    'dataset_name': 'cifar10',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'preferential_class',

    'model_name': 'resnet18',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 40,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    } 
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# FMNIST random

init_params_dict : InitParamsDict = {
    'test_name': 'FMNIST_random_PAPER',

    'dataset_name': 'FashionMNIST',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'random',

    'model_name': 'resnet18',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 40,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    } 
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# FMNIST preferential

init_params_dict : InitParamsDict = {
    'test_name': 'FMNIST_pref_PAPER',

    'dataset_name': 'FashionMNIST',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'preferential_class',

    'model_name': 'resnet18',
    'loss_name': 'cross_entropy',

    'trainer_name': 'sgd',
    'train_epochs': 40,

    'target_client': 0,
    'num_tests': num_tests
}

test_params_dict_0 : TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage


test_params_dict_1 : TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, workers=8)


# POISON ATTACK

init_params_dict: InitParamsDict = {
    'test_name': 'POISON_PAPER',
    'dataset_name': 'mnist',
    'num_clients': 5,
    'num_classes': 10,
    'distribution_type': 'random',
    'model_name': 'simple_cnn',
    'loss_name': 'cross_entropy',
    'trainer_name': 'sgd',
    'train_epochs': 4,
    'target_client': 0,
    'num_tests': num_tests,
    'poison' : True,
    'target_label': 9 
}

test_params_dict_0: TestParamsDict = {
        'subtest': 0,
        'unlearning_method': 'information',
        'tests': ['attack_success_rate', 'unlearning_accuracy'], 
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 90, 5)
test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_0[i]['unlearning_percentage'] = percentage

test_params_dict_1: TestParamsDict = {
        'subtest': 1,
        'unlearning_method': 'parameters',
        'tests': ['attack_success_rate', 'unlearning_accuracy'],
        'mia_classifier_types': ['nn', 'logistic'],
        'retrain_epochs': 1
    }
percentages = np.arange(5, 50, 5)
test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
for i, percentage in enumerate(percentages):
    test_params_dicts_1[i]['unlearning_percentage'] = percentage


test_params_dicts = test_params_dicts_0 + test_params_dicts_1

run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=8)