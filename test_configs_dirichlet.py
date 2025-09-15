import numpy as np
from tester import run_repeated_tests, InitParamsDict, TestParamsDict

def generate_params_ranges(test_params_dict):
    # Tests with info percentage based resetting
    test_params_dict_0 = test_params_dict.copy()
    test_params_dict_0['subtest'] = 0
    test_params_dict_0['unlearning_method'] = 'information'

    percentages = np.arange(5, 95, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    # Tests with parameters percentage based resetting
    test_params_dict_1 = test_params_dict.copy()
    test_params_dict_1['subtest'] = 1
    test_params_dict_1['unlearning_method'] = 'parameters'

    percentages = np.arange(5, 55, 5)
    test_params_dicts_1 = [test_params_dict_1.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_1[i]['unlearning_percentage'] = percentage

    # Complete reset for benchmark
    test_params_dict_2 = test_params_dict.copy()
    test_params_dict_2['subtest'] = 2
    test_params_dict_2['unlearning_method'] = 'parameters'

    percentages = [100]
    test_params_dicts_2 = [test_params_dict_2.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_2[i]['unlearning_percentage'] = percentage


    return test_params_dicts_0 + test_params_dicts_2

if __name__ == "__main__":

    save_path = './stat_tests/MC'

    num_tests = 10
    num_workers = 5
    hessian_method = 'diag_ggn_mc'

    # MNIST dirichlet

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_dirichlet_PAPER',

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,

        'distribution_type': 'dirichlet',
        'dirichlet_alpha': 0.8,

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 5,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR10 dirichlet

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR_dirichlet_PAPER',

        'dataset_name': 'cifar10',
        'num_clients': 5,
        'num_classes': 10,

        'distribution_type': 'dirichlet',
        'dirichlet_alpha': 0.8,

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }


    test_params_dicts = generate_params_ranges(test_params_dict)
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # FMNIST dirichlet

    init_params_dict : InitParamsDict = {
        'test_name': 'FMNIST_dirichlet_PAPER',

        'dataset_name': 'FashionMNIST',
        'num_clients': 5,
        'num_classes': 10,

        'distribution_type': 'dirichlet',
        'dirichlet_alpha': 0.8,

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)
