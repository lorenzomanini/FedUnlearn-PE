import numpy as np
import new_tester as tester
from new_tester import run_repeated_tests, InitParamsDict, TestParamsDict

def generate_params_ranges(test_params_dict):
    # Tests with info percentage based resetting
    test_params_dict_0 = test_params_dict.copy()
    test_params_dict_0['subtest'] = 0
    test_params_dict_0['unlearning_method'] = 'information'

    percentages = [0,40,50,60]
    # percentages = np.arange(0, 91, 10)
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

    percentages = [10]
    test_params_dicts_2 = [test_params_dict_2.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_2[i]['unlearning_percentage'] = percentage


    return test_params_dicts_0

if __name__ == "__main__":

    save_path = './stat_tests/NEW_TESTER'

    num_tests = 3
    num_workers = 1
    hessian_method = 'diag_ggn_mc'

    tester.set_batch_sizes(128,128,128,128)

    # MNIST random

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_random',

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 5,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method,
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # MNIST preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'MNIST_pref',

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'simple_cnn',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 5,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method,
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'retrain_epochs': 1
        }
    
    test_params_dicts = generate_params_ranges(test_params_dict)

    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }


    test_params_dicts = generate_params_ranges(test_params_dict)
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
            'retrain_epochs': 1
        }
    
    test_params_dicts = generate_params_ranges(test_params_dict)
    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # FMNIST random

    init_params_dict : InitParamsDict = {
        'test_name': 'FMNIST_random_STOCHASTIC',

        'dataset_name': 'FashionMNIST',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method,
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # FMNIST preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'FMNIST_pref_STOCHASTIC',

        'dataset_name': 'FashionMNIST',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'preferential_class',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method,
        'stochastic_correction': True
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR100 random

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR100_random_PAPER',

        'dataset_name': 'cifar100',
        'num_clients': 5,
        'num_classes': 100,
        'distribution_type': 'random',

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
    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # CIFAR100 preferential

    init_params_dict : InitParamsDict = {
        'test_name': 'CIFAR100_pref_PAPER',

        'dataset_name': 'cifar100',
        'num_clients': 5,
        'num_classes': 100,
        'distribution_type': 'preferential_class',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 40,

        'target_client': 0,
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)
    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


    # POISON ATTACK

    init_params_dict: InitParamsDict = {
        'test_name': 'POISON_PAPER',

        'dataset_name': 'cifar10',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'random',

        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',

        'trainer_name': 'sgd',
        'train_epochs': 5,

        'target_client': 0,
        'num_tests': num_tests,
        
        'poison' : True,
        'target_label': 9,
        'hessian_method': hessian_method
    }

    test_params_dict: TestParamsDict = {
            'tests': ['poisoned_backdoor_accuracy', 'clean_backdoor_accuracy'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    # run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)