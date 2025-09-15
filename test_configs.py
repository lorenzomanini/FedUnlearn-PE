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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
            'retrain_epochs': 1
        }
    
    test_params_dicts = generate_params_ranges(test_params_dict)

    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    #run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'num_tests': num_tests,
        'hessian_method': hessian_method
    }

    test_params_dict : TestParamsDict = {
            'tests': ['test_accuracy', 'clients_accuracies', 'class_accuracies'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)


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
        'train_epochs': 5,

        'target_client': 0,
        'num_tests': num_tests,
        
        'poison' : True,
        'target_label': 9,
        'hessian_method': hessian_method
    }

    test_params_dict: TestParamsDict = {
            'tests': ['attack_success_rate', 'unlearning_accuracy'], 
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    test_params_dicts = generate_params_ranges(test_params_dict)

    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=num_workers)