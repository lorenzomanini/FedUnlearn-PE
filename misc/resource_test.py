# %%
import numpy as np
import tester
from tester import run_repeated_tests, InitParamsDict, TestParamsDict
import os

# %%
import time
import threading
import psutil
import numpy as np

if __name__ == "__main__":
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        GPU_AVAILABLE = True
    except:
        GPU_AVAILABLE = False

    class ResourceTracker:
        def __init__(self, interval=0.1):
            self.interval = interval
            self._running = False
            self._samples = []

        def _sample(self):
            process = psutil.Process()
            while self._running:
                cpu = psutil.cpu_percent(interval=None)
                ram = process.memory_info().rss / 1e6  # MB

                gpu, vram = (0.0, 0.0)
                if GPU_AVAILABLE:
                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    gpu = util.gpu
                    vram = mem.used / 1e6  # MB

                self._samples.append((cpu, ram, gpu, vram))
                time.sleep(self.interval)

        def start(self):
            self._samples.clear()
            self._running = True
            self._start_time = time.time()
            self._thread = threading.Thread(target=self._sample)
            self._thread.start()

        def stop(self):
            self._running = False
            self._thread.join()
            duration = time.time() - self._start_time

            if self._samples:
                arr = np.array(self._samples)
                avg_cpu, avg_ram, avg_gpu, avg_vram = arr.mean(axis=0)
            else:
                avg_cpu = avg_ram = avg_gpu = avg_vram = 0.0

            return {
                'avg_cpu_percent': avg_cpu,
                'avg_ram_mb': avg_ram,
                'avg_gpu_percent': avg_gpu,
                'avg_vram_mb': avg_vram,
                'duration_sec': duration
            }

    # %%
    save_path = 'resource_tests'
    stats_path = os.path.join(save_path, 'stats.txt')
    #tester.set_device('cuda:0')

    # %%
    init_params_dict : InitParamsDict = {
        'test_name': 'resource_test_mnist_sp', # Changed name slightly

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'random',     # Distribution type

        'model_name': 'simple_cnn',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 3,                # Initial training epochs

        'target_client': 0,               # Client to unlearn
        'num_tests': 1                   # Number of independent repetitions
    }

    test_params_dict_0 : TestParamsDict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    percentages = np.arange(5, 20, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    tracker = ResourceTracker()
    tracker.start()
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1)
    stats=tracker.stop()

    with open(stats_path, "a") as f:
        f.write(f"\n=== Run {init_params_dict['test_name']} ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"\n=== Run {init_params_dict['test_name']} ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # %%
    init_params_dict : InitParamsDict = {
        'test_name': 'resource_test_mnist_mp', # Changed name slightly

        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'random',     # Distribution type

        'model_name': 'simple_cnn',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 3,                # Initial training epochs

        'target_client': 0,               # Client to unlearn
        'num_tests': 10                   # Number of independent repetitions
    }

    test_params_dict_0 : TestParamsDict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }

    percentages = np.arange(5, 20, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    tracker = ResourceTracker()
    tracker.start()
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=10)
    stats=tracker.stop()

    with open(stats_path, "a") as f:
        f.write(f"\n=== Run {init_params_dict['test_name']} ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"\n=== Run {init_params_dict['test_name']} ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # %%
    init_params_dict : InitParamsDict = {
        'test_name': 'resource_test_cifar_sp', # Changed name slightly

        'dataset_name': 'cifar10',
        'num_clients': 5,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'random',     # Distribution type

        'model_name': 'resnet18',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 3,                # Initial training epochs

        'target_client': 0,               # Client to unlearn
        'num_tests': 1                   # Number of independent repetitions
    }

    test_params_dict_0 : TestParamsDict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }


    percentages = np.arange(5, 20, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    tracker = ResourceTracker()
    tracker.start()
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=1)
    stats=tracker.stop()

    with open(stats_path, "a") as f:
        f.write(f"\n=== Run {init_params_dict['test_name']} ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"\n=== Run {init_params_dict['test_name']} ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # %%
    init_params_dict : InitParamsDict = {
        'test_name': 'resource_test_cifar_mp', # Changed name slightly

        'dataset_name': 'cifar10',
        'num_clients': 5,
        'num_classes': 10,                # Number of classes in the dataset
        'distribution_type': 'random',     # Distribution type

        'model_name': 'resnet18',       # Model architecture
        'loss_name': 'cross_entropy',     # Loss function

        'trainer_name': 'sgd',            # Trainer type
        'train_epochs': 3,                # Initial training epochs

        'target_client': 0,               # Client to unlearn
        'num_tests': 10                   # Number of independent repetitions
    }

    test_params_dict_0 : TestParamsDict = {
            'subtest': 0,
            'unlearning_method': 'information',
            'tests': ['test_accuracy', 'clients_accuracies', 'mia'],
            'mia_classifier_types': ['nn', 'logistic'],
            'retrain_epochs': 1
        }


    percentages = np.arange(5, 20, 5)
    test_params_dicts_0 = [test_params_dict_0.copy() for _ in range(len(percentages))]
    for i, percentage in enumerate(percentages):
        test_params_dicts_0[i]['unlearning_percentage'] = percentage

    test_params_dicts = test_params_dicts_0

    tracker = ResourceTracker()
    tracker.start()
    run_repeated_tests(init_params_dict, test_params_dicts, save_path, num_workers=10)
    stats=tracker.stop()

    with open(stats_path, "a") as f:
        f.write(f"\n=== Run {init_params_dict['test_name']} ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"\n=== Run {init_params_dict['test_name']} ===")
    for key, value in stats.items():
        print(f"{key}: {value}")



