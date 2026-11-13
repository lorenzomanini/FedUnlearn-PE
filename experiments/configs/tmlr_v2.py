"""TMLR v2 suites: the experiments the v1 results showed to be missing.

Reuses the v1 helpers so splits and seeds match the v1 suites. Run exactly like v1:
  python -m experiments.configs.tmlr_v2 --list
  python -m experiments.configs.tmlr_v2 --suite NAME --output stat_tests/TMLR_v2

E1  one class per client (CIFAR-10, ten clients, three target classes): class-level
    deletion without a shared-class slice, the user-requested main setting.
E2  rerun of preferential strength 0.75 with drop_last (v1 originals did not
    interpolate because of a 2-record final batch).
S1  FedAvg-trained originals on the preferential split, to test the stationarity
    assumption when clients take local steps.
Paired arms (diagonal, leverage, ratio, contrast, output-row reset, fine-tuning) for
every v1 and v2 suite come from experiments/rescore_saved.py on checkpoints
explicitly kept with --save-models or --checkpoint-dir. Remove temporary checkpoint
directories after rescoring; the Python launcher does not clean them up.
"""
from experiments.configs import tmlr_paper as v1

SUITES = {}


def add(name, dataset_name, partition, method, cases, **overrides):
    estimator = {'core': v1.CORE, 'generalized': v1.GENERALIZED, 'diagonal': v1.DIAGONAL}[method]
    config = {**v1.base_config(dataset_name), **partition, **estimator, 'drop_last': True,
              **overrides, 'test_name': name}
    SUITES[name] = {'method': method, 'config': config, 'cases': cases}


def categorical():
    return {'distribution_type': 'categorical', 'num_clients': 10}


# E1. One class per client; three targets give the client-to-client spread.
for target in (5, 3, 9):
    add(f'c10_cat_t{target}_core', 'cifar10', categorical(), 'core', v1.main_cases(), target_client=target)
add('fm_cat_t5_core', 'FashionMNIST', categorical(), 'core', v1.main_cases(), target_client=5)

# E2. Strength 0.75 with a complete final batch.
add('c10_mix075_core_droplast', 'cifar10', v1.mix(0.75), 'core', v1.main_cases())

# S1. Local-update training (FedAvg, 1 local epoch, AdamW as implemented in
#     experiments.training.fedavg_trainer); 40 rounds to match 40 epochs.
add('c10_mix100_core_fedavg', 'cifar10', v1.mix(1.0), 'core', v1.ablation_cases(),
    trainer_name='fedavg', local_epochs=1, participation_rate=1.0, lr=1e-3)

if __name__ == '__main__':
    # Reuse the v1 command-line interface with this module's suite table.
    v1.SUITES = SUITES
    v1.main()
