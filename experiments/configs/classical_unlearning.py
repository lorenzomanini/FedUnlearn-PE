"""Focused classical forgetting batch; see analysis/tmlr_review_2026-11-23/classical_protocol.md.

Uses the existing runner and CLI: --list, --download, --suite, --describe,
--smoke, --output and --save-models. These are new experiments, not reproductions
of the historical paper tables or additions to the CIFAR-100 primary analysis.
"""
from experiments.configs import tmlr_paper as common


SUITES = {}
for setting, partition in (
    ('class5', {'distribution_type': 'categorical', 'target_client': 5}),
    ('random', {'distribution_type': 'random', 'target_client': 0}),
    ('dir080', {'distribution_type': 'dirichlet', 'dirichlet_alpha': 0.8, 'target_client': 0}),
):
    for method in ('core', 'diagonal'):
        name = f'c10_{setting}_{method}'
        estimator = ({**common.CORE, 'spectral_curvature_backend': 'ggn'}
                     if method == 'core' else common.DIAGONAL)
        SUITES[name] = {
            'method': method,
            'config': {
                **common.base_config('cifar10'), **partition, **estimator,
                'test_name': name, 'num_clients': 10, 'drop_last': True,
                'num_tests': 12, 'repetition_seed': 4000,
            },
            # Each case also runs a layer-count-matched random reset and recovery.
            'cases': [common.case(0)] + [common.case(mass) for mass in common.MASSES]
                     + [common.case(0, scope='all')],
        }


if __name__ == '__main__':
    common.SUITES = SUITES
    common.main()
