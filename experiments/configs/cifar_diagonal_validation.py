"""Old diagonal CIFAR10 sweep, with stochastic correction off or on.

Reuse the spectral benchmark's training, recovery, repetitions and LiRA settings.
The old score uses all client records, including retained validation records.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from experiments import runner
from experiments.configs.cifar_full_validation import build_cases, build_config as spectral_config
from experiments.models import create_resnet
from fisherunlearn import compute_client_information


def build_config(stochastic_correction):
    config = {key: value for key, value in spectral_config().items()
              if not key.startswith('spectral_')}
    mode = 'on' if stochastic_correction else 'off'
    config.update(
        test_name=f'CIFAR_random_diag_ggn_mc_stochastic_{mode}',
        hessian_method='diag_ggn_mc',
        stochastic_correction=stochastic_correction,
    )
    return config


def smoke(stochastic_correction, device):
    """Exercise the actual ResNet18/BackPACK path without training or downloads."""
    runner.set_device(torch.device(device))
    runner.set_batch_sizes(2, 2, 2, 2)
    torch.manual_seed(2026)
    config = build_config(stochastic_correction)
    clients = [TensorDataset(torch.randn(4, 3, 64, 64), torch.randint(10, (4,)))
               for _ in range(2)]
    model = create_resnet(config)
    information = compute_client_information(
        0, model, nn.CrossEntropyLoss(), clients,
        method=config['hessian_method'], stochastic_correction=stochastic_correction,
        use_converter=True, learning_rate=config['learning_rate'], momentum=config['momentum'],
    )
    expected = {name: parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    assert information.keys() == expected.keys()
    assert all(value.shape == expected[name].shape and torch.isfinite(value).all()
               and (value >= 0).all() for name, value in information.items())
    assert any((value > 0).any() for value in information.values())
    print(f'PASS: ResNet18 diagonal GGN-MC, stochastic correction={stochastic_correction}, device={device}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stochastic-correction', choices=('off', 'on'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda', help='Device for the smoke check')
    args = parser.parse_args()
    correction = args.stochastic_correction == 'on'
    if args.smoke:
        smoke(correction, args.device)
        return
    if args.output is None or args.output.exists():
        parser.error('--output must name a new directory')

    np.random.seed(2026)
    torch.manual_seed(2026)
    runner.set_batch_sizes(128, 128, 128, 128)
    runner.run_repeated_tests(
        build_config(correction), build_cases(), str(args.output),
        num_workers=1, save_models=False,
    )


if __name__ == '__main__':
    main()
