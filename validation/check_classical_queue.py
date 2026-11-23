"""CPU-only checks: configuration, real partitioner, and mocked Slurm dispatch.

Run: MPLCONFIGDIR=/tmp/fedunlearn-mpl .venv/bin/python validation/check_classical_queue.py
No downloads, training or scheduler submission.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.configs.classical_unlearning import SUITES
from experiments.datasets import get_clients_subsets
from validation.accuracy_audit.analyze_saved_cifar import partition_groups


class LabelsOnly:
    targets = np.repeat(np.arange(10), 5000)

    def __len__(self):
        return len(self.targets)


names = list(SUITES)
assert names == [f'c10_{setting}_{method}' for setting in ('class5', 'random', 'dir080')
                 for method in ('core', 'diagonal')]
partition_counts = {}
for name, suite in SUITES.items():
    config, cases = suite['config'], suite['cases']
    assert config['num_clients'] == 10 and config['num_tests'] == 12
    assert config['repetition_seed'] == 4000 and config['num_shadow_models'] == 64
    assert config['trainer_name'] == 'sgd' and config['drop_last']
    assert len(cases) == 16 and all(case['tests'] == ['LiRA'] for case in cases)
    assert cases[0]['unlearning_percentage'] == cases[-1]['unlearning_percentage'] == 0
    assert cases[-1]['recovery_scope'] == 'all'
    if suite['method'] == 'core':
        assert config['spectral_curvature_backend'] == 'ggn'
    np.random.seed(2026)
    torch.manual_seed(2026)
    clients = get_clients_subsets(LabelsOnly(), config)
    indices = [list(client.indices) for client in clients]
    assigned = np.concatenate(indices)
    target = np.asarray(indices[config['target_client']])
    assert len(assigned) == len(np.unique(assigned)) and len(target) > 0
    if config['distribution_type'] == 'categorical':
        assert len(target) == 5000 and np.all(LabelsOnly.targets[target] == 5)
    if config['distribution_type'] in ('categorical', 'random'):
        assert len(assigned) == 50000
    else:
        assert len(assigned) < 50000
    partition_counts[name] = {'assigned': len(assigned), 'target': len(target)}
    if suite['method'] == 'diagonal':
        assert indices == core_indices, 'Core and diagonal partitions differ'
    else:
        core_indices = indices

groups = partition_groups([[0, 1], [2, 3]], np.array([0, 1, 2]), np.array([0, 1]), 6)
assert groups['assigned_retain_heldout'].tolist() == [3]
assert groups['unassigned'].tolist() == [4, 5]
assert groups['retain_heldout'].tolist() == [3, 4, 5]
try:
    partition_groups([[0, 1], [1, 2]], np.array([0, 1]), np.array([0]), 3)
except AssertionError:
    pass
else:
    raise AssertionError('Overlapping clients were accepted')

subprocess.run(['bash', '-n', 'queue.sh'], cwd=ROOT, check=True)
with tempfile.TemporaryDirectory(prefix='classical-queue-') as temporary:
    temp = Path(temporary)
    bin_dir = temp / 'venv' / 'bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'activate').write_text(':\n')
    stub = f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['QUEUE_CHECK_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if '--name' in args:
    print({names!r}[int(args[args.index('--name') + 1])])
elif '--describe' in args:
    print('{{}}')
elif 'validation/accuracy_audit/analyze_saved_cifar.py' in args:
    sys.exit(int(os.environ.get('QUEUE_CHECK_FAIL_AUDIT', '0')))
elif args == ['-']:
    sys.stdin.read()
    print({str(ROOT)!r})
'''
    (bin_dir / 'python').write_text(stub)
    (bin_dir / 'python').chmod(0o755)
    for command, content in {'module': '#!/bin/sh\nexit 0\n',
                             'srun': '#!/bin/sh\nexec "$@"\n'}.items():
        path = bin_dir / command
        path.write_text(content)
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'VENV_PATH': str(temp / 'venv'), 'SLURM_SUBMIT_DIR': str(ROOT),
           'PROJECT_DIR': str(ROOT), 'SLURM_ARRAY_JOB_ID': 'test',
           'CLASSICAL_ROOT': str(temp / 'results'), 'QUEUE_CHECK_LOG': str(temp / 'calls.jsonl')}
    for task, name in enumerate(names, 18):
        env['SLURM_ARRAY_TASK_ID'] = str(task)
        subprocess.run(['bash', 'queue.sh'], cwd=ROOT, env=env, check=True, capture_output=True)
        job_dir = temp / 'results' / f'{name}_job_test'
        assert (job_dir / 'COMPLETE').is_file()
        assert (job_dir / 'protocol.json').is_file() and (job_dir / 'source.tar.gz').is_file()
    calls = [json.loads(line) for line in (temp / 'calls.jsonl').read_text().splitlines()]
    for name in names:
        assert any('--save-models' in call and '--output' in call and name in call for call in calls)
        assert any('--smoke' in call and name in call for call in calls)
    assert sum('validation/accuracy_audit/analyze_saved_cifar.py' in call for call in calls) == 6
    assert not any('experiments.within_class_finalize' in call for call in calls)
    # Refuse collisions rather than silently overwriting completed outputs.
    collision = subprocess.run(['bash', 'queue.sh'], cwd=ROOT, env=env, capture_output=True)
    assert collision.returncode == 2 and b'already exists' in collision.stderr
    env.update(SLURM_ARRAY_TASK_ID='18', SLURM_ARRAY_JOB_ID='failed', QUEUE_CHECK_FAIL_AUDIT='1')
    failed = subprocess.run(['bash', 'queue.sh'], cwd=ROOT, env=env, capture_output=True)
    assert failed.returncode != 0
    assert not (temp / 'results' / f'{names[0]}_job_failed' / 'COMPLETE').exists()
    for task in ('24', '-1', 'oops'):
        env['SLURM_ARRAY_TASK_ID'] = task
        assert subprocess.run(['bash', 'queue.sh'], cwd=ROOT, env=env, capture_output=True).returncode == 2

print(json.dumps({'checks': 'PASS', 'partition_counts': partition_counts}, indent=2))
