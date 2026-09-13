from pathlib import Path

import pytest

from rl.training.distributed_sb3 import Peer, ssh_control_directory, ssh_options


def test_long_run_names_fit_openssh_socket_with_temporary_suffix():
    root = Path('/tmp/sts_dist_ssh_mux')
    run_id = 'nosl-rootpaper-graph-' + 'experiment-' * 30
    directory = ssh_control_directory(root, run_id, 3)
    options = ssh_options(Peer(0, 'localhost', 22), Path('key'), directory)
    template = next(s.removeprefix('ControlPath=') for s in options if s.startswith('ControlPath='))
    actual = template.replace('%C', 'a' * 40) + '.' + 'b' * 16
    assert len(actual.encode()) < 104
    assert directory == ssh_control_directory(root, run_id, 3)
    assert directory != ssh_control_directory(root, run_id, 2)
    assert directory != ssh_control_directory(root, run_id + '-other', 3)


def test_too_long_control_root_fails_before_invoking_ssh():
    with pytest.raises(ValueError, match='shorter --ssh-control-dir'):
        ssh_options(Peer(0, 'localhost', 22), Path('key'), Path('/tmp/' + 'x' * 100))
