"""Dependency discovery may skip inaccessible history, never explicit overrides."""
import errno
from pathlib import Path
from zipfile import ZipFile

import pytest
from stable_baselines3.common.save_util import data_to_json

from rl.policies.checkpoint import custom_objects
from rl.policies import checkpoint as checkpoint_module


@pytest.fixture
def inaccessible_encoder(tmp_path, monkeypatch):
    blocked = tmp_path / 'private' / 'encoder-run' / 'last.pt'
    blocked.parent.mkdir(parents=True)
    blocked.touch()
    checkpoint = tmp_path / 'model.zip'
    with ZipFile(checkpoint, 'w') as archive:
        archive.writestr('data', data_to_json({'policy_kwargs': {
            'card_vector_checkpoint': str(blocked)}}))
    original_stat = Path.stat

    def denied_stat(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError(errno.EACCES, 'Permission denied', str(path))
        return original_stat(path, *args, **kwargs)

    # A permissions-based fixture cannot rely on chmod when CI runs as root or
    # on Windows. Inject the exact pathlib failure reported by the Linux job.
    monkeypatch.setattr(Path, 'stat', denied_stat)
    monkeypatch.delenv('STS_CARD_VECTOR_CHECKPOINT', raising=False)
    monkeypatch.setattr(checkpoint_module, '__file__',
                        str(tmp_path / 'isolated-repo/rl/policies/checkpoint.py'))
    return checkpoint, blocked


def test_inaccessible_saved_path_falls_back_to_checkpoint_sibling(inaccessible_encoder):
    checkpoint, _ = inaccessible_encoder
    encoder = checkpoint.parent / 'encoder-run' / 'last.pt'
    encoder.parent.mkdir()
    encoder.touch()
    assert custom_objects(checkpoint)['policy_kwargs']['card_vector_checkpoint'] == str(encoder.resolve())


def test_inaccessible_history_without_replacement_reports_missing_dependency(inaccessible_encoder):
    checkpoint, _ = inaccessible_encoder
    with pytest.raises(FileNotFoundError, match='missing card-vector encoder'):
        custom_objects(checkpoint)


@pytest.mark.parametrize('source', ['argument', 'environment'])
def test_explicit_inaccessible_encoder_never_silently_falls_back(inaccessible_encoder, monkeypatch, source):
    checkpoint, blocked = inaccessible_encoder
    replacement = checkpoint.parent / 'encoder-run' / 'last.pt'
    replacement.parent.mkdir()
    replacement.touch()
    kwargs = {}
    if source == 'environment':
        monkeypatch.setenv('STS_CARD_VECTOR_CHECKPOINT', str(blocked))
    else:
        kwargs['card_vector_checkpoint'] = blocked
    with pytest.raises(PermissionError):
        custom_objects(checkpoint, **kwargs)
