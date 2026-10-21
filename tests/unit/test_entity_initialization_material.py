"""Initialization export must never masquerade as old-objective resume."""
import hashlib
import json

import pytest
import torch

from tools.checkpoints.replace_combat import export_entity_material


def test_components_preserve_actor_and_value_without_optimizer_or_alias(entity_material_source):
    source, cards, profile, out, state, _ = entity_material_source
    before = source.read_bytes()
    manifest = export_entity_material(source, cards, profile, out)
    upper = torch.load(out / 'upper.pt', weights_only=True)
    lower = torch.load(out / 'lower-init.pt', weights_only=True)
    for name, tensor in upper.items():
        assert torch.equal(tensor, state[name])
        assert not tensor.requires_grad
    for name, tensor in lower.items():
        assert torch.equal(tensor, state[name])
        assert not name.startswith(('core.combat.distribution.', 'core.combat.transition.'))
    lower['core.combat.card.vectors'].add_(10)
    assert torch.equal(upper['core.upper_tokenizer.card.vectors'], state['core.upper_tokenizer.card.vectors'])
    assert source.read_bytes() == before
    assert manifest['new_global_step'] == 0
    assert manifest['launch_ready'] is manifest['optimizer_imported'] is manifest['replay_imported'] is False
    assert set(p.name for p in out.iterdir()) == {
        'upper.pt', 'lower-init.pt', 'frozen-card-vectors.pt', 'constructor.json', 'manifest.json'}
    for name, row in manifest['files'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == row['sha256']
    constructor = json.loads((out / 'constructor.json').read_text())
    assert constructor['observation_space'] == {'fixture': 'original-space'}
    assert constructor['policy_kwargs']['entity_config']['card_vector_checkpoint'] == 'frozen-card-vectors.pt'


@pytest.mark.parametrize('field', ['model_sha256', 'card_vectors_sha256'])
def test_wrong_identity_fails_before_deserializing(entity_material_source, monkeypatch, field):
    source, cards, profile, out, _, plan = entity_material_source
    plan['source'][field] = '0' * 64
    profile.write_text(json.dumps(plan))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('deserialized unverified source'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        export_entity_material(source, cards, profile, out)
    assert not out.exists()


@pytest.mark.parametrize('change', ['source_step', 'component_count', 'launch_ready'])
def test_wrong_contract_is_not_exported(entity_material_source, change):
    source, cards, profile, out, _, plan = entity_material_source
    if change == 'source_step':
        plan['source_checkpoint_contract']['num_timesteps'] += 1
    elif change == 'component_count':
        plan['source_components']['upper']['tensors'] += 1
    else:
        plan['launch_ready'] = True
    profile.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        export_entity_material(source, cards, profile, out)
    assert not out.exists()


def test_existing_output_is_preserved(entity_material_source):
    source, cards, profile, out, _, _ = entity_material_source
    out.mkdir()
    (out / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        export_entity_material(source, cards, profile, out)
    assert (out / 'sentinel').read_text() == 'keep'


def test_partial_write_never_publishes_manifest(entity_material_source, monkeypatch):
    source, cards, profile, out, _, _ = entity_material_source

    def disk_failure(*args, **kwargs):
        raise OSError('fixture disk failure')

    monkeypatch.setattr(torch, 'save', disk_failure)
    with pytest.raises(OSError, match='disk failure'):
        export_entity_material(source, cards, profile, out)
    assert not out.exists()
    assert not list(out.parent.glob('.entity-material-*'))
