"""Replace split-policy combat, or export entity-joint initialization material.

The entity-material mode exports tensors only, not a runnable training checkpoint.
Neither mode resumes the source optimizer, replay or progress.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import torch


def replace_with_mlp(model, *, seed=0):
    from rl.policies.split_policy import SplitStSPointerPolicy

    if type(model.policy) is not SplitStSPointerPolicy:
        raise ValueError("combat replacement requires SplitStSPointerPolicy")
    if not (model.nosl_is_az and model.nosl_is_az_inner_only):
        raise ValueError("combat replacement requires inner-only NOSL")
    if model.policy_kwargs.get("combat_arch", "transformer") != "transformer":
        raise ValueError("source combat architecture must be transformer")
    if model.device.type != "cpu":
        raise ValueError("construct the replacement checkpoint on CPU")
    frozen = {key: value for key, value in model.policy.state_dict().items()
              if not key.startswith("core.combat.")}
    kwargs = dict(model.policy_kwargs, combat_arch="mlp")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = SplitStSPointerPolicy(
            model.observation_space, model.action_space, model.lr_schedule, **kwargs,
        )
    missing, unexpected = policy.load_state_dict(frozen, strict=False)
    if unexpected or any(not key.startswith("core.combat.") for key in missing):
        raise ValueError(f"frozen policy contract changed: missing={missing}, unexpected={unexpected}")
    assert all(torch.equal(policy.state_dict()[key], value) for key, value in frozen.items())
    model.policy = policy
    model.policy_kwargs = kwargs
    model.seed = seed
    model.num_timesteps = model._n_updates = model._total_timesteps = 0
    model._num_timesteps_at_start = 0
    model._last_obs = model._last_episode_starts = None
    model._nosl_is_az_replay = None
    model._nosl_is_az_pending_fights = []
    model._nosl_is_az_samples = []
    model._nosl_is_az_fresh_samples = []
    model.configure_nosl_is_az_inner_only()
    return model


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _tensor_manifest(state):
    return {key: {
        'shape': list(value.shape), 'dtype': str(value.dtype),
        'numel': value.numel(),
        'sha256': hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    } for key, value in sorted(state.items())}


def export_entity_material(source, card_vectors, profile_path, out):
    """Export independent CPU tensors; validate identity before deserializing weights.

    The constructor's SB3 space records are preserved as JSON, never unpickled
    here. A future loader must check the manifest before using those records.
    Runtime freezing, scalar value construction and new-objective training are
    deliberately not represented by a legacy model.zip.
    """
    source, card_vectors, profile_path, out = map(Path, (source, card_vectors, profile_path, out))
    if out.exists():
        raise FileExistsError(out)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if profile['kind'] != 'combat01-preparation-v1' or profile['launch_ready'] is not False:
        raise ValueError('expected a COMBAT01 preparation profile, not a launch authorization')
    for label, path in [('model', source), ('card_vectors', card_vectors)]:
        if _sha(path) != profile['source'][label + '_sha256']:
            raise ValueError(f'{label} SHA256 mismatch')
    with ZipFile(source) as archive:
        metadata = json.loads(archive.read('data'))
        for key, expected in profile['source_checkpoint_contract'].items():
            if metadata.get(key) != expected:
                raise ValueError(f'source checkpoint contract mismatch: {key}')
        if metadata['policy_class'].get('__module__') != 'rl.policies.entity_joint':
            raise ValueError('entity-material requires an entity-joint checkpoint')
        state = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    if any(not key.startswith(('core.upper_tokenizer.', 'core.overworld.', 'core.combat.')) for key in state):
        raise ValueError('unrecognized policy tensor namespace')
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('policy must contain finite tensors only')
    upper = {key: value.detach().clone() for key, value in state.items()
             if key.startswith(('core.upper_tokenizer.', 'core.overworld.'))}
    dropped = {key: value for key, value in state.items()
               if key.startswith(('core.combat.distribution.', 'core.combat.transition.'))}
    lower = {key: value.detach().clone() for key, value in state.items()
             if key.startswith('core.combat.') and key not in dropped}
    required = ('core.upper_tokenizer.card.vectors', 'core.combat.card.vectors')
    if (not upper or not lower or not dropped or any(key not in state for key in required)
            or not any(key.startswith('core.overworld.') for key in upper)):
        raise ValueError('missing entity-joint component or legacy value head')
    if not torch.equal(state[required[0]], state[required[1]]):
        raise ValueError('upper/lower card tables disagree')
    component_manifest = {name: _tensor_manifest(values) for name, values in
                          [('upper', upper), ('lower_init', lower), ('discarded', dropped)]}
    for name, expected in profile['source_components'].items():
        rows = component_manifest[name]
        if len(rows) != expected['tensors'] or sum(r['numel'] for r in rows.values()) != expected['elements']:
            raise ValueError(f'component schema mismatch: {name}')
    constructor = {
        'format': 'sb3-source-spaces-v1',
        'observation_space': metadata['observation_space'],
        'action_space': metadata['action_space'],
        'policy_kwargs': metadata['policy_kwargs'],
    }
    # The copied source config describes the old architecture, not the new loss.
    constructor['policy_kwargs']['entity_config']['card_vector_checkpoint'] = 'frozen-card-vectors.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.entity-material-', dir=out.parent)).resolve()
    try:
        torch.save(upper, staging / 'upper.pt')
        torch.save(lower, staging / 'lower-init.pt')
        shutil.copyfile(card_vectors, staging / 'frozen-card-vectors.pt')
        (staging / 'constructor.json').write_text(json.dumps(constructor, indent=2) + '\n', encoding='utf-8')
        manifest = {
            'schema': 'entity-joint-initialization-material-v1', 'launch_ready': False,
            'source_model_sha256': _sha(source), 'source_step': metadata['num_timesteps'],
            'profile_sha256': _sha(profile_path), 'simulator': profile['simulator'],
            'purpose': 'upper actor/critic extraction and lower actor/encoder warm start; no runtime freeze claim',
            'optimizer_imported': False, 'replay_imported': False, 'new_global_step': 0,
            'lower_value_head': 'absent: construct new FP32 scalar outer-return head before training',
            'files': {p.name: {'sha256': _sha(p), 'bytes': p.stat().st_size}
                      for p in sorted(staging.iterdir())},
            'components': component_manifest,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if out.exists():
            raise FileExistsError(out)
        os.rename(staging, out)
    finally:
        # Only remove the private, newly-created staging directory on failure.
        if staging.exists() and staging.parent == out.parent.resolve() and staging.name.startswith('.entity-material-'):
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("mlp", "entity-material"), default="mlp")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--card-vectors", type=Path)
    args = parser.parse_args()
    if args.mode == "entity-material":
        if args.profile is None or args.card_vectors is None:
            parser.error("entity-material requires --profile and --card-vectors")
        torch.set_num_threads(2)
        manifest = export_entity_material(args.source, args.card_vectors, args.profile, args.out)
        print(f"exported={args.out} launch_ready={manifest['launch_ready']} optimizer/replay=not-imported")
        return
    if args.profile is not None or args.card_vectors is not None:
        parser.error("--profile and --card-vectors require --mode entity-material")
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.policies.checkpoint import custom_objects

    if args.out.suffix != ".zip":
        args.out = Path(str(args.out) + ".zip")
    if args.out.exists():
        raise FileExistsError(args.out)
    model = AccumulatingMaskablePPO.load(
        args.source, device="cpu", custom_objects=custom_objects(args.source),
    )
    old_count = sum(parameter.numel() for parameter in model.policy.core.combat.parameters())
    replace_with_mlp(model, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    new_count = sum(parameter.numel() for parameter in model.policy.core.combat.parameters())
    print(f"saved={args.out} seed={args.seed} step=0 replay=empty Adam=fresh")
    print(f"combat_parameters={old_count}->{new_count} input_width={model.policy.core.combat.input_width}")


if __name__ == "__main__":
    main()
