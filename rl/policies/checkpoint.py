"""Load SB3 checkpoints with relocated encoder dependencies resolved first."""
import os
from pathlib import Path
from zipfile import ZipFile

from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import json_to_data


def custom_objects(checkpoint, card_vector_checkpoint=None):
    checkpoint = Path(checkpoint)
    # SB3 accepts extensionless checkpoint paths and appends .zip on load.
    # The metadata preflight must resolve the same file as the actual loader.
    if not checkpoint.exists() and Path(str(checkpoint) + '.zip').is_file():
        checkpoint = Path(str(checkpoint) + '.zip')
    # Inspect metadata only: do not deserialize all weight/optimizer tensors
    # once here and then a second time inside MaskablePPO.load.
    with ZipFile(checkpoint) as archive:
        data = json_to_data(archive.read('data').decode('utf-8'))
    policy_kwargs = dict(data.get('policy_kwargs') or {})
    entity_config = dict(policy_kwargs.get('entity_config') or {})
    saved = (entity_config.get('card_vector_checkpoint')
             or policy_kwargs.get('card_vector_checkpoint'))
    if not saved:
        return {}
    explicit = card_vector_checkpoint or os.environ.get('STS_CARD_VECTOR_CHECKPOINT')
    if explicit:
        resolved = Path(explicit).expanduser().resolve(strict=True)
    else:
        parts = str(saved).replace('\\', '/').split('/')
        candidates = [Path(saved).expanduser()]
        for parent in (checkpoint.parent, Path(__file__).resolve().parents[2]):
            if len(parts) >= 2:
                candidates.append(parent / parts[-2] / parts[-1])
            candidates.append(parent / '.tmp-vector-rule-v7-contract-v10-step10000/last.pt')
        resolved = None
        for candidate in candidates:
            try:
                if candidate.is_file():
                    resolved = candidate.resolve()
                    break
            except PermissionError:
                # A checkpoint may record another machine's private /root
                # path. Discovery must still try relocated local candidates.
                # Explicit overrides above deliberately keep their errors.
                continue
        if resolved is None:
            raise FileNotFoundError(f'missing card-vector encoder {saved}; pass --card-vector-checkpoint or set STS_CARD_VECTOR_CHECKPOINT')
    if not resolved.is_file():
        raise FileNotFoundError(f'card-vector encoder is not a file: {resolved}')
    if entity_config.get('card_vector_checkpoint'):
        entity_config['card_vector_checkpoint'] = str(resolved)
        policy_kwargs['entity_config'] = entity_config
    else:
        policy_kwargs['card_vector_checkpoint'] = str(resolved)
    return {'policy_kwargs': policy_kwargs}


def load_checkpoint(checkpoint, *, device='auto', card_vector_checkpoint=None):
    path = Path(checkpoint)
    if not path.exists() and Path(str(path) + '.zip').is_file():
        path = Path(str(path) + '.zip')
    algorithm = MaskablePPO
    with ZipFile(path) as archive:
        if 'policy.upper_optimizer.pth' in archive.namelist():
            # The joint policy owns two optimizers. Its training class declares
            # both saved state dictionaries; the generic SB3 loader knows one.
            from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
            algorithm = AccumulatingMaskablePPO
    model = algorithm.load(path, device=device,
                           custom_objects=custom_objects(path, card_vector_checkpoint))
    model.policy.set_training_mode(False)
    return model
