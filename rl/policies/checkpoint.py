"""Load SB3 checkpoints with relocated encoder dependencies resolved first."""
import os
from pathlib import Path
from zipfile import ZipFile

from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import json_to_data


def custom_objects(checkpoint, card_vector_checkpoint=None):
    checkpoint = Path(checkpoint)
    # Inspect metadata only: do not deserialize all weight/optimizer tensors
    # once here and then a second time inside MaskablePPO.load.
    with ZipFile(checkpoint) as archive:
        data = json_to_data(archive.read('data').decode('utf-8'))
    policy_kwargs = dict(data.get('policy_kwargs') or {})
    saved = policy_kwargs.get('card_vector_checkpoint')
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
        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f'missing card-vector encoder {saved}; pass --card-vector-checkpoint or set STS_CARD_VECTOR_CHECKPOINT')
    if not resolved.is_file():
        raise FileNotFoundError(f'card-vector encoder is not a file: {resolved}')
    policy_kwargs['card_vector_checkpoint'] = str(resolved)
    return {'policy_kwargs': policy_kwargs}


def load_checkpoint(checkpoint, *, device='auto', card_vector_checkpoint=None):
    model = MaskablePPO.load(checkpoint, device=device,
                             custom_objects=custom_objects(checkpoint, card_vector_checkpoint))
    model.policy.set_training_mode(False)
    return model
