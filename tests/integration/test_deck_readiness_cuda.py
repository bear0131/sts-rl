"""Independent readiness inference must be able to use the parent's idle VRAM."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from rl.algorithms.deck_schedule import DeckTrainingSchedule
from rl.training.deck_schedule import at_collection_boundary

pytestmark = [pytest.mark.training, pytest.mark.cuda, pytest.mark.timeout(120)]


def test_readiness_child_can_allocate_after_parent_update_cache(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), 'explicit CUDA readiness qualification required'
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    # The child needs more than the device remainder while this is cached.
    size = int(free * .75) // (2**20) * (2**20)
    live = torch.arange(1024, device='cuda')
    scratch = torch.empty(size, dtype=torch.uint8, device='cuda')
    scratch.fill_(7)
    torch.cuda.synchronize()
    del scratch
    assert torch.cuda.memory_reserved() >= size
    launch = subprocess.run

    def child(command, **kwargs):
        output = command[command.index('--output') + 1]
        code = (
            'import json,torch;from pathlib import Path;'
            f'x=torch.empty({size},dtype=torch.uint8,device="cuda");x.fill_(9);'
            'torch.cuda.synchronize();'
            f'Path({output!r}).write_text(json.dumps({dict(natural=64, censored=0, anchors=128, target_sum=64, target_sq_sum=64, error_sum=0, error_sq_sum=1, relic_offered=30, relic_taken=25, floor_sum=512, wins=2, timeout=0)!r}))'
        )
        return launch([sys.executable, '-c', code], **kwargs)

    monkeypatch.setattr('rl.training.deck_schedule.subprocess.run', child)
    model = SimpleNamespace(
        device=torch.device('cuda'), ppo_rollouts_completed=100,
        policy=SimpleNamespace(core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
                               combat_value_target='whole-run'), policy_kwargs={},
        deck_training_schedule=asdict(DeckTrainingSchedule()),
        save=lambda path: Path(path).write_bytes(b'checkpoint'),
        logger=SimpleNamespace(record=lambda *args: None))
    at_collection_boundary(model, rank=0, world_size=1, local_root=tmp_path,
                           ascension=0, average=lambda values: values)
    assert model.deck_training_state['validation']['passed']
    assert model.deck_training_state['good_checks'] == 1
    torch.testing.assert_close(live, torch.arange(1024, device='cuda'))
    assert not (tmp_path / 'anchor-readiness/model.zip').exists()
    saved = json.loads((tmp_path / 'anchor-readiness/round-100-global.json').read_text())
    assert saved == model.deck_training_state
