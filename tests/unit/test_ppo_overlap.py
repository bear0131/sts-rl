"""Readiness and ownership checks for the CUDA prototype, including absent phases."""
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from tools.benchmarks.ppo_overlap import EarlyUpperReduction


@pytest.mark.parametrize('phases', [('combat',), ('map',), ('combat', 'map')])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires pinned CUDA transfers')
def test_upper_copy_is_immutable_and_collective_order_survives_missing_phase(phases):
    policy = torch.nn.Module()
    core = policy.core = torch.nn.Module()
    core.context = torch.nn.Linear(1, 1, bias=False)
    core.combat = torch.nn.Identity()
    core.anchor = torch.nn.Identity()
    core.upper_tokenizer = core.context  # shared, so it must remain in the late bucket
    core.overworld = torch.nn.Linear(1, 1, bias=False)
    policy.cuda()
    model = SimpleNamespace(policy=policy, device=torch.device('cuda'), ppo_update_loss_scale=1,
                            ppo_update_config=dict(precision='fp32', upper_screen_max_rows=4),
                            _ppo_microbatch_stats=dict(chunks=0))
    begun, release = Event(), Event()
    batches = []
    class Ring:
        def average(self, host):
            batches.append(host.clone())
            if len(batches) == 1:
                begun.set()
                assert release.wait(5)
            return host.add_(2), host.nbytes
    reducer = EarlyUpperReduction(model, Ring())
    try:
        if 'combat' not in phases:
            release.set()
        for *_, phase in reducer.schedule([(None, None, None, p) for p in phases]):
            model._ppo_microbatch_stats['chunks'] += 1
            if phase == 'map':
                core.overworld.weight.grad = torch.full_like(core.overworld.weight, 7)
                core.context.weight.grad = torch.full_like(core.context.weight, 3)
            else:
                # The transfer is allowed to progress even while consume-side
                # backward continues. Simulate a later zero/rebinding of .grad.
                assert begun.wait(5)
                core.overworld.weight.grad = None
                if core.context.weight.grad is None:
                    core.context.weight.grad = torch.zeros_like(core.context.weight)
                core.context.weight.grad.add_(11)
                release.set()
        reducer.synchronize(0)
        expected = [7 if 'map' in phases else 0, 3*('map' in phases)+11*('combat' in phases)]
        assert [x.item() for x in batches] == expected
        assert core.overworld.weight.grad.item() == expected[0]+2
        assert core.context.weight.grad.item() == expected[1]+2
    finally:
        release.set()
        reducer.close()
