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
            reducer.after_physical_chunk(phase, 1, 1)
        reducer.synchronize(0)
        expected = [7 if 'map' in phases else 0, 3*('map' in phases)+11*('combat' in phases)]
        assert [x.item() for x in batches] == expected
        assert core.overworld.weight.grad.item() == expected[0]+2
        assert core.context.weight.grad.item() == expected[1]+2
    finally:
        release.set()
        reducer.close()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA gradient spill and return'))])
def test_oom_subdivision_keeps_each_original_fp32_addition(monkeypatch, device):
    from rl.algorithms import updates
    from rl.algorithms.ppo import PPOLossComponents
    from tools.benchmarks.ppo_overlap import UpperFirstSchedule

    def run(scheduled):
        policy = torch.nn.Module()
        core = policy.core = torch.nn.Module()
        core.context = torch.nn.Linear(1, 1, bias=False)
        core.anchor = core.combat = torch.nn.Identity()
        core.upper_tokenizer = core.context
        core.overworld = torch.nn.Linear(1, 1, bias=False)
        policy.to(device)
        learner = SimpleNamespace(policy=policy, device=torch.device(device))
        def forward(model, batch, *args, **kwargs):
            if len(batch.advantages) == 2:
                # A failed chunk may have dirtied a gradient before throwing.
                core.context.weight.grad = torch.full_like(core.context.weight, 123.)
                raise torch.OutOfMemoryError('injected')
            loss = (batch.advantages * core.context.weight).sum()
            z = loss.detach()*0
            return PPOLossComponents(z, z, z, z, z, 1, 0, 1, None, None), loss
        monkeypatch.setattr(updates, 'ppo_forward', forward)
        def chunk(values, phase):
            xs = torch.tensor(values, device=device)
            return SimpleNamespace(advantages=xs), xs, 1., phase
        # (((2**25) + (-2**25)) + 1) = 1, but 2**25 + ((-2**25) + 1) = 0.
        chunks = [chunk([2.**25], 'combat'), chunk([-2.**25, 1.], 'map')]
        schedule = UpperFirstSchedule(learner) if scheduled else None
        for batch, adv, _, phase in (schedule.schedule(chunks) if schedule else chunks):
            updates.ppo_chunked_backward(learner, batch, .2, None, pre_normalized_advantages=adv,
                backward_scale=len(adv), oom_group=phase,
                after_physical_chunk=schedule.after_physical_chunk if schedule else None)
        assert learner._ppo_microbatch_stats['oom_retries'] == 1
        assert learner._ppo_update_metric_count == 3
        return core.context.weight.grad.item()
    assert run(False) == run(True) == 1.
