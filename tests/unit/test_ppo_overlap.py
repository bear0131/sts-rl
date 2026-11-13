"""Readiness and ownership checks for the CUDA prototype, including absent phases."""
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from rl.training.ppo_overlap import EarlyUpperReduction


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
    from rl.training.ppo_overlap import UpperFirstSchedule

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


def test_failed_first_graph_capture_retries_eagerly(monkeypatch):
    from rl.policies import packed_update_graphs as graphs

    core = SimpleNamespace(overworld=torch.nn.Identity(), upper_tokenizer=torch.nn.Identity())
    cache = graphs.PackedHeadGraphs(core)
    untouched = dict(cache.stats)
    assert untouched == {head + '_' + event: 0 for head in ('combat', 'upper')
                         for event in ('captures', 'replays', 'retired')}

    def fail(*args):
        raise torch.OutOfMemoryError('injected during first capture')

    monkeypatch.setattr(graphs, '_CapturedHead', fail)
    with pytest.raises(torch.OutOfMemoryError):
        cache._entry('upper', None, {}, None, 256, (32, 16, 128))
    assert cache.recover_oom()
    assert cache._entry('upper', None, {}, None, 256, (32, 16, 128)) is None
    assert cache.stats['upper_retired'] == 1
    assert set(cache.stats) == set(untouched)
    assert not cache.recover_oom()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA training graphs')
def test_captured_head_preserves_tied_parameters_and_default_stream_gradients():
    from rl.policies.packed_update_graphs import _CapturedHead

    class TiedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cards = torch.nn.Linear(4, 4, bias=False)
            self.alias = self.cards
            self.readout = torch.nn.Linear(4, 3)

        def forward(self, obs):
            x = self.cards(obs['features']) + self.alias(obs['_run_context'])
            return self.readout(x), x.mean(dim=1, keepdim=True), {}

    device = torch.device('cuda')
    model = TiedHead().to(device)
    weight = model.cards.weight
    parameters = tuple(model.parameters())
    sample = torch.randn(5, 4, device=device)
    z = torch.randn_like(sample, requires_grad=True)
    # Materialize the original leaf accumulators on the default stream first.
    with torch.cuda.stream(torch.cuda.default_stream(device)):
        model(dict(features=sample, _run_context=z))[0].sum().backward()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        entry = _CapturedHead(model, dict(features=sample), z, 8, 1)
        for n in (5, 8, 3):
            with torch.no_grad():
                weight.add_(.01)
            x = torch.randn(n, 4, device=device)
            z = torch.randn_like(x, requires_grad=True)
            actual = entry.replay(dict(features=x), z)
            expected = model(dict(features=x, _run_context=z))[:2]
            dy = tuple(torch.randn_like(o) for o in actual)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b)
            for a, b in zip(torch.autograd.grad(actual, (z, *parameters), dy),
                            torch.autograd.grad(expected, (z, *parameters), dy), strict=True):
                torch.testing.assert_close(a, b)
            assert model.cards.weight is weight and model.alias.weight is weight
            assert entry.graph_module.alias.weight is entry.graph_module.cards.weight
            assert entry.graph_module.cards.weight is not weight
            assert entry.graph_module.cards.weight.data_ptr() == weight.data_ptr()
    torch.cuda.current_stream(device).wait_stream(stream)
