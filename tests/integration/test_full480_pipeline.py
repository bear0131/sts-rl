"""Original full480 execution mechanisms, with explicit frozen artifacts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def test_original_full480_whole_step_graph_and_frame_growth(combat_factory):
    """The old model is a porting oracle, never the current training policy."""
    assert torch.cuda.is_available(), 'full480 qualification requires CUDA and its explicit artifacts'
    from rl.search.full480.run_device import cpu_bridge
    from rl.search.full480.combo_runtime import BattleArena, BattlePool
    from rl.policies.token_regret import OnlineTokenRegretPolicy, encode_public_roots, pack_rows

    torch.manual_seed(20261111)
    torch.set_float32_matmul_precision('highest')
    cpu = cpu_bridge()
    envs = [combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 4 + ['BASH'],
                           seed=20261111 + i, combat_action_contract='autoregressive-v2')
            for i in range(2)]
    with torch.inference_mode():
        net = OnlineTokenRegretPolicy(785).cuda().eval()
        flat = SimpleNamespace(net=net, precision='tf32x3',
            lookup=net.symbol_lift(net.symbol.weight).detach(),
            head_weight=torch.cat((net.source_score.weight, net.left.weight,
                                  net.target_score.weight, net.right.weight), 0).detach())
        arena = BattleArena(2)
        pool = None
        try:
            versions = arena.load({i: cpu.State(env.thick.bc).snapshot()
                                   for i, env in enumerate(envs)})
            pool = BattlePool(arena, 16, net=net, flat=flat, frame=32, tile=32,
                              frame_tokens=4, frame_queries=4)
            rows = {}
            for i, env in enumerate(envs):
                row = cpu.State(env.thick.bc).root_plan(20261111 + i, 16, 1., 4, False)
                row.update(cpu.inventory(env.thick.gc))
                row['mirror'] = (i, *versions[i])
                rows[i] = row
            pool.submit(rows)
            pool.observe()
            raw = pool.public()
            pool.score()
            for row, tokens in zip(raw, encode_public_roots(raw), strict=True):
                row['tokens'] = tokens
            expected = net.forward_packed(*pack_rows(raw, 'cuda')).cpu().numpy()
            metadata = pool.meta[:pool.count].cpu().numpy()
            scores = pool.scores.cpu().numpy()
            for i, row in enumerate(raw):
                count, _, start, _ = map(int, metadata[i])
                if count > 1:
                    np.testing.assert_allclose(scores[start:start + count], expected[i, :count],
                                               atol=3e-5, rtol=1e-5)
            pool.apply()
            pool.refresh()
            temporary = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda')
            temporary.fill_(7)
            del temporary
            torch.cuda.empty_cache()
            completed = {}
            for _ in range(8):
                completed.update(pool.receive())
                if not pool.pending:
                    break
            assert set(completed) == set(rows)
            assert all(r['simulations'] == 16 for r in completed.values())
            statistics = pool.graph_scorer.statistics()
            assert statistics['overflow_retries'] > 0
            assert statistics['step_captures'] > 0 and statistics['step_replays'] > 0
            assert statistics['workspace_alias_version'] == 1
            assert not pool.page_leases and not pool.ledger.pending
            pool.close()
            assert pool.state_pages.closed
        finally:
            if pool is not None and not pool.closed:
                pool.graph_scorer.close()
                pool.launch('stream_destroy', [pool.arena, pool.constructed], [pool.n], grid=pool.n)
                pool.launch('stream_destroy', [pool.roots, pool.root_constructed], [pool.m], grid=pool.m)
                torch.cuda.synchronize()
                pool.state_pages.release(list(pool.page_leases.values()))
                pool.page_leases.clear()
                pool.state_pages.close()
                pool.modules.references -= 1
                pool.closed = True
            arena.close()
        assert arena.modules.references == 0 and arena.modules.closed
