"""Generated combat piles must retain all cards beyond inline storage."""
import gc
import numpy as np
import pytest
import torch
from env.core.run_env import _nosl_compact_lut
from env.search.cuda_battle import CudaBattleArena

pytestmark = [pytest.mark.native, pytest.mark.cuda, pytest.mark.timeout(90)]


def test_resident_generated_pile_grows_past_inline_capacity_and_forks(complete_combat_factory, cuda_battle_artifacts):
    from tests.bridge_fixtures import native_card, native_card_state
    root = complete_combat_factory(['ANGER'] * 5, seed=42, deck_latent_context=True)
    root.thick.bc.set_card_state(native_card_state(
        hand=[native_card('ANGER', 0, base_cost=0, cost=0)],
        discard=[native_card('WOUND', i, base_cost=-2, cost=-2) for i in range(1, 129)],
        next_uid=129))
    root._invalidate_backend_cache()
    folder, manifest, bridge = cuda_battle_artifacts
    initial = bridge.snapshot(root.thick.bc)
    assert initial['max_vector'] == 128 and initial['portable']
    arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 3,
        card_count=_nosl_compact_lut()['layout']['card_count'], graph_capacity=1)
    branch = None
    try:
        arena.load([initial, initial], [0, 0])
        arena.observe(2)
        assert root.action_masks()[root._policy_offset + 1]
        arena.apply([-1, 1]);root.step(root._policy_offset + 1)
        grown = bridge.snapshot(root.thick.bc)
        assert grown['max_vector'] == 130
        arena.clear_graphs();gc.collect();torch.cuda.empty_cache()
        arena.fork([1], [2], [31])
        branch = root.fork_run(worldline=31)
        branch.thick.bc.nosl_redeterminize(31);branch._invalidate_backend_cache()
        arena.observe(3);arena.apply([-1, -1, 0]);branch.step(branch._policy_offset)
        words, metadata = arena.snapshot(3)
        for i, expected in enumerate((initial, grown, bridge.snapshot(branch.thick.bc))):
            np.testing.assert_array_equal(words[i, :len(expected['words'])], expected['words'])
            assert metadata[i, 2] == expected['max_vector']
    finally:
        arena.close()
        if branch is not None:branch.close()


