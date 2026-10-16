"""OBS02: compact leaves must preserve canonical live relic observations."""
from dataclasses import asdict

import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


class LiveRelicObservationMismatch(AssertionError):
    """Only the known OBS02 field mismatch is an expected failure."""


@pytest.mark.parametrize("relics", [
    ["Centennial Puzzle"], ["Lizard Tail"], ["Happy Flower", "Pen Nib", "InkBottle"],
])
@pytest.mark.xfail(strict=True, raises=LiveRelicObservationMismatch, reason=(
    "OBS02: packed/native leaf relic_counters copy persistent root metadata "
    "instead of canonical branch runtime flags/counters; numerical equality fails"
))
def test_leaf_live_relic_counters_match_canonical(combat_factory, relics):
    def make():
        return combat_factory(["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"],
            "Cultist", seed=20261016, relics=relics,
            combat_action_contract="autoregressive-v2", card_language=False)

    native, reference = make(), make()
    config = NativeSearchConfig(simulations=128, leaf_batch_size=1,
        fixed_worldline=True, merge_equivalent_actions=False)
    logits = np.linspace(-.3, .7, native.action_space.n, dtype=np.float32)
    kwargs = dict(**asdict(config), root_value=.5, state_nonce=1729,
                  compact_observation=True)
    pool = NativeSearchBatch([native], 1)
    try:
        pool.begin({0: ((logits,), kwargs)})
        session = reference.nosl_is_az_cpp_begin(logits, **kwargs, _defer_leaf_selection=True)
        comparisons = []
        while not pool.done:
            requests = session.next_batch(1)
            assert len(requests) == 1
            request = requests[0]
            canonical = reference._native_leaf_observation(request["state"], combat_only=True)
            actual = pool.observations()
            comparisons.append((actual["relic_counters"][0].copy(), canonical["relic_counters"].copy()))
            session.backprop(request["request_id"], .5,
                logits.astype(np.float64) if request.get("needs_policy", True) else np.empty(0))
            pool.advance(logits[None], np.array([.5], dtype=np.float32))
        assert len(comparisons) == 128 and session.done()
        # Restrict expected failure to the actual observation contract assertion.
        for actual, expected in comparisons:
            try:
                np.testing.assert_array_equal(actual, expected,
                    err_msg="OBS02: native leaf must encode current branch relic state")
            except AssertionError as exc:
                raise LiveRelicObservationMismatch(str(exc)) from exc
    finally:
        pool.close()
        reference.nosl_is_az_cpp_abort()
