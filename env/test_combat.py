"""End-to-end regression test for SimCombatEnv. Requires the sim to be built
(sim/sts_lightspeed/build). Run on the training machine:

    python env/test_combat.py

Plays N random-but-legal episodes across the act-1 pools and asserts the env
contract holds: obs shape/dtype/finiteness, the action mask agrees with what
bc.execute accepts, episodes terminate, and rewards stay finite.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from combat_env import SimCombatEnv  # noqa: E402
import encode                        # noqa: E402
import actions                       # noqa: E402


def _check_obs(obs):
    assert obs.shape == (encode.OBS_SIZE,), f"obs shape {obs.shape}"
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
    assert np.all(np.isfinite(obs)), "obs has non-finite values"


def _play_one(env, rng, reset_seed=None, max_steps=300):
    obs, info = env.reset(seed=reset_seed)
    _check_obs(obs)
    mask = info["action_mask"]
    assert mask.shape == (actions.N_ACTIONS,)
    assert mask.any(), "reset produced no legal actions"
    # A combat with no monsters never terminates (and means the encounter name
    # failed to resolve in create_battle — see encounters.py). Catch it loudly.
    n_mon = len(env.bc.to_dict()["monsters"])
    assert n_mon > 0, f"combat spawned 0 monsters (bad encounter name?)"

    steps = 0
    total_r = 0.0
    while steps < max_steps:
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal))
        obs, r, term, trunc, info = env.step(a)
        _check_obs(obs)
        assert np.isfinite(r), f"non-finite reward {r}"
        total_r += r
        mask = info["action_mask"]
        steps += 1
        if term or trunc:
            # On termination the last reward carries the ±1 win/loss bonus.
            # Verify its sign matches the actual outcome (guards the
            # Outcome-vs-GameOutcome enum mismatch).
            raw = env.bc.to_dict()
            outcome = raw["outcome"]  # combat Outcome: 1=victory, 2=loss
            won = outcome == 1
            if won:
                assert r > 0, f"victory but terminal reward {r} <= 0"
                assert raw["player"]["hp"] > 0, "victory but player hp <= 0"
            else:
                assert r < 0, f"loss but terminal reward {r} >= 0"
            return steps, total_r, term, won
    raise AssertionError(f"episode did not terminate in {max_steps} steps")


def _check_all_encounters_spawn():
    """Every encounter name must resolve in create_battle (>=1 monster).
    Guards against the SaveFileMappings-vs-monsterEncounterStrings name mismatch."""
    import encounters
    from factories import fixed_factory
    bad = []
    for pool, (names, _) in encounters.POOLS.items():
        for nm in names:
            env = SimCombatEnv(gc_factory=fixed_factory(seed=42, encounter=nm, floor=2))
            env.reset()
            if len(env.bc.to_dict()["monsters"]) == 0:
                bad.append((pool, nm))
    assert not bad, f"encounters spawned 0 monsters (bad names): {bad}"
    print("OK — all act-1 encounter names resolve to >=1 monster.")


def main():
    _check_all_encounters_spawn()
    rng = np.random.default_rng(0)
    n_per_pool = 5
    for pool in ("normal", "elite", "boss"):
        env = SimCombatEnv(pool=pool)
        for i in range(n_per_pool):
            # Distinct reset seed per episode for reproducibility + coverage.
            steps, total_r, term, won = _play_one(env, rng, reset_seed=1000 * (i + 1) + len(pool))
            print(f"[{pool}] episode {i}: {steps} steps, reward {total_r:+.3f}, "
                  f"won={won}, terminated={term}")
    print("OK — SimCombatEnv contract holds across act-1 pools.")


if __name__ == "__main__":
    main()
