"""rl: training layer on top of env/.

Depends on env/ (which owns the sim glue); env/ must NOT import rl/. We use
sb3-contrib's MaskablePPO because the action space is fixed-size but only a
subset is legal each step — the agent picks structured actions (play card,
choose target, select from a screen) and the mask blocks the rest. As the
abstraction grows (out-of-combat, map, events) the legal set keeps shifting,
so action masking is load-bearing, not a convenience. Build it in from day one.

Run from the repo root as a module so `env` and the sim path resolve:
    python -m rl.trainer --task wholerun --help
    python -m rl.trainer --task combat --help
"""
