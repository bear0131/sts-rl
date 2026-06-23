"""env/ — Gymnasium environments wrapping the sts_lightspeed sim.

Layout (a real package; run scripts as modules from the repo root, e.g.
`python -m env.scripts.play`):

  core/                 the env core logic
    cards_vocab.py      static canonical card vocabulary (checked in)
    encode.py           canonical combat dict -> fixed obs vector  (pure, no sim)
    actions.py          fixed action space + decode/encode          (pure, no sim)
    render.py           canonical state/action -> human string      (pure, no sim)
    encounters.py       act-1 encounter name lists                  (pure, no sim)
    factories.py        gc_factory builders                         (imports sim)
    combat_env.py       SimCombatEnv                                (imports sim)
    __init__.py         sys.path bootstrap for sim build dir + bridge

  scripts/              runnable entry points (not imported by core)
    play.py             interactive human player
    test_pure.py        pure-module smoke test (no sim needed)
    test_combat.py      sim-backed regression test (needs sim built)

The pure modules (encode/actions/render/encounters/cards_vocab) are
backend-agnostic and import-safe without the sim, so a future RealCombatEnv
(real game via bridge) reuses them unchanged.
"""
