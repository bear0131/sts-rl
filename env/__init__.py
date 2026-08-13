"""env/ — Gymnasium environments wrapping the sts_lightspeed sim.

Layout (a real package; run scripts as modules from the repo root, e.g.
`python -m env.scripts.play`):

  core/                 the env core logic
    cards_vocab.py      static canonical card vocabulary (checked in)
    vocab.py            generated categorical vocabularies (monster/move/...) (pure)
    intents.py          move name -> INTENT_CATEGORY derivation               (pure)
    encode.py           canonical dict -> structured Dict obs (OBS_VERSION 11) (pure)
    actions.py          combat-internal sts<->id mapping (71 ids)              (pure)
    action_space.py     thin run-wide 181-id space (ACTION_VERSION 6)          (pure)
    game_actions.py     backend-agnostic semantic-action vocabulary            (pure)
    render.py           canonical state/action -> human string                 (pure)
    encounters.py       act-1 encounter name lists                             (pure)
    spiremap.py         RunMap: rebuilt SpireMap reachability               (imports sim)
    factories.py        gc_factory builders                                 (imports sim)
    combat_env.py       SimCombatEnv (single combat)                        (imports sim)
    game_env.py         SimGameEnv (thick whole-run state machine)          (imports sim)
    run_env.py          SlayTheSpireEnv (thin run-wide gym.Env)             (imports sim)
    __init__.py         sys.path bootstrap for sim build dir + bridge

  scripts/              runnable entry points (not imported by core)
    play.py             interactive single-combat player
    play_run.py / play_run_v2.py   interactive whole-run players
    test_pure.py        pure-module smoke test (no sim needed)
    test_combat.py      single-combat regression test (needs sim built)
    test_game.py        thick whole-run regression (needs sim)
    test_run_v2.py      thin whole-run regression (needs sim)

The pure modules listed above are backend-agnostic and import-safe without the
sim, so a future RealCombatEnv / RealGameEnv (real game via bridge) reuses them
unchanged.
"""
