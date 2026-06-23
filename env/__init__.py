"""env/ — Gymnasium environments wrapping the sts_lightspeed sim.

Layout (flat, not installed as a package — import by adding env/ to sys.path):
  cards_vocab.py  static canonical card vocabulary (checked in)
  encode.py       canonical combat dict -> fixed obs vector  (pure, no sim)
  actions.py      fixed action space + decode/encode          (pure, no sim)
  combat_env.py   SimCombatEnv (the only file importing slaythespire)

encode.py and actions.py are backend-agnostic so a future RealCombatEnv (real
game via bridge) reuses them unchanged.
"""
