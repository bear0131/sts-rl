"""env.core — the combat-env core logic.

Import discipline (unchanged from the flat layout, just enforced by the package):
  - pure modules (cards_vocab, encode, actions, encounters, render) must NOT
    import slaythespire — they are backend-agnostic and import-safe without the
    sim built, so local/CI tests can exercise them anywhere.
  - only combat_env and factories touch the sim. They rely on the sys.path setup
    below to find the built slaythespire module and the bridge schemas.

This __init__ only manipulates sys.path (cheap, no heavy imports), so
`import env.core.encode` works without the sim, while `import env.core.combat_env`
pulls in slaythespire lazily.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# bridge/ holds the (namespace) `schemas` package used by the normalizers.
_BRIDGE = os.path.join(_REPO, "bridge")
if _BRIDGE not in sys.path:
    sys.path.insert(0, _BRIDGE)

# the built sim module (.so/.pyd) lives in the sim build dir.
_BUILD = os.path.join(_REPO, "sim", "sts_lightspeed", "build")
# add_dll_directory is Windows-only (needed there to find the mingw runtime dlls
# next to the .pyd); on Linux/mac the .so resolves its deps normally.
if hasattr(os, "add_dll_directory") and os.path.isdir(_BUILD):
    os.add_dll_directory(os.path.abspath(_BUILD))
if _BUILD not in sys.path:
    sys.path.insert(0, os.path.abspath(_BUILD))
