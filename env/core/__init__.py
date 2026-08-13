"""env.core — the env core logic (single combat + whole run).

Import discipline (unchanged from the flat layout, just enforced by the package):
  - pure modules (cards_vocab, vocab, intents, encode, actions, action_space,
    game_actions, encounters, render) must NOT import slaythespire — they are
    backend-agnostic and import-safe without the sim built, so local/CI tests can
    exercise them anywhere.
  - only the sim-backed modules (combat_env, game_env, run_env, factories,
    spiremap) touch the sim. They rely on the sys.path setup below to find the
    built slaythespire module and the bridge schemas.

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

# The built sim module (.so/.pyd) normally lives in the default build dir.
# A separate build can be selected without replacing a module that is currently
# loaded by another Windows process (for example, a running dashboard).
_DEFAULT_BUILD = os.path.join(_REPO, "sim", "sts_lightspeed", "build")
_BUILD = os.path.abspath(
    os.path.expanduser(
        os.environ.get("STS_LIGHTSPEED_BUILD_DIR", _DEFAULT_BUILD)
    )
)
if "STS_LIGHTSPEED_BUILD_DIR" in os.environ and not os.path.isdir(_BUILD):
    raise RuntimeError(
        f"STS_LIGHTSPEED_BUILD_DIR does not exist or is not a directory: {_BUILD}"
    )
# add_dll_directory is Windows-only (needed there to find the mingw runtime dlls
# next to the .pyd); on Linux/mac the .so resolves its deps normally.
_DLL_DIR_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for dll_dir in os.environ.get("STS_LIGHTSPEED_DLL_DIRS", "").split(
        os.pathsep
    ):
        if not dll_dir:
            continue
        dll_dir = os.path.abspath(os.path.expanduser(dll_dir))
        if not os.path.isdir(dll_dir):
            raise RuntimeError(
                "STS_LIGHTSPEED_DLL_DIRS contains a missing directory: "
                f"{dll_dir}"
            )
        _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))
if hasattr(os, "add_dll_directory") and os.path.isdir(_BUILD):
    _DLL_DIR_HANDLES.append(os.add_dll_directory(_BUILD))
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
