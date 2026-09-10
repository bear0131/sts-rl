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
# The current Full-Gumbel/NOSL build is kept separately so it can be rebuilt
# without replacing a module that is currently loaded by another Windows
# process (for example, a running dashboard).  Prefer that build for the
# no-configuration path, while retaining the older NOSL build as a fallback
# for clean checkouts and remote environments.
_DEFAULT_BUILD = os.path.join(_REPO, "sim", "sts_lightspeed", "build")
_PREFERRED_BUILD = os.path.join(
    _REPO, "sim", "sts_lightspeed", "build-gumbel-current"
)
_LEGACY_PREFERRED_BUILD = os.path.join(
    _REPO, "sim", "sts_lightspeed", "build-nosl311"
)
_configured_build = os.environ.get("STS_LIGHTSPEED_BUILD_DIR")
if _configured_build is None:
    if os.path.isdir(_PREFERRED_BUILD):
        _configured_build = _PREFERRED_BUILD
    elif os.path.isdir(_LEGACY_PREFERRED_BUILD):
        _configured_build = _LEGACY_PREFERRED_BUILD
    else:
        _configured_build = _DEFAULT_BUILD
_BUILD = os.path.abspath(os.path.expanduser(_configured_build))
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
