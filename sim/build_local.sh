#!/usr/bin/env bash
# Build the slaythespire pybind module for THIS machine's local 3.11 .venv.
# Run from the repo root, after `uv sync`:
#     bash sim/build_local.sh
#
# Auto-detects the platform: forces MinGW g++ + copies the runtime DLLs on
# Windows; lets cmake pick the system compiler on mac/Linux (the .so resolves its
# own deps there). The .pyd/.so binds to whatever Python the .venv holds (3.11).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/sim/sts_lightspeed"

# The BASE interpreter behind the .venv (not the venv launcher) — it's the one
# with include/ + libs/ next to it, which cmake's FindPythonLibs needs. The
# module's ABI binds to this Python (3.11 -> cp311). PYTHON_EXECUTABLE is what
# pybind11's classic FindPythonInterp consumes; Python3_EXECUTABLE covers the
# modern FindPython path (mac/Linux). Pass both; the unused one only warns.
PYBIN="$(cd "$ROOT" && uv run python -c "import sys; print(sys._base_executable)")"

args=(-S . -B build -DPYTHON_EXECUTABLE="$PYBIN" -DPython3_EXECUTABLE="$PYBIN")
command -v ninja >/dev/null 2>&1 && args+=(-G Ninja)

windows=0
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) args+=(-DCMAKE_CXX_COMPILER=g++); windows=1 ;;
esac

cmake "${args[@]}"
cmake --build build --target slaythespire -j

if [ "$windows" = 1 ]; then
    # a MinGW-built extension needs these runtime DLLs sitting next to it, so
    # env/'s add_dll_directory(build/) can find them at import.
    gxx_bin="$(dirname "$(command -v g++)")"
    cp "$gxx_bin"/{libgcc_s_seh-1,libstdc++-6,libwinpthread-1}.dll build/
fi

echo "built: $(ls build/slaythespire.* 2>/dev/null || echo '??? (check build output above)')"
