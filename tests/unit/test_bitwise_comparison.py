import json

import numpy as np
import pytest

from tools.validation.bitwise import _record, compare


def receipts(tmp_path, **changed):
    identity = dict(kind="native-search", native_sha256="baseline", compiler_sha256="gcc14",
                    build_flags={"CMAKE_BUILD_TYPE": "Release"}, device="cpu", threads=1)
    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for path, updates in zip(paths, ({}, changed)):
        path.write_text(json.dumps(dict(identity=identity | updates, values={"q": 0.4})),
                        encoding="utf-8")
    return paths


def test_native_rebuild_requires_explicit_opt_in(tmp_path):
    paths = receipts(tmp_path, native_sha256="candidate")
    with pytest.raises(SystemExit, match="identities differ"):
        compare(*paths)
    compare(*paths, allow_native_rebuild=True)


@pytest.mark.parametrize("changed", [dict(compiler_sha256="gcc15"), dict(device="cuda"),
                                    dict(build_flags={"CMAKE_BUILD_TYPE": "Debug"}),
                                    dict(threads=2)])
def test_native_rebuild_does_not_relax_runtime_or_compiler_identity(tmp_path, changed):
    with pytest.raises(SystemExit, match="identities differ"):
        compare(*receipts(tmp_path, **changed), allow_native_rebuild=True)


def test_native_rebuild_rejects_unidentified_capture(tmp_path):
    with pytest.raises(SystemExit, match="requires capture-search"):
        compare(*receipts(tmp_path, kind="training"), allow_native_rebuild=True)


def test_native_rebuild_still_compares_every_behavior_field(tmp_path):
    paths = receipts(tmp_path, native_sha256="candidate")
    changed = json.loads(paths[1].read_text())
    changed["values"]["q"] = 0.41
    paths[1].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="bitwise differences"):
        compare(*paths, allow_native_rebuild=True)


def test_record_preserves_dtype_shape_and_byte_identity():
    a, b = {}, {}
    _record(a, "obs", {"values": np.asarray([1.0, 2.0], dtype=np.float32)})
    _record(b, "obs", {"values": np.asarray([1.0, 2.0], dtype=np.float64)})
    assert a["obs/values"]["shape"] == [2]
    assert a["obs/values"] != b["obs/values"]
