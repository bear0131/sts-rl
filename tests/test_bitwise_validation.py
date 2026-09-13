"""The gate must detect changed values, missing fields and runtime drift."""
import json
import pytest
from tools.validation.bitwise import compare


@pytest.mark.parametrize("change", ["value", "missing", "runtime"])
def test_bitwise_gate_rejects_regressions(tmp_path, change):
    before = dict(identity={"native": "fixed"}, commit="before", values={"weight": "abc"})
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(before))
    after = json.loads(json.dumps(before))
    after["commit"] = "after"
    if change == "runtime":
        after["identity"]["native"] = "other"
    elif change == "missing":
        after["values"].clear()
    else:
        after["values"]["weight"] = "changed"
    b.write_text(json.dumps(after))
    with pytest.raises(SystemExit):
        compare(a, b)


def test_bitwise_gate_allows_source_revision_change(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for path, commit in ((a, "before"), (b, "after")):
        path.write_text(json.dumps(dict(identity={}, commit=commit, values={"x": 1})))
    compare(a, b)
