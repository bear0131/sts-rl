"""Ranking files and decisions share one explicit, deterministic contract."""
import json

import pytest

from apps.card_ranking import RankingStore, preview_ranking
from env.core.ironclad_cards import IRONCLAD_CARDS
from env.search.card_ranking import CardRanking, SKIP


def ranking_with(*first):
    return CardRanking(tuple(first) + tuple(cid for cid in IRONCLAD_CARDS if cid not in first)
                       + (() if SKIP in first else (SKIP,)))


def test_pick_uses_offer_index_not_catalog_or_priority_index():
    ranking = ranking_with("Demon Form", "Sword Boomerang", SKIP)
    assert ranking.choose(["Clash", "Sword Boomerang", "Demon Form"]) == 2
    assert ranking.choose(["Demon Form", "Demon Form"]) == 0
    assert ranking.choose(["Clash", "Defend_R"]) is None
    assert ranking.choose(["Clash", "Defend_R"], can_skip=False) == 1
    assert ranking.choose([]) is None
    with pytest.raises(ValueError, match="不能跳过"):
        ranking.choose([], can_skip=False)
    with pytest.raises(ValueError, match="未排名"):
        ranking.choose(["Unknown", "Demon Form"])


@pytest.mark.parametrize("change", [
    lambda d: d["order"].pop(),
    lambda d: d["order"].__setitem__(1, d["order"][0]),
    lambda d: d["order"].__setitem__(0, "unknown"),
    lambda d: d["order"].__setitem__(0, {}),
    lambda d: d.__setitem__("schema_version", True),
    lambda d: d.__setitem__("schema_version", 2),
    lambda d: d.__setitem__("character", "SILENT"),
    lambda d: d.__setitem__("upgrade_policy", "separate"),
])
def test_invalid_file_never_overwrites_saved_ranking(tmp_path, change):
    store = RankingStore(tmp_path / "ranking.json")
    data = ranking_with("Demon Form", SKIP).to_dict()
    state = store.save(dict(ranking=data, base_revision=None))
    original = store.path.read_bytes()
    change(data)
    with pytest.raises(ValueError):
        store.save(dict(ranking=data, base_revision=state["revision"]))
    assert store.path.read_bytes() == original


def test_roundtrip_and_stale_tab_protection(tmp_path):
    store = RankingStore(tmp_path / "nested/ranking.json")
    initial = store.load()
    assert initial["revision"] is None and not store.path.exists()
    assert {c["id"] for c in initial["cards"]} == set(IRONCLAD_CARDS)
    data = ranking_with("Demon Form", SKIP).to_dict()
    saved = store.save(dict(ranking=data, base_revision=None))
    assert RankingStore(store.path).load() == saved
    assert json.loads(store.path.read_text()) == data
    changed = ranking_with(SKIP).to_dict()
    with pytest.raises(ValueError, match="其他页面"):
        store.save(dict(ranking=changed, base_revision=None))
    assert store.load() == saved
    updated = store.save(dict(ranking=changed, base_revision=saved["revision"]))
    assert updated["ranking"] == changed
    assert updated["revision"] != saved["revision"]


def test_preview_uses_shared_ranking_not_separate_ui_scoring():
    ranking = ranking_with("Demon Form", SKIP)
    assert preview_ranking(dict(ranking=ranking.to_dict(), cards=["Clash", "Demon Form"])) == dict(index=1, card="Demon Form")
    assert preview_ranking(dict(ranking=ranking.to_dict(), cards=["Clash"])) == dict(index=None, card=SKIP)
