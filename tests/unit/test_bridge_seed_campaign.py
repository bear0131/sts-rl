import json

import pytest

from bridge.campaign import SeedCampaign


def frame(*, in_game, seed=None, game_over=False, victory=False):
    gs = {}
    if seed is not None:
        gs["seed"] = seed
    if game_over:
        gs.update({
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": victory, "score": 123},
            "floor": 17,
            "act": 1,
        })
    return {"in_game": in_game, "game_state": gs}


def test_seed_campaign_reuses_pending_seed_then_advances_after_game_exit(tmp_path):
    path = tmp_path / "campaign.jsonl"
    campaign = SeedCampaign(["310001", "310002"], path)
    assert campaign.start_command("IRONCLAD", 0).endswith(" 310001")
    assert campaign.start_command("IRONCLAD", 0) is None
    assert campaign.observe(frame(in_game=True, seed=310001)) is None
    row = campaign.observe(frame(in_game=True, seed=310001, game_over=True))
    assert row["requested_seed"] == "310001" and not row["victory"]
    assert campaign.observe(frame(in_game=True, seed=310001, game_over=True)) is None
    campaign.observe(frame(in_game=False))
    assert campaign.start_command("IRONCLAD", 0).endswith(" 310002")
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 1 and saved[0]["actual_seed"] == "310001"


def test_seed_campaign_persists_search_evidence(tmp_path):
    path = tmp_path / "campaign.jsonl"
    campaign = SeedCampaign(["7"], path)
    campaign.start_command("IRONCLAD", 0)
    row = campaign.observe(
        frame(in_game=True, seed=7, game_over=True),
        metadata={"search_decisions": 91, "search_errors": 0},
    )
    assert row["search_decisions"] == 91
    assert json.loads(path.read_text(encoding="utf-8"))["search_errors"] == 0


def test_seed_campaign_rejects_duplicate_or_wrong_actual_seed(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        SeedCampaign(["7", "7"])
    campaign = SeedCampaign(["7"], tmp_path / "campaign.jsonl")
    campaign.observe(frame(in_game=True, seed=8))
    with pytest.raises(RuntimeError, match="seed mismatch"):
        campaign.observe(frame(in_game=True, seed=8, game_over=True))
