from __future__ import annotations

from env.core.card_language import LANGUAGE_CARDS
from env.core.rule_language import COMBAT_RELIC_RULES
from rl.render_card_relic_rules import render_rules


def test_catalog_has_one_line_per_ironclad_status_version_and_relic():
    lines = render_rules().splitlines()
    assert len(lines) == len(LANGUAGE_CARDS) * 2 + len(COMBAT_RELIC_RULES)
    assert sum(line.startswith("CARD\t") for line in lines) == len(LANGUAGE_CARDS) * 2
    assert sum(line.startswith("RELIC\t") for line in lines) == len(COMBAT_RELIC_RULES)
    assert all(len(line.split("\t")) == 4 for line in lines if line.startswith("CARD\t"))
    assert all(len(line.split("\t")) == 3 for line in lines if line.startswith("RELIC\t"))
    assert any(line.startswith("CARD\tBash\tbase\t") for line in lines)
    assert any(line.startswith("CARD\tBurn\tupgraded\t") for line in lines)
    assert any(line.startswith("RELIC\tSundial\t") for line in lines)
    assert not any(line.startswith("CARD\tAccuracy\t") for line in lines)
