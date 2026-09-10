"""Canonical one-line rule language for rule-model supervision.

Cards use the existing exact card DSL.  Relics use one stable sentence per
combat relic sampled by the rule/combat loadout factory.  These strings are
model inputs, not display copy: changing their wording changes the rule-data
contract and therefore requires a language-version bump.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .card_language import LANGUAGE_CARDS, card_program


RULE_LANGUAGE_VERSION = 1


EXTRA_CARD_RULES: dict[str, str] = {
    "Clumsy": "CARD Clumsy TYPE CURSE TRAITS UNPLAYABLE ETHEREAL",
    "Decay": "CARD Decay TYPE CURSE TRIGGER END_TURN_IN_HAND LOSE_HP 2",
    "Doubt": "CARD Doubt TYPE CURSE TRIGGER END_TURN_IN_HAND APPLY WEAK 1 SELF",
    "Injury": "CARD Injury TYPE CURSE TRAIT UNPLAYABLE",
    "Normality": "CARD Normality TYPE CURSE WHILE_IN_HAND MAX_CARDS_PLAYED_THIS_TURN 3",
    "Pain": "CARD Pain TYPE CURSE TRIGGER ON_PLAY_OTHER_CARD LOSE_HP 1 SELF",
    "Parasite": "CARD Parasite TYPE CURSE TRAIT UNPLAYABLE TRIGGER ON_REMOVE_FROM_DECK LOSE_MAX_HP 3",
    "Regret": "CARD Regret TYPE CURSE TRIGGER END_TURN_IN_HAND LOSE_HP CARDS_IN_HAND",
    "Shame": "CARD Shame TYPE CURSE TRIGGER END_TURN_IN_HAND APPLY FRAIL 1 SELF",
    "Writhe": "CARD Writhe TYPE CURSE TRAITS UNPLAYABLE INNATE",
}
RULE_CARD_IDS = LANGUAGE_CARDS | EXTRA_CARD_RULES.keys()


# Stable ids are the ids exported by GameContext.relic_state().  Scope is the
# complete relic population used by random_gauntlet_factory plus the Ironclad
# starter relic.  Out-of-combat-only relics are deliberately outside the first
# rule-model dataset.
COMBAT_RELIC_RULES: dict[str, str] = {
    "Akabeko": "The first Attack played each combat deals 8 additional damage.",
    "Anchor": "At the start of combat, gain 10 Block.",
    "Bag of Marbles": "At the start of combat, apply 1 Vulnerable to all enemies.",
    "Bag of Preparation": "At the start of combat, draw 2 additional cards.",
    "Blood Vial": "At the start of combat, heal 2 HP.",
    "Bronze Scales": "At the start of combat, gain 3 Thorns.",
    "Busted Crown": "Gain 1 Energy each turn, but card rewards contain 2 fewer cards.",
    "Burning Blood": "At the end of combat, heal 6 HP.",
    "Coffee Dripper": "Gain 1 Energy each turn, but Rest sites cannot be used to rest.",
    "Cursed Key": "Gain 1 Energy each turn, but opening a non-boss chest adds a Curse.",
    "Fusion Hammer": "Gain 1 Energy each turn, but Rest sites cannot be used to smith.",
    "Gambling Chip": "At the start of combat, cards may be discarded and the same number are drawn.",
    "Gremlin Horn": "Whenever an enemy dies, gain 1 Energy and draw 1 card.",
    "Happy Flower": "Every 3 turns, gain 1 Energy.",
    "Ice Cream": "Unspent Energy is retained between turns.",
    "Kunai": "After playing 3 Attacks in one turn, gain 1 Dexterity.",
    "Lantern": "At the start of combat, gain 1 Energy.",
    "Letter Opener": "After playing 3 Skills in one turn, deal 5 damage to all enemies.",
    "Mark of Pain": "Gain 1 Energy each turn and shuffle 2 Wounds into the draw pile at combat start.",
    "Mercury Hourglass": "At the start of each turn, deal 3 damage to all enemies.",
    "Nunchaku": "Every 10 Attacks played, gain 1 Energy.",
    "Oddly Smooth Stone": "At the start of combat, gain 1 Dexterity.",
    "Ornamental Fan": "After playing 3 Attacks in one turn, gain 4 Block.",
    "Paper Crane": "Weak enemies deal 40 percent less attack damage instead of 25 percent less.",
    "Paper Frog": "Vulnerable enemies take 75 percent more attack damage instead of 50 percent more.",
    "Pen Nib": "Every 10th Attack played deals double damage.",
    "Philosopher's Stone": "Gain 1 Energy each turn and all enemies gain 1 Strength at combat start.",
    "Red Skull": "While HP is at or below 50 percent, gain 3 Strength.",
    "Runic Cube": "Whenever HP is lost, draw 1 card.",
    "Runic Dome": "Gain 1 Energy each turn, but enemy intents are hidden.",
    "Runic Pyramid": "Cards in hand are not discarded at the end of the turn.",
    "Shuriken": "After playing 3 Attacks in one turn, gain 1 Strength.",
    "SlaversCollar": "During Elite and Boss combats, gain 1 Energy each turn.",
    "Snecko Eye": "Draw 2 additional cards each turn and randomize drawn card costs between 0 and 3.",
    "Sozu": "Gain 1 Energy each turn, but potions can no longer be obtained.",
    "StrikeDummy": "Cards whose name contains Strike deal 3 additional damage.",
    "Sundial": "Every 3 times the draw pile is shuffled, gain 2 Energy.",
    "Vajra": "At the start of combat, gain 1 Strength.",
    "Velvet Choker": "Gain 1 Energy each turn, but no more than 6 cards may be played each turn.",
}


def _card_version_name(upgrades: int) -> str:
    return "upgraded" if int(upgrades) > 0 else "base"


def card_rule_sentence(card_id: str, upgrades: int = 0) -> str:
    """Return one exact, whitespace-tokenizable statement for a card version."""
    if card_id in EXTRA_CARD_RULES:
        return EXTRA_CARD_RULES[card_id]
    if card_id not in LANGUAGE_CARDS:
        raise KeyError(f"no rule language for card {card_id!r}")
    program = " ".join(card_program(card_id, bool(upgrades)))
    return f"CARD {card_id} VERSION {_card_version_name(upgrades)} PROGRAM {program}"


def relic_rule_sentence(relic_id: str) -> str:
    try:
        effect = COMBAT_RELIC_RULES[relic_id]
    except KeyError as exc:
        raise KeyError(f"no first-version combat rule language for relic {relic_id!r}") from exc
    return f"RELIC {relic_id} EFFECT {effect}"


def deck_rule_sentence(cards: Iterable[tuple[str, int] | dict]) -> str:
    """Describe a complete deck and every distinct card version in one line."""
    versions: Counter[tuple[str, int]] = Counter()
    for card in cards:
        if isinstance(card, dict):
            card_id = str(card["id"])
            upgrades = int(card.get("up", card.get("upgrades", 0)))
        else:
            card_id, upgrades = str(card[0]), int(card[1])
        versions[(card_id, upgrades)] += 1
    parts = ["DECK"]
    for (card_id, upgrades), count in sorted(versions.items()):
        parts.append(
            f"COUNT {count} {card_rule_sentence(card_id, upgrades)}"
        )
    return " ; ".join(parts)


def relic_loadout_sentence(relic_ids: Iterable[str]) -> str:
    rules = [relic_rule_sentence(str(relic_id)) for relic_id in relic_ids]
    return " ; ".join(("RELIC_LOADOUT", *rules))


def validate_first_version_language(
    card_ids: Iterable[str], relic_ids: Iterable[str]
) -> None:
    missing_cards = sorted(set(card_ids) - RULE_CARD_IDS)
    missing_relics = sorted(set(relic_ids) - COMBAT_RELIC_RULES.keys())
    if missing_cards or missing_relics:
        raise ValueError(
            "incomplete first-version rule language: "
            f"cards={missing_cards}, relics={missing_relics}"
        )
