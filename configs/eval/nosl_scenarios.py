"""Pure public-state presets; no simulator, checkpoint or execution logic."""

# The first three fillers are Defends, making the initial hand contain no
# second attack/skill that would obscure the two cards under test.  The rest of
# the deck deliberately mixes ordinary attacks, skills, and powers so the next
# hand can have materially different HP outcomes; this is not a curse-only
# draw-pile probe.
FILLER_CARDS = (
    "DEFEND_RED", "DEFEND_RED", "DEFEND_RED", "ANGER", "CLEAVE", "WARCRY",
    "FLEX", "IRON_WAVE", "BODY_SLAM", "TRUE_GRIT", "SHRUG_IT_OFF", "CLASH",
    "THUNDERCLAP", "POMMEL_STRIKE", "TWIN_STRIKE", "CLOTHESLINE", "HAVOC",
    "HEADBUTT",
)
SCENARIOS = {
    "bash_plus_strike": (("BASH", 1), ("STRIKE_RED", 0)),
    "armaments_plus_strike": (("ARMAMENTS", 1), ("STRIKE_RED", 0)),
    # All remaining cards are attacks so Clash becomes legal immediately
    # after Bash removes the only skill from the hand.
    "bash_plus_clash": (("BASH", 0), ("CLASH", 0)),
}

FILLER_PROFILES = {
    "two_cards": (),
    # Keep the future draw pile non-empty while making the opening hand a
    # strict two-choice root.  Bludgeon is an Attack (so Clash remains legal)
    # but costs 3, hence it is masked at the controlled energy=2 root.  The
    # remaining ordinary cards stay in the draw pile and preserve NOSL future
    # variation after the first card.
    "strict_root_normal_draw": (
        "BLUDGEON", "BLUDGEON", "BLUDGEON", "BLUDGEON", "BLUDGEON",
        "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD",
        "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD",
        "STRIKE_RED", "DEFEND_RED", "IRON_WAVE", "POMMEL_STRIKE",
        "SHRUG_IT_OFF", "ANGER", "CLEAVE", "TRUE_GRIT", "FLEX",
        "THUNDERCLAP",
    ),
    "strikes": ("STRIKE_RED",) * 8,
    "mixed_attacks": (
        "STRIKE_RED", "CLEAVE", "IRON_WAVE", "BODY_SLAM",
        "HEADBUTT", "POMMEL_STRIKE", "TWIN_STRIKE", "THUNDERCLAP",
    ),
}
