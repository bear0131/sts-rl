"""Human-readable rendering of canonical combat state + actions.

Like encode.py and actions.py, this module is PURE — it consumes the canonical
combat dict (bridge.schemas.combat_state.normalize_sim/normalize_real output)
and the abstract action space, and imports neither slaythespire nor numpy. So
the SAME renderer works for the sim backend and a future real-game backend, and
it can be exercised in tests without the sim built.

This is a debugging/teaching view, not part of the obs the policy sees (that is
the lossy, scaled vector from encode.py). Card upgrades show as a trailing '+'.
"""

from . import actions


# ----------------------------- helpers -----------------------------

def _card_label(card):
    """{'id','up'} -> 'Strike_R' or 'Strike_R+2' (just '+' for a single upgrade)."""
    up = card.get("up", 0)
    if not up:
        return card["id"]
    return f"{card['id']}+" if up == 1 else f"{card['id']}+{up}"


def _powers_str(powers):
    """{power_id: amt} -> '[Strength 2, Vulnerable 1]' (empty -> '')."""
    if not powers:
        return ""
    inner = ", ".join(f"{name} {amt}" for name, amt in powers.items())
    return f"[{inner}]"


def _hand_labels(hand):
    return [_card_label(c) for c in hand]


# --------------------------- state render ---------------------------

def render_state(canon):
    """canonical combat dict -> multi-line human-readable string."""
    lines = []
    p = canon["player"]
    lines.append(f"== turn {canon['turn']} " + "=" * 30)

    pw = _powers_str(p["powers"])
    pw = f"  {pw}" if pw else ""
    lines.append(
        f"You    HP {p['hp']}/{p['max_hp']}  Block {p['block']}  "
        f"Energy {p['energy']}{pw}"
    )

    # Monsters keep their slot index — that index is the target_idx used by the
    # action space, so show it explicitly. Dead/gone ones are marked, not hidden,
    # so the numbering matches what encode/actions saw.
    lines.append("Foes")
    if not canon["monsters"]:
        lines.append("  (none)")
    for i, m in enumerate(canon["monsters"]):
        if m["gone"]:
            lines.append(f"  #{i} {m['id']:<16} (gone)")
            continue
        if m["base_damage"] > 0:
            intent = f"atk {m.get('adjusted_damage', m['base_damage'])}x{m['hits']}"
        else:
            intent = "       "  # unknown/non-attack intent (sim doesn't export it)
        mw = _powers_str(m["powers"])
        mw = f"  {mw}" if mw else ""
        lines.append(
            f"  #{i} {m['id']:<16} {m['hp']:>3}/{m['max_hp']:<3}  "
            f"Block {m['block']:<2}  {intent}{mw}"
        )

    # Hand is positional: slot i is what "play hand[i]" addresses.
    hand = _hand_labels(canon["hand"])
    if hand:
        hand_str = " ".join(f"[{i}]{lab}" for i, lab in enumerate(hand))
    else:
        hand_str = "(empty)"
    lines.append(f"Hand   {hand_str}")
    lines.append(
        f"Piles  draw {len(canon['draw'])}  "
        f"discard {len(canon['discard'])}  exhaust {len(canon['exhaust'])}"
    )
    return "\n".join(lines)


# ------------------------- overworld render -------------------------

def render_overworld(canon):
    """canonical overworld dict (normalize_sim_overworld + env-filled options) ->
    multi-line human-readable string. Branches on the semantic phase."""
    phase = canon.get("phase", "?")
    lines = [f"== {phase} " + "=" * 30]
    lines.append(
        f"floor {canon.get('floor')}  HP {canon['hp']}/{canon['max_hp']}  "
        f"gold {canon['gold']}  deck {len(canon['deck'])}  relics {len(canon['relics'])}"
    )
    opts = canon.get("options", {}) or {}

    if phase == "map":
        nodes = opts.get("nodes", [])
        shown = "  ".join(f"x={x}:{room}" for x, room in nodes) or "(none)"
        lines.append(f"Map next: {shown}")
    elif phase == "combat_reward":
        bits = []
        if opts.get("gold"):    bits.append(f"gold {opts['gold']}")
        if opts.get("relics"):  bits.append(f"relics {opts['relics']}")
        if opts.get("potions"): bits.append(f"potions {opts['potions']}")
        lines.append("Rewards: " + ("  ".join(bits) if bits else "(none)"))
        if opts.get("card_count"):
            cards = opts.get("cards", [])
            shown = " ".join(f"[{j}]{cid}{'+' if up else ''}"
                             for j, (cid, up) in enumerate(cards)) or f"x{opts['card_count']}"
            lines.append(f"Card reward: {shown}")
    elif phase == "card_select":
        cards = opts.get("cards", [])
        shown = " ".join(f"[{k}]{cid}{'+' if up else ''}"
                         for k, (cid, up) in enumerate(cards)) or "(none)"
        lines.append(f"Choose: {shown}")
    elif phase == "event":
        lines.append(f"Event: {opts.get('event', '?')}")
    return "\n".join(lines)


def render_action(action_id, canon=None):
    """Abstract action id -> human-readable string.

    If `canon` is given, fills in the actual card name / monster id from the
    current state (e.g. 'play Bash -> #0 Cultist'); otherwise renders structurally
    (e.g. 'play hand[2] -> foe#0')."""
    kind = actions.decode(int(action_id))

    if kind[0] == actions.KIND_END:
        return "end turn"

    if kind[0] == actions.KIND_PLAY:
        _, h, t = kind
        card = _hand_card_label(canon, h)
        if t is None:
            return f"play {card}"
        tgt = _foe_label(canon, t)
        return f"play {card} -> {tgt}"

    if kind[0] == actions.KIND_SELECT:
        return f"select card #{kind[1]}"

    return f"<action {action_id}>"


def _hand_card_label(canon, h):
    if canon is not None and h < len(canon["hand"]):
        return _card_label(canon["hand"][h])
    return f"hand[{h}]"


def _foe_label(canon, t):
    if canon is not None and t < len(canon["monsters"]):
        return f"#{t} {canon['monsters'][t]['id']}"
    return f"foe#{t}"
