"""Replay a whole real run inside the sim and diff at every step.

Reads an orchestrator.log, drives a sts_lightspeed GameContext through the same
decisions the player made (Neow / map / rewards / events / shop / rest /
combat), and diffs the sim against the real game after each command:
  - in combat:   full combat-state diff (engine + RNG alignment)
  - out of combat: overworld diff (deck / relics / hp / gold / floor)

The screen_type -> sim API mapping follows doc/0623_game_flow.md; this file does
not re-derive the game's structure, it only reads each frame for the player's
actual choice (which node, which card, which option) as replay input.

Usage: python replay.py <orchestrator.log>
"""

from collections import Counter
import json
import os
import re
import sys

# ---- load the sim module (built .so/.pyd lives in the build dir) ----
_BUILD = os.path.join(os.path.dirname(__file__), "..", "sim", "sts_lightspeed", "build")
# add_dll_directory is Windows-only (needed there to find the mingw runtime dlls
# sitting next to the .pyd); on Linux/mac the .so resolves its deps normally.
if hasattr(os, "add_dll_directory") and os.path.isdir(_BUILD):
    os.add_dll_directory(os.path.abspath(_BUILD))
sys.path.insert(0, os.path.abspath(_BUILD))
sys.path.insert(0, os.path.dirname(__file__))  # for `schemas`
import slaythespire as sts  # noqa: E402

from schemas.combat_state import (  # noqa: E402
    normalize_real, normalize_sim, diff,
    normalize_real_overworld, normalize_sim_overworld, diff_overworld,
    real_to_sim_card_enum, _monster_real, _norm_sim_monster,
)
from schemas.card_state import real_master_deck_state  # noqa: E402

# sim InputState enum (combat): 1 = PLAYER_NORMAL, 2 = CARD_SELECT
_CARD_SELECT = 2
# campfire option name -> fixed sim index (GameContext::chooseCampfireOption)
_CAMPFIRE = {"rest": 0, "smith": 1, "recall": 2, "lift": 3, "toke": 4, "dig": 5}


def parse_log(path):
    """Linear list of ('recv', game_state) / ('send', cmd) events.
    A command that produced an error (next RECV carries an `error` field and no
    game_state) did nothing to the game -- drop it."""
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace")
    events = []
    for line in text.splitlines():
        m = re.search(r"RECV (\{.*\})", line)
        if m:
            try:
                msg = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if "error" in msg:  # command was rejected -> undo the last send
                if events and events[-1][0] == "send":
                    events.pop()
                continue
            events.append(("recv", msg.get("game_state", {})))
            continue
        m = re.search(r"SEND (.*)", line)
        if m:
            events.append(("send", m.group(1).strip()))
    return events


def build_steps(events):
    """(initial_gs, [(cmd, post_gs), ...]). post_gs = the settled game_state
    right before the next command (i.e. the state this command resolved to)."""
    steps, cur_cmd, last_gs, initial = [], None, None, None
    for k, d in events:
        if k == "recv":
            if d:
                last_gs = d
                if initial is None:
                    initial = d
        else:
            if cur_cmd is not None:
                steps.append((cur_cmd, last_gs))
            cur_cmd = d
    if cur_cmd is not None:
        steps.append((cur_cmd, last_gs))
    return initial, steps


def _encounter_name(gc):
    tail = str(gc.encounter).split(".")[-1]
    return " ".join(w.capitalize() for w in tail.split("_"))


def _find_select_idx(gc, card_id, up):
    """Index in the sim's CARD_SELECT grid of the first card matching the real
    chosen card (by id+upgrade); duplicates are interchangeable for the diff."""
    from schemas.combat_state import _sim_card_id
    for i, (enum, u) in enumerate(gc.get_select_cards()):
        if _sim_card_id(enum) == card_id and u == up:
            return i
    raise ValueError(f"grid card {card_id!r}+{up} not found; sim screen="
                     f"{gc.screen_state_name} list={gc.get_select_cards()}")


class Driver:
    def __init__(self, gc):
        self.gc = gc
        self.bc = None          # active BattleContext, or None when out of combat
        self.neow_done = False

    # ---- combat ----
    def start_battle(self):
        self.bc = sts.create_battle(self.gc, _encounter_name(self.gc))

    def combat_cmd(self, cmd, pre=None):
        return _combat_cmd(self.bc, cmd, pre=pre)

    # ---- overworld ----
    def overworld_cmd(self, pre, cmd, post=None):
        st = pre.get("screen_type")
        ss = pre.get("screen_state", {}) or {}
        cl = pre.get("choice_list", []) or []
        parts = cmd.split()
        verb = parts[0].lower()
        idx = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
        gc = self.gc

        if verb in ("ready", "start", "click", "key", "wait", "state"):
            return  # setup / non-actions

        if verb == "potion":
            # POTION <use|discard> <slot> [target] held out of combat (e.g.
            # discarding a held potion to make room for a combat-reward potion).
            action, slot = parts[1].lower(), int(parts[2])
            if action == "discard" and 0 <= slot < len(gc.overworld_potions):
                gc.discard_potion(slot)
            # out-of-combat potion *use* is rare in these logs; ignore for now.
            return

        if st == "EVENT":
            if ss.get("event_id") == "Neow Event":
                if not self.neow_done and len(cl) > 1:
                    gc.choose_neow_option(idx)
                    self.neow_done = True
                # talk frames (1 option) are no-ops in the sim
            else:
                # Some events show a follow-up "you ate the X / leave" frame
                # after a choice resolves; in the sim the choice already called
                # regainControl() so the sim has left the event (now on the map).
                # The real "leave" CHOOSE on that frame is a no-op for the sim.
                if not gc.screen_state_name.endswith("EVENT_SCREEN"):
                    return
                real_ev = ss.get("event_id")
                if gc.cur_event_name != real_ev:
                    raise ValueError(f"event mismatch: real={real_ev!r} sim={gc.cur_event_name!r} "
                                     f"(event-selection RNG diverged)")
                gc.choose_event_option(idx)
        elif st == "CARD_REWARD":
            if verb == "skip":
                # CommunicationMod SKIP on a card-reward sub-screen reached via a
                # combat reward is a *cancel* (back out) -- the card stays
                # claimable -- not a true skip. Only treat it as a real skip if
                # the resulting frame no longer offers the card reward.
                post_cl = (post or {}).get("choice_list") or []
                post_st = (post or {}).get("screen_type")
                still_offered = post_st == "CARD_REWARD" or "card" in post_cl
                if not still_offered:
                    gc.skip_reward_cards()
            else:
                gc.pick_reward_card(gc.get_card_reward()[idx])
        elif st == "MAP":
            if cl[idx] == "boss":
                gc.choose_map_node(gc.cur_map_node_x)
            else:
                gc.choose_map_node(int(cl[idx].split("=")[1]))  # "x=N"
            if gc.screen_state_name.endswith("BATTLE"):
                self.start_battle()
        elif st == "COMBAT_REWARD":
            if verb == "proceed":
                gc.reward_proceed()
            else:
                label = cl[idx]
                if label in ("gold", "stolen_gold"):
                    gc.claim_gold(0)
                elif label == "relic":
                    gc.claim_relic(0)
                elif label == "potion":
                    gc.claim_potion(0)
                # 'card' -> the CARD_REWARD sub-screen follows; nothing to do here
        elif st == "GRID":
            if verb == "confirm":
                return
            card = ss["cards"][idx]
            gc.choose_select_card_option(_find_select_idx(gc, card["id"], card["upgrades"]))
        elif st == "REST":
            if verb == "proceed":
                if gc.screen_state == sts.ScreenState.REST_ROOM:
                    if 6 not in gc.legal_campfire_options():
                        raise ValueError("rest proceed while campfire actions remain available")
                    gc.choose_campfire_option(6)
                elif gc.screen_state == sts.ScreenState.REWARDS:
                    gc.reward_proceed()
                elif gc.screen_state != sts.ScreenState.MAP_SCREEN:
                    raise ValueError(f"rest proceed with pending sim screen {gc.screen_state_name}")
            else:
                gc.choose_campfire_option(_CAMPFIRE[cl[idx]])
        elif st == "CHEST":
            if verb == "proceed":
                if gc.screen_state == sts.ScreenState.TREASURE_ROOM:
                    gc.choose_treasure_room(False)
                elif gc.screen_state == sts.ScreenState.REWARDS:
                    gc.reward_proceed()
                elif gc.screen_state != sts.ScreenState.MAP_SCREEN:
                    raise ValueError(f"chest proceed with pending sim screen {gc.screen_state_name}")
            else:
                gc.choose_treasure_room(True)
        elif st == "BOSS_REWARD":
            # boss-relic pick from the act boss treasure chest
            if verb != "proceed":
                gc.choose_boss_relic(idx)
        elif st == "SHOP_ROOM":
            if verb == "proceed":
                gc.shop_proceed()
            # 'choose 0' (enter merchant) -> sim is already in the shop room
        elif st == "SHOP_SCREEN":
            if verb == "leave":
                return  # leaving handled by the following SHOP_ROOM proceed
            label = cl[idx]
            if label == "purge":
                gc.shop_remove()
            else:
                # choice_list lists only affordable entries; 'purge' (the card
                # removal service) is one such entry and drops out once its cost
                # exceeds the player's gold, just like an unaffordable item. Map
                # idx to its rank among the real (non-purge) item entries.
                item_pos = idx - sum(1 for j in range(idx) if cl[j] == "purge")
                self._shop_buy(ss, pre.get("gold", 0), item_pos)
        elif st == "GAME_OVER":
            pass
        # NONE with no combat: transient, ignore

    def _shop_buy(self, ss, gold, item_pos):
        """Buy the item at choice_list position item_pos. CommunicationMod's shop
        choice_list lists only *affordable* items, ordered cards, relics, potions;
        map that position back to the item's index in its (full, unfiltered) list,
        which is what shop_buy_* expects."""
        cards, relics, potions = ss.get("cards", []), ss.get("relics", []), ss.get("potions", [])
        affordable = ([("card", i) for i, c in enumerate(cards) if c["price"] <= gold]
                      + [("relic", i) for i, r in enumerate(relics) if r["price"] <= gold]
                      + [("potion", i) for i, p in enumerate(potions) if p["price"] <= gold])
        kind, full_i = affordable[item_pos]
        if kind == "card":
            self.gc.shop_buy_card(full_i)
        elif kind == "relic":
            self.gc.shop_buy_relic(full_i)
        else:
            self.gc.shop_buy_potion(full_i)


def _report(label, mismatches, verbose):
    if not mismatches:
        if verbose:
            print(f"  [OK] {label}")
        return 0
    print(f"  [DIFF] {label}  ({len(mismatches)} mismatch)")
    for path, real, sim in mismatches:
        print(f"      {path}: real={real!r}  sim={sim!r}")
    return 1


# ===================== decoupled per-combat validation =====================
# Out-of-combat RNG (events/map) is not yet aligned (see DEVNOTES), so instead
# of chaining the whole run we validate every combat independently: force the
# sim gc to the real pre-combat snapshot (deck/relics/hp/floor) read from the
# log, force the encounter, then replay + diff. This isolates the combat engine
# (whose RNG is Random(seed+floor), independent of event RNG) across the run.

def extract_combats(initial, steps):
    """Group the command stream into combats. Each combat = {init, steps}:
    init = the first in-combat frame (carries the master deck/relics/hp/floor),
    steps = [(cmd, post_frame), ...] for the duration of that combat."""
    combats, pre, cur = [], initial, None
    for cmd, post in steps:
        if pre is not None and pre.get("combat_state") and cur is None:
            cur = {"init": pre, "steps": []}
        if cur is not None:
            cur["steps"].append((cmd, post))
            if post is None or not post.get("combat_state"):
                combats.append(cur)
                cur = None
        pre = post
    if cur is not None:
        combats.append(cur)
    return combats


def build_encounter_map(tmpgc):
    """sorted real-monster-id tuple -> encounter name, by forcing every known
    encounter on a throwaway gc and reading back its monster composition."""
    emap = {}
    for name in sts.encounter_names():
        try:
            bc = sts.create_battle(tmpgc, name)
            ids = tuple(sorted(m["id"] for m in normalize_sim(bc.to_dict())["monsters"]
                               if m["id"] != "PAD"))
        except Exception:
            continue
        emap.setdefault(ids, name)
    return emap


def _sync_gc(seed, asc, init):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, asc)
    gc.floor_num = init["floor"]
    gc.cur_hp = init["current_hp"]
    gc.max_hp = init["max_hp"]
    gc.gold = init["gold"]
    gc.set_deck_state([
        dict(card, id=real_to_sim_card_enum(card["id"]))
        for card in real_master_deck_state(init["deck"])
    ])
    gc.set_relics([r["id"] for r in init["relics"]])
    # if this combat is the seed's burning (emerald-key) elite, sit the gc on
    # that map node (floor == burningEliteY+1) so create_battle applies the
    # elite buff (e.g. +Strength) -- otherwise the forced battle is un-buffed.
    if init["floor"] == gc.burning_elite_y + 1:
        gc.cur_map_node_x = gc.burning_elite_x
        gc.cur_map_node_y = gc.burning_elite_y
    return gc


def validate_combats(events, verbose):
    initial, steps = build_steps(events)
    seed = initial["seed"]
    asc = initial.get("ascension_level", 0)
    combats = extract_combats(initial, steps)
    emap = build_encounter_map(sts.GameContext(sts.CharacterClass.IRONCLAD, seed, asc))
    print(f"seed={seed} ascension={asc} -- {len(combats)} combats\n")

    total_diff = 0
    for c in combats:
        init = c["init"]
        floor = init["floor"]
        key = tuple(sorted(m["id"] for m in init["combat_state"]["monsters"]))
        encounter = emap.get(key)
        if encounter is None:
            print(f"floor {floor}: no encounter matches {key} -- skipped")
            continue

        gc = _sync_gc(seed, asc, init)
        bc = sts.create_battle(gc, encounter)
        start_diff = diff(normalize_real(init), normalize_sim(bc.to_dict()))
        # If the opening hand/draw order differs (the reconstructed deck order is
        # not the game's true internal order -- see DEVNOTES), positional replay
        # would play the wrong cards and corrupt the sim. Report and skip.
        if any(k.startswith(("hand", "draw", "discard", "exhaust")) for k, _, _ in start_diff):
            _report(f"floor {floor} {encounter!r} start", start_diff, verbose=True)
            print(f"floor {floor:>2} {encounter:<16} [START-DESYNC: deck order]")
            total_diff += len(start_diff)
            continue
        n = _report(f"floor {floor} {encounter!r} start", start_diff, verbose)
        steps_ok = 0
        pre = init
        for cmd, post in c["steps"]:
            try:
                _combat_cmd(bc, cmd, pre=pre)
            except Exception as e:
                print(f"  floor {floor}: error on {cmd!r}: {e}")
                n += 1
                break
            # HAND_SELECT frames are mid-resolution (the real game defers the
            # card's effect until CONFIRM, the sim resolves on select) -> not a
            # settled state, skip diffing it.
            if (post is not None and post.get("combat_state")
                    and post.get("screen_type") != "HAND_SELECT"):
                n += _report(f"floor {floor} after {cmd!r}",
                             diff(normalize_real(post), normalize_sim(bc.to_dict())), verbose)
                steps_ok += 1
            pre = post
        status = "OK" if n == 0 else f"{n} DIFF"
        print(f"floor {floor:>2} {encounter:<16} [{status}]  ({steps_ok} steps)")
        total_diff += n
    print(f"\ndone. {total_diff} total diffs across {len(combats)} combats.")


def _replay_target(bc, pre, wire_index):
    """Resolve a client handle by matching the complete live entity multiset.

    Neither backend guarantees the same physical slots. Legacy recordings have
    no shared monster UUID, so indistinguishable targets must fail explicitly.
    """
    if not pre or not isinstance(pre.get("combat_state"), dict):
        raise ValueError("targeted replay requires the pre-command combat frame")
    records = pre["combat_state"].get("monsters")
    if not isinstance(records, list) or not 0 <= wire_index < len(records):
        raise ValueError(f"invalid client monster target: {wire_index}")
    real = [_monster_real(m) for m in records]
    native = [_norm_sim_monster(m) for m in bc.to_dict()["monsters"]]
    def alive(m):
        return not m["gone"] and m["hp"] > 0
    def identity(m):
        return (m["id"], m["hp"], m["max_hp"], m["block"],
                m["base_damage"], m["hits"], tuple(sorted(m["powers"].items())))
    if not alive(real[wire_index]):
        raise ValueError(f"client target {wire_index} is not a live monster")
    if Counter(identity(m) for m in real if alive(m)) != Counter(
            identity(m) for m in native if alive(m)):
        raise ValueError("client/native monster states disagree before targeted replay")
    target = identity(real[wire_index])
    matches = [i for i, m in enumerate(native) if alive(m) and identity(m) == target]
    if len(matches) != 1:
        raise ValueError(f"ambiguous client monster target {wire_index}; shared entity identity required")
    return matches[0]


def _combat_cmd(bc, cmd, pre=None):
    parts = cmd.split()
    if not parts:
        raise ValueError("empty combat command")
    verb = parts[0].lower()
    if verb in ("confirm", "wait", "state"):
        return
    if verb == "potion":
        operation, slot = parts[1].lower(), int(parts[2])
        if operation not in ("use", "discard"):
            raise ValueError(f"unknown potion operation: {operation}")
        # Historical recordings may contain a rejected out-of-range command.
        if slot < 0 or slot >= len(bc.potions):
            return
        if operation == "discard":
            bc.execute(sts.Action(sts.ActionType.POTION, slot, 6))
            return
        option = next((p for p in bc.legal_potions() if p["slot"] == slot), None)
        if option is None:
            raise ValueError(f"potion {slot} cannot be used in the current state")
        target = 0
        if option["requires_target"]:
            if len(parts) != 4:
                raise ValueError("targeted potion replay requires a client target")
            target = _replay_target(bc, pre, int(parts[3]))
            if target not in option["targets"]:
                raise ValueError("mapped potion target is not legal")
        bc.execute(sts.Action(sts.ActionType.POTION, slot, target))
        return
    if verb == "play":
        src = int(parts[1]) - 1
        target = _replay_target(bc, pre, int(parts[2])) if len(parts) >= 3 else 0
        requested = (sts.ActionType.CARD, src, target)
    elif verb == "end":
        requested = (sts.ActionType.END_TURN, 0, 0)
    elif verb == "choose":
        requested = (sts.ActionType.SINGLE_CARD_SELECT, int(parts[1]), 0)
    else:
        raise ValueError(f"unhandled combat command: {cmd!r}")
    def signature(a):
        return (a.action_type, a.source_idx, a.target_idx)
    # Match before constructing an Action: its packed integers can wrap.
    action = next((a for a in bc.legal_actions() if signature(a) == requested), None)
    if action is None:
        raise ValueError(f"recorded combat action is not legal: {cmd!r}")
    bc.execute(action)


def run_natural(events, verbose):
    """Drive the gc through the whole run by the player's real decisions. Stops
    at the first out-of-combat RNG desync (event/map). Used to study that gap."""
    initial, steps = build_steps(events)
    seed = initial["seed"]
    asc = initial.get("ascension_level", 0)
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, asc)
    drv = Driver(gc)
    print(f"seed={seed} ascension={asc} -- replaying {len(steps)} commands\n")

    pre, n_diff = initial, 0
    for cmd, post in steps:
        if pre is None:
            pre = post
            continue
        # CommunicationMod debug/UI commands (help, click, menu, ...) are not
        # gameplay actions and appear as trailing noise once the recording's real
        # play has ended -- skip them so they don't drive a phantom battle.
        if cmd.split()[0].lower() in ("help", "click", "menu"):
            pre = post if post is not None else pre
            continue
        try:
            if drv.bc is not None:
                drv.combat_cmd(cmd, pre=pre)
            else:
                drv.overworld_cmd(pre, cmd, post)
        except Exception as e:
            print(f"\n  [DESYNC] cannot replay {cmd!r} at floor {pre.get('floor')} "
                  f"({pre.get('screen_type')}): {e}")
            print("  -> sim diverged from the real run before this point; stopping.")
            break
        if drv.bc is not None and drv.bc.is_battle_over:
            drv.bc.exit_battle(gc)
            drv.bc = None
        # Selection-in-progress frames (HAND_SELECT in combat, GRID out of it)
        # are mid-resolution: the real game defers the card's effect to CONFIRM
        # while the sim resolves on select, so these frames mismatch transiently
        # and reconverge at the next settled frame -- don't diff them.
        post_st = post.get("screen_type") if post else None
        if post is None:
            pass
        elif post.get("combat_state"):
            if post_st != "HAND_SELECT" and drv.bc:
                n_diff += _report(f"f{post.get('floor')} {cmd!r}",
                                  diff(normalize_real(post), normalize_sim(drv.bc.to_dict())),
                                  verbose)
        elif post_st not in (None, "NONE", "GAME_OVER", "GRID"):
            n_diff += _report(f"f{post.get('floor')} {post_st} after {cmd!r}",
                              diff_overworld(normalize_real_overworld(post),
                                             normalize_sim_overworld(gc)), verbose)
        pre = post
    print(f"\ndone. {n_diff} diffs. final: floor={gc.floor_num} hp={gc.cur_hp} "
          f"deck={len(gc.deck_ids())} outcome={gc.outcome}")


def main():
    log_path = sys.argv[1]
    verbose = "-v" in sys.argv
    events = parse_log(log_path)
    if "--natural" in sys.argv:
        run_natural(events, verbose)
    else:
        validate_combats(events, verbose)


if __name__ == "__main__":
    main()
