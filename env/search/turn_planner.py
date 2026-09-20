"""Bounded, model-free combat planning over independent hidden-state samples.

One or two player turns are searched. Native transitions own all card and
enemy mechanics; the scalar leaf heuristic expresses preferences, not a learned
win probability. Samples never use the live future RNG or draw order. Per-root
budgets prevent beam pruning from excluding an entire first-action alternative.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

from env.core import _BUILD  # noqa: F401  (explicit native build bootstrap)
import slaythespire as sts
from env.core.native_actions import complete_actions
from env.search.nosl_sampler import NOSLSampler


PLANNER_VERSION = 'ironclad-turn-v11-ritual-pressure'
_END = 4 << 29
_DRAW_SELECT = frozenset(('SECRET_TECHNIQUE', 'SECRET_WEAPON', 'SEEK'))
_JUNK = frozenset(('WOUND', 'DAZED', 'BURN', 'VOID', 'SLIMED', 'INJURY',
    'DOUBT', 'SHAME', 'PAIN', 'REGRET', 'DECAY', 'NORMALITY', 'WRITHE'))

# A closed set is intentional: an unknown card may generate an Attack, recover
# an exhausted card, or deal conditional damage. Never call it harmless merely
# because it is not one of Ironclad's familiar attacks. Body Slam, Flame Barrier,
# Infernal Blade, Exhume and all Powers therefore remain potential damage sources.
_NO_DAMAGE_CARDS = _JUNK | frozenset((
    'DEFEND_RED', 'ARMAMENTS', 'BATTLE_TRANCE', 'BLOODLETTING', 'BURNING_PACT',
    'DISARM', 'DUAL_WIELD', 'ENTRENCH', 'FLEX', 'HAVOC', 'INTIMIDATE',
    'POWER_THROUGH', 'RAGE', 'SECOND_WIND', 'SEEING_RED', 'SENTINEL',
    'SHOCKWAVE', 'SHRUG_IT_OFF', 'TRUE_GRIT', 'WARCRY', 'IMPERVIOUS',
    'GHOSTLY_ARMOR', 'LIMIT_BREAK', 'ASCENDERS_BANE', 'CLUMSY', 'CURSE_OF_THE_BELL',
    'NECRONOMICURSE', 'PARASITE', 'PRIDE', 'WRITHE',
))
_NON_DAMAGE_POWERS = frozenset((
    'strength', 'dexterity', 'artifact', 'weak', 'vulnerable', 'frail',
    'metallicize', 'platedarmor', 'barricade', 'berserk', 'demonform',
    'feelnopain', 'darkembrace', 'corruption', 'evolve', 'rupture', 'rage',
    'nodraw', 'drawreduction', 'entangled', 'confused', 'intangible', 'buffer',
    'energized', 'noblock', 'hex', 'constricted', 'drawcardnextturn',
    'doubletap', 'doubledamage', 'vigor', 'losestrength', 'dexteritydown',
))
_DAMAGE_OR_CARD_RELICS = frozenset((
    'mercuryhourglass', 'bronzescales', 'charonsashes', 'letteropener',
    'stonecalendar', 'deadbranch', 'nilryscodex', 'tingsha', 'toolbox',
))


def no_remaining_damage_source(raw, public_relics=None):
    """Recognize a narrow, proven resource dead end; unknown means possible.

    This does not certify that an existing channel can finish the fight. A
    remaining Attack, damage/creation Power, potion, damage relic, future bomb,
    enemy Poison/Fading, or unresolved selection conservatively disables it.
    Relics are supplied by the public run state; BattleContext.to_dict omits
    them, so callers without that context must not infer their absence.
    """
    if public_relics is None or raw['outcome'] != 0 or raw['input_state'] == 2:
        return False
    cards = raw['hand'] + raw['draw'] + raw['discard']
    if any(card[0] not in _NO_DAMAGE_CARDS for card in cards):
        return False
    powers = _powers(raw['player'])
    if any(value > 0 and name not in _NON_DAMAGE_POWERS for name, value in powers.items()):
        return False
    counters = raw['player'].get('combat_counters', {})
    if any(counters.get(f'bomb_{n}', 0) > 0 for n in (1, 2, 3)):
        return False
    if any(p not in ('INVALID', 'EMPTY_POTION_SLOT') for p in raw['potions']):
        return False
    relics = {''.join(c for c in name.lower() if c.isalnum()) for name in public_relics}
    if relics & _DAMAGE_OR_CARD_RELICS:
        return False
    for monster in raw['monsters']:
        if monster['hp'] <= 0 or monster['gone']:
            continue
        mp = _powers(monster)
        if any(mp.get(name, 0) > 0 for name in ('poison', 'choked', 'fading', 'explosive')):
            return False
        if monster['id'] in ('TRANSIENT', 'DAGGER', 'EXPLODER'):
            return False
    return True


@dataclass(frozen=True)
class TurnPlannerConfig:
    worlds: int = 2
    node_budget: int = 512
    beam_width: int = 8
    max_depth: int = 16
    master_nonce: int = 20260918
    lookahead_turns: int = 1

    def __post_init__(self):
        if self.lookahead_turns not in (1, 2):
            raise ValueError('lookahead_turns must be 1 or 2')
        for key in ('worlds', 'node_budget', 'beam_width', 'max_depth'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')


@dataclass(frozen=True)
class PlanResult:
    action_bits: int
    diagnostics: dict


def _powers(raw):
    return {name.replace(' ', '').replace('_', '').lower(): value
            for name, value in raw.get('powers', ())}


def power_horizon(raw):
    """Estimate payoff duration once from the public decision root."""
    remaining = sum(m['hp'] for m in raw['monsters'] if m['hp'] > 0 and not m['gone'])
    strength = _powers(raw['player']).get('strength', 0)
    return min(5., max(1., remaining / max(25., 25 + 3 * strength)))



def scaling_damage_premiums(raw, base_damage_weight=.26):
    """Root-anchored marginal urgency for visible Ritual's growing attacks.

    Eighteen damage is three ordinary Strikes, adjusted by public Strength.
    The marginal cost of an extra turn near the estimated kill time grows with
    Ritual. This deliberately rough, capped prior supplements the base HP cost;
    it is neither a learned value nor a forecast of actual future draws/block.
    Slot and ID jointly identify the root monster so transformed bodies do not
    inherit another enemy's premium.
    """
    damage_rate = max(9., 18 + 3 * _powers(raw['player']).get('strength', 0))
    premiums = {}
    for slot, monster in enumerate(raw['monsters']):
        ritual = _powers(monster).get('ritual', 0)
        if monster['hp'] <= 0 or monster['gone'] or ritual <= 0:
            continue
        remaining_turns = monster['hp'] / damage_rate
        attack = max(0, monster['adjusted_damage']) * max(0, monster['hits'])
        marginal = min(2., 1.3 * (attack + ritual * remaining_turns) / damage_rate)
        premium = max(0., marginal - base_damage_weight)
        if premium:
            premiums[(slot, monster['id'])] = premium
    return premiums


def enemy_burden(monsters):
    """Approximate remaining damage and bodies through public monster phases.

    Assume a split at the native half-HP threshold, counting the HP inherited
    by the children. Below that threshold a split inherits the current HP.
    This is a progress potential, not minimum lethal damage: a burst kill can
    bypass slime phases entirely. Odd max HP follows native integer division.
    Awakened One's public Curiosity marks its first life; half_dead marks the
    zero-HP rebirth interval. Both still owe another life, approximated by
    visible max HP: exact at native A0 (300), up to 20 HP low at higher
    ascensions where native first-life HP is sampled but rebirth is fixed320.
    """
    damage, bodies = 0, 0
    for monster in monsters:
        hp = monster['hp']
        if monster['id'] == 'AWAKENED_ONE':
            if monster.get('half_dead', False):
                # Native gone includes half-dead; rebirth is neither a corpse
                # nor a victory. Retain the expected second-life burden.
                damage += monster['max_hp']
                bodies += 1
                continue
            if hp > 0 and not monster['gone'] and _powers(monster).get('curiosity', 0) > 0:
                damage += hp + monster['max_hp']
                bodies += 1
                continue
        if hp <= 0 or monster['gone']:
            continue
        split_hp = min(hp, monster['max_hp'] // 2)
        if monster['id'] in ('ACID_SLIME_L', 'SPIKE_SLIME_L'):
            damage += hp + split_hp
            bodies += 2
        elif monster['id'] == 'SLIME_BOSS':
            damage += hp + split_hp + 2 * (split_hp // 2)
            bodies += 4
        else:
            damage += hp
            bodies += 1
    return damage, bodies


def state_value(raw, *, partial=False, public_relics=None, horizon=None, damage_premiums=None):
    """HP, remaining threats, lasting growth, and reusable combat resources.

    At enemy boundaries actual loss already includes block, Weak, Intangible,
    potions, end-turn damage and enemy mechanics. Partial values only order the
    bounded frontier; completed turn leaves decide between root alternatives.
    The planner supplies one root-anchored horizon to every leaf. Recomputing it
    after dealing damage can subtract more power value than the damage gains.
    Standalone callers may omit it to evaluate an isolated state as before.
    """
    p = raw['player']
    if p['hp'] <= 0 or raw['outcome'] == 2:
        return -100000.0
    powers = _powers(p)
    monsters = [m for m in raw['monsters'] if m['hp'] > 0 and not m['gone']]
    remaining, future_bodies = enemy_burden(raw['monsters'])
    horizon = power_horizon(raw) if horizon is None else horizon
    value = (1.3 * p['hp'] + 12 * math.log(max(.01, p['hp'] / p['max_hp']))
             + 1.8 * p['max_hp'] + .025 * p['gold'])
    value += 5 * sum(x not in ('INVALID', 'EMPTY_POTION_SLOT') for x in raw['potions'])
    if raw['outcome'] == 1:
        return value + 1000
    if no_remaining_damage_source(raw, public_relics):
        # Losing every remaining output channel is worse than ordinary HP or
        # damage differences, but remains separate from an actual native death.
        value -= 10000
    value -= .26 * remaining + 10 * future_bodies
    if damage_premiums:
        value -= sum(damage_premiums.get((slot, monster['id']), 0.) * monster['hp']
                     for slot, monster in enumerate(raw['monsters'])
                     if monster['hp'] > 0 and not monster['gone'])
    for m in monsters:
        mp = _powers(m)
        # Angry is reactive growth on every hit, not free growth from waiting.
        # The native enemy boundary already charges this turn's stronger hit.
        # Applying the generic multi-turn Strength penalty again can make
        # every safe Strike worse than blocking forever against Mad Gremlin.
        strength_weight = .5 if mp.get('angry', 0) > 0 else 1.6 * horizon
        value -= strength_weight * mp.get('strength', 0)
        value -= horizon * mp.get('ritual', 0)
        value += 3 * min(horizon, mp.get('weak', 0))
        value += 3 * min(horizon, mp.get('vulnerable', 0))
        value -= .6 * min(m['block'], 30)
    value += horizon * (2.3 * powers.get('strength', 0)
        + 1.8 * powers.get('dexterity', 0)
        + .8 * powers.get('metallicize', 0)
        + .8 * powers.get('platedarmor', 0)
        + 1.2 * powers.get('combust', 0)
        + 2 * powers.get('berserk', 0))
    value += 1.2 * horizon * horizon * powers.get('demonform', 0)
    value += horizon * (2 * powers.get('feelnopain', 0)
        + 5 * powers.get('darkembrace', 0) + 6 * powers.get('corruption', 0))
    value -= 2 * min(horizon, powers.get('weak', 0))
    value -= 2 * min(horizon, powers.get('vulnerable', 0))
    if powers.get('barricade', 0):
        value += .55 * min(150, p['block']) + 4 * horizon
    piles = raw['hand'] + raw['draw'] + raw['discard']
    value -= .7 * sum(c[0] in _JUNK for c in piles)
    if partial:
        incoming = sum(max(0, m['adjusted_damage']) * max(0, m['hits']) for m in monsters)
        value -= 1.3 * max(0, incoming - p['block'])
        value += 2 * p['energy'] + .4 * len(raw['hand'])
    return value


def leaf_value(raw, *, public_relics=None, horizon=None, damage_premiums=None):
    """Lexicographic terminal class and the unchanged scalar heuristic.

    The heuristic alone is not a terminal utility: a huge live power can exceed
    the historical victory bonus. Keep it for ordering within a terminal class.
    """
    rank = -1 if raw['player']['hp'] <= 0 or raw['outcome'] == 2 else 1 if raw['outcome'] == 1 else 0
    value = state_value(raw, public_relics=public_relics, horizon=horizon,
                        damage_premiums=damage_premiums)
    if not math.isfinite(value):
        raise ValueError('planner leaf heuristic must be finite')
    return rank, value


def aggregate_world_values(values):
    """Prefer fewer sampled deaths, then more wins, then mean old heuristic.

    This is an explicit conservative risk preference, not a win probability or
    an equivalent rescaling. Equal-world roots are compared lexicographically.
    All-live roots retain exactly the previous arithmetic mean and ordering.
    """
    if not values:
        raise ValueError('world values must not be empty')
    if any(rank not in (-1, 0, 1) or not math.isfinite(value) for rank, value in values):
        raise ValueError('world outcomes and heuristic values must be valid and finite')
    mean = sum(value for _, value in values) / len(values)
    if not math.isfinite(mean):
        # Avoid overflowing the sum of otherwise finite extreme fixture values.
        mean = math.fsum(value / len(values) for _, value in values)
    if not math.isfinite(mean):
        raise ValueError('world mean must be finite')
    return (-sum(rank == -1 for rank, _ in values),
            sum(rank == 1 for rank, _ in values), mean)


class NativeTurnPlanner:
    """Receding-horizon best-first beams, averaged across sampled futures.

    The caller must expose the current enemy intent (the strategy adapter falls
    back to public rules with Runic Dome). This is determinization search, with optimistic within-sample continuations;
    it is not an exact information-set solver. Only the first action is executed
    before observing and replanning. No transposition merge is attempted because
    raw observations omit simulator state needed to prove transition equivalence.
    """
    def __init__(self, config: TurnPlannerConfig = TurnPlannerConfig()):
        self.config = config
        self.sampler = NOSLSampler(config.master_nonce)

    def choose(self, battle, *, state_nonce: int, check=None, public_relics=None) -> PlanResult:
        check = check or (lambda: None)
        check()
        started = time.monotonic()
        roots = complete_actions(battle)
        if not roots:
            raise ValueError('turn planner requires a live combat decision')
        original = battle.to_dict()
        if original['post_enemy_turn_boundary']:
            raise ValueError('turn planner requires a live decision, not a projected enemy boundary')
        final_player_turn = original['turn'] + self.config.lookahead_turns - 1
        # A shared public reference makes the payoff duration comparable across
        # damage, growth and enemy debuff branches, worlds, and partial leaves.
        horizon = power_horizon(original)
        damage_premiums = scaling_damage_premiums(original)
        # Root draw selections refer to a public displayed index, which NOSL
        # shuffles. Remap the same native entity without executing it first:
        # pending exhaust/draw triggers may otherwise reveal the live future.
        select_draw = (original['input_state'] == 2
                       and original['select']['context'] in _DRAW_SELECT)
        budget = max(self.config.node_budget, (self.config.lookahead_turns + 1) * len(roots))
        quota = budget // len(roots)
        values = {bits: [] for bits in roots}
        unresolved = {bits: 0 for bits in roots}
        winning_steps = {bits: 0 for bits in roots}
        completed_steps = {bits: 0 for bits in roots}
        stats = dict(native_transitions=0, expanded_nodes=0, worlds_completed=0,
            budget_exhausted=False, effective_node_budget=budget, root_actions=len(roots),
            effective_total_budget=budget * self.config.worlds, fallback_leaves=0,
            second_turn_entries=0, completed_horizon_leaves=0, max_action_depth=0)
        for world in range(self.config.worlds):
            sample = self.sampler.clone_and_redeterminize(
                battle, state_nonce=state_nonce, simulation_index=world)
            for bits in roots:
                check()
                action = sample.remap_draw_selection_from(battle, bits) if select_draw else bits
                score, counts = self._root(sample, action, quota, check, public_relics,
                                           horizon, final_player_turn, damage_premiums)
                values[bits].append(score)
                unresolved[bits] += counts['fallback_leaves']
                winning_steps[bits] += counts['winning_steps']
                completed_steps[bits] += counts['completed_steps']
                for key in ('native_transitions', 'expanded_nodes', 'fallback_leaves',
                            'second_turn_entries', 'completed_horizon_leaves'):
                    stats[key] += counts[key]
                stats['max_action_depth'] = max(stats['max_action_depth'], counts['max_action_depth'])
                stats['budget_exhausted'] |= counts['budget_exhausted']
            stats['worlds_completed'] += 1
        priorities = {bits: aggregate_world_values(scores) for bits, scores in values.items()}
        # Only break an exact tie in death count, win count and mean utility.
        # Otherwise receding-horizon planning can forever postpone an equally
        # good proved win behind a harmless, self-replacing zero-cost card.
        # Retain the existing winning-length preference, then extend progress
        # to equally valued surviving boundaries. A partial estimate cannot
        # establish an equivalent completed plan; disable the new tie whenever
        # any tied alternative is unresolved or includes a sampled death.
        primary = {bits: (priorities[bits], -winning_steps[bits]) for bits in roots}
        best = max(primary.values())
        tied = [bits for bits in roots if primary[bits] == best]
        complete_safe_tie = all(unresolved[bits] == 0 and priorities[bits][0] == 0
                                for bits in tied)
        selected = max(tied, key=lambda bits: (
            -completed_steps[bits] if complete_safe_tie else 0, -bits))
        stats.update(root_values={str(k): v[2] for k, v in priorities.items()},
            root_outcomes={str(k): dict(deaths=-v[0], wins=v[1], live=self.config.worlds+v[0]-v[1],
                                       unresolved=unresolved[k],
                                       boundary_live=self.config.worlds+v[0]-v[1]-unresolved[k],
                                       heuristic_mean=v[2], completed_steps_total=completed_steps[k],
                                       winning_steps_total=winning_steps[k],
                                       winning_steps_mean=winning_steps[k]/v[1] if v[1] else 0.)
                           for k, v in priorities.items()},
            root_damage_premiums={f'{slot}:{enemy}': value
                                  for (slot, enemy), value in damage_premiums.items()},
            search_seconds=time.monotonic() - started, search_decisions=1)
        return PlanResult(selected, stats)

    def _root(self, branch, bits, quota, check, public_relics, horizon, final_player_turn,
              damage_premiums=None):
        stats = dict(native_transitions=0, expanded_nodes=0, fallback_leaves=0,
                     budget_exhausted=False, second_turn_entries=0,
                     completed_horizon_leaves=0, max_action_depth=0, winning_steps=0,
                     completed_steps=0)
        frontier = []
        serial = 0
        best = None
        best_partial = None

        def take(state, action, depth):
            nonlocal serial, best, best_partial
            stats['native_transitions'] += 1
            stats['max_action_depth'] = max(stats['max_action_depth'], depth)
            child = state.clone()
            # End Turn can pause for Codex; a normal card can force the turn
            # to end (Time Eater). Keep the cloned boundary flag across either
            # path and continue intermediate selections without a new draw.
            if self.config.lookahead_turns == 1 or state.turn >= final_player_turn:
                raw = child.execute_to_enemy_boundary(action)
            else:
                # Stay in the same independently sampled world. Normal execute
                # owns enemy actions, next-turn powers/draw and pending choices;
                # redeterminizing here would erase known Headbutt top cards.
                child.execute(sts.Action(action))
                raw = child.to_dict()
                if raw['post_enemy_turn_boundary']:
                    raise ValueError('two-turn planning cannot resume an already armed search boundary')
                if raw['outcome'] == 0 and raw['turn'] >= final_player_turn:
                    stats['second_turn_entries'] += 1
            if raw['outcome'] != 0 or raw['post_enemy_turn_boundary']:
                stats['completed_horizon_leaves'] += int(raw['post_enemy_turn_boundary'])
                value = leaf_value(raw, public_relics=public_relics, horizon=horizon,
                                   damage_premiums=damage_premiums)
                if best is None or value > best:
                    best = value
                    stats['winning_steps'] = depth if value[0] == 1 else 0
                    stats['completed_steps'] = depth if value[0] >= 0 else 0
                elif value == best and value[0] >= 0:
                    stats['completed_steps'] = min(stats['completed_steps'], depth)
                    if value[0] == 1:
                        stats['winning_steps'] = min(stats['winning_steps'], depth)
                return
            partial = state_value(raw, partial=True, public_relics=public_relics, horizon=horizon,
                                  damage_premiums=damage_premiums)
            if not math.isfinite(partial):
                raise ValueError('planner partial heuristic must be finite')
            best_partial = partial if best_partial is None else max(best_partial, partial)
            serial += 1
            heapq.heappush(frontier, (-partial, serial, depth, child))

        take(branch, bits, 1)
        while frontier and stats['native_transitions'] < quota:
            check()
            _, _, depth, node = heapq.heappop(frontier)
            stats['expanded_nodes'] += 1
            legal = complete_actions(node)
            # Two-turn depth counts every action across both turns, including
            # END and pending selections; do not sneak a last END past its cap.
            if self.config.lookahead_turns == 2 and depth >= self.config.max_depth:
                continue
            # END prioritizes a feasible continuation. In two-turn mode the
            # first END can start another player turn rather than finish a leaf.
            if _END in legal:
                take(node, _END, depth + 1)
            if depth >= self.config.max_depth:
                continue
            for action in legal:
                if action == _END:
                    continue
                if stats['native_transitions'] >= quota:
                    stats['budget_exhausted'] = True
                    break
                take(node, action, depth + 1)
            if len(frontier) > self.config.beam_width:
                frontier = heapq.nsmallest(self.config.beam_width, frontier)
                heapq.heapify(frontier)
        stats['budget_exhausted'] |= bool(frontier)
        if best is None:
            # A pending selection may outlast the allocated expansion. This is
            # explicitly approximate; never execute extra unbudgeted rollouts.
            stats['fallback_leaves'] += 1
            if best_partial is None:
                raise RuntimeError('planner produced no complete or partial leaf')
            best = (0, best_partial)
        return best, stats
