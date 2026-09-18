"""Bounded, model-free combat planning over independent hidden-state samples.

Only the current player turn is searched. Native transitions own all card and
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
from env.core.native_actions import complete_actions
from env.search.nosl_sampler import NOSLSampler


PLANNER_VERSION = 'ironclad-turn-v2'
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

    def __post_init__(self):
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


def state_value(raw, *, partial=False, public_relics=None):
    """HP, remaining threats, lasting growth, and reusable combat resources.

    At enemy boundaries actual loss already includes block, Weak, Intangible,
    potions, end-turn damage and enemy mechanics. Partial values only order the
    bounded frontier; completed turn leaves decide between root alternatives.
    """
    p = raw['player']
    if p['hp'] <= 0 or raw['outcome'] == 2:
        return -100000.0
    powers = _powers(p)
    monsters = [m for m in raw['monsters'] if m['hp'] > 0 and not m['gone']]
    remaining = sum(m['hp'] for m in monsters)
    horizon = min(5., max(1., remaining / max(25., 25 + 3 * powers.get('strength', 0))))
    value = (1.3 * p['hp'] + 12 * math.log(max(.01, p['hp'] / p['max_hp']))
             + 1.8 * p['max_hp'] + .025 * p['gold'])
    value += 5 * sum(x not in ('INVALID', 'EMPTY_POTION_SLOT') for x in raw['potions'])
    if raw['outcome'] == 1:
        return value + 1000
    if no_remaining_damage_source(raw, public_relics):
        # Losing every remaining output channel is worse than ordinary HP or
        # damage differences, but remains separate from an actual native death.
        value -= 10000
    value -= .26 * remaining + 10 * len(monsters)
    for m in monsters:
        mp = _powers(m)
        value -= 1.6 * horizon * mp.get('strength', 0)
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
        # Root draw selections refer to a public displayed index, which NOSL
        # shuffles. Remap the same native entity without executing it first:
        # pending exhaust/draw triggers may otherwise reveal the live future.
        select_draw = (original['input_state'] == 2
                       and original['select']['context'] in _DRAW_SELECT)
        budget = max(self.config.node_budget, 2 * len(roots))
        quota = budget // len(roots)
        values = {bits: [] for bits in roots}
        stats = dict(native_transitions=0, expanded_nodes=0, worlds_completed=0,
            budget_exhausted=False, effective_node_budget=budget, root_actions=len(roots),
            effective_total_budget=budget * self.config.worlds, fallback_leaves=0)
        for world in range(self.config.worlds):
            sample = self.sampler.clone_and_redeterminize(
                battle, state_nonce=state_nonce, simulation_index=world)
            for bits in roots:
                check()
                action = sample.remap_draw_selection_from(battle, bits) if select_draw else bits
                score, counts = self._root(sample, action, quota, check, public_relics)
                values[bits].append(score)
                for key in ('native_transitions', 'expanded_nodes', 'fallback_leaves'):
                    stats[key] += counts[key]
                stats['budget_exhausted'] |= counts['budget_exhausted']
            stats['worlds_completed'] += 1
        mean = {bits: sum(scores) / len(scores) for bits, scores in values.items()}
        selected = max(roots, key=lambda bits: (mean[bits], -bits))
        stats.update(root_values={str(k): v for k, v in mean.items()},
            search_seconds=time.monotonic() - started, search_decisions=1)
        return PlanResult(selected, stats)

    def _root(self, branch, bits, quota, check, public_relics=None):
        stats = dict(native_transitions=0, expanded_nodes=0, fallback_leaves=0,
                     budget_exhausted=False)
        frontier = []
        serial = 0
        best = -math.inf
        best_partial = -math.inf

        def take(state, action, depth):
            nonlocal serial, best, best_partial
            stats['native_transitions'] += 1
            child = state.clone()
            # End Turn can pause for Codex; a normal card can force the turn
            # to end (Time Eater). Keep the cloned boundary flag across either
            # path and continue intermediate selections without a new draw.
            raw = child.execute_to_enemy_boundary(action)
            if raw['outcome'] != 0 or raw['post_enemy_turn_boundary']:
                best = max(best, state_value(raw, public_relics=public_relics))
                return
            partial = state_value(raw, partial=True, public_relics=public_relics)
            best_partial = max(best_partial, partial)
            serial += 1
            heapq.heappush(frontier, (-partial, serial, depth, child))

        take(branch, bits, 1)
        while frontier and stats['native_transitions'] < quota:
            check()
            _, _, depth, node = heapq.heappop(frontier)
            stats['expanded_nodes'] += 1
            legal = complete_actions(node)
            # Every expanded normal node first contributes a feasible complete
            # turn; incomplete prefixes must not win because of unused energy.
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
        if best == -math.inf:
            # A pending selection may outlast the allocated expansion. This is
            # explicitly approximate; never execute extra unbudgeted rollouts.
            stats['fallback_leaves'] += 1
            best = best_partial
        return best, stats
