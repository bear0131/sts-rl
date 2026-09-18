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
import slaythespire as sts
from env.core.native_actions import complete_actions
from env.search.nosl_sampler import NOSLSampler


PLANNER_VERSION = 'ironclad-turn-v1'
_END = 4 << 29
_DRAW_SELECT = frozenset(('SECRET_TECHNIQUE', 'SECRET_WEAPON', 'SEEK'))
_JUNK = frozenset(('WOUND', 'DAZED', 'BURN', 'VOID', 'SLIMED', 'INJURY',
    'DOUBT', 'SHAME', 'PAIN', 'REGRET', 'DECAY', 'NORMALITY', 'WRITHE'))


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


def state_value(raw, *, partial=False):
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

    def choose(self, battle, *, state_nonce: int, check=None) -> PlanResult:
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
                score, counts = self._root(sample, action, quota, check)
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

    def _root(self, branch, bits, quota, check):
        stats = dict(native_transitions=0, expanded_nodes=0, fallback_leaves=0,
                     budget_exhausted=False)
        frontier = []
        serial = 0
        best = -math.inf

        def take(state, action, depth):
            nonlocal serial, best
            stats['native_transitions'] += 1
            if action == _END:
                boundary = state.preview_end_turn_result(action)
                best = max(best, state_value(boundary))
                return
            child = state.clone()
            child.execute(sts.Action(action))
            raw = child.to_dict()
            if raw['outcome'] != 0:
                best = max(best, state_value(raw))
                return
            serial += 1
            heapq.heappush(frontier, (-state_value(raw, partial=True), serial, depth, child))

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
            best = -frontier[0][0] if frontier else state_value(branch.to_dict(), partial=True)
        return best, stats
