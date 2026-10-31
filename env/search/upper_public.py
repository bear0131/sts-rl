"""Bounded upper lookahead over audited public, deterministic transitions.

The builder never owns a neural model and never executes an unmodelled random
transition. Such nodes remain value leaves. This is a simulator-only capability;
the whitelist is deliberately narrower than the game's complete legal mask.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from env.core import action_space as A, encode
from env.core.outer_return import strategic_reward

VERSION = 'upper-public-aux-v1'
PHASES = ('combat_reward', 'shop', 'event', 'rest', 'card_select')
UPPER_KEYS = encode.OVERWORLD_OBSERVATION_KEYS | frozenset().union(
    *encode.OVERWORLD_PHASE_OBSERVATION_KEYS.values())

# GameContext::chooseEventOption: these options neither draw a hidden outcome
# nor enter combat. Opening a remove/upgrade grid is itself deterministic.
EVENT_OPTIONS = {
    'The Cleric': frozenset((0, 1, 2)),
    'Purifier': frozenset((0, 1)),
    'Upgrade Shrine': frozenset((0, 1)),
    'Back to Basics': frozenset((0, 1)),
    'Accursed Blacksmith': frozenset((0, 2)),
    'World of Goop': frozenset((0, 1)),
}


@dataclass(frozen=True)
class SearchBudget:
    depth: int = 3
    transitions: int = 64
    root_actions: int = 32
    seconds: float = 2.

    def __post_init__(self):
        if (any(type(n) is not int or n < 1 for n in
                (self.depth, self.transitions, self.root_actions))
                or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError('invalid public upper search budget')


def supported_actions(state, legal, *, room):
    """Use public current-screen semantics only, never RNG state or future stock.

`room` is the already entered room, not a future unknown map node. Card grids
need their public origin as well as context: REMOVE alone does not prove its
continuation is deterministic (e.g. a Neow delayed random reward).
"""
    phase, options = state['phase'], state.get('options', {})
    relics = set(state.get('global', {}).get('relics', ()))
    result = []
    for action in legal:
        segment, index = A.segment_of(int(action))
        safe = False
        if phase == 'combat_reward':
            safe = segment in ('REWARD_GOLD', 'REWARD_POTION')
            # Only the first pack is public; removing it must not reveal an
            # already generated but still hidden subsequent pack to the critic.
            safe |= segment in ('REWARD_CARD', 'REWARD_SKIP') and options.get('card_count', 0) == 1
            # Boss PROCEED can generate unseen relics/next-act map. Other event
            # and relic callbacks are not assumed to be a plain return to map.
            safe |= segment == 'PROCEED' and room in ('MONSTER', 'ELITE')
        elif phase == 'shop':
            safe = segment in ('SHOP_REMOVE', 'PROCEED')
            safe |= (segment in ('SHOP_CARD', 'SHOP_POTION') and 'The Courier' not in relics)
        elif phase == 'event':
            safe = segment == 'EVENT_OPT' and index in EVENT_OPTIONS.get(options.get('event'), ())
        elif phase == 'rest':
            safe = segment == 'PROCEED' or (segment == 'CAMPFIRE' and index in (1, 2, 3, 4))
            safe |= segment == 'CAMPFIRE' and index == 0 and 'Dream Catcher' not in relics
        elif phase == 'card_select':
            origin = state.get('public_event')
            safe = (segment == 'PICK_CARD'
                    and options.get('select', {}).get('context') in ('REMOVE', 'UPGRADE')
                    and (room in ('SHOP', 'REST') or origin in EVENT_OPTIONS))
        if safe:
            result.append(int(action))
    return result


def _actions(env):
    return supported_actions(env.thick.obs_canonical(), np.flatnonzero(env.action_masks()),
                             room=str(env.thick.gc.cur_room).split('.')[-1])


def build_tree(env, budget=SearchBudget()):
    """Return owned CPU leaves and edges; the actual trajectory is unchanged.

    Expansion is level-synchronous and all-or-nothing per level. A node ceiling
    cannot give early root actions a deeper lookahead than later actions.
    Timeouts/censoring discard the entire tree, never turn it into a loss label.
    """
    started = time.perf_counter()
    def result(reason, nodes=None, transitions=0):
        return dict(version=VERSION, phase=env._phase(), reason=reason,
                    nodes=nodes or [], transitions=transitions,
                    seconds=time.perf_counter() - started)
    if env._phase() not in PHASES or env.thick.is_over:
        return result('unsupported_phase')
    root_actions = _actions(env)
    if len(root_actions) < 2:
        return result('insufficient_actions')
    if len(root_actions) > min(budget.root_actions, budget.transitions):
        return result('root_budget')
    nodes = [dict(action=-1, reward=0., terminal=False, children=[], observation=None)]
    live = {0: env}
    frontier = [0]
    transitions = 0
    for depth in range(budget.depth):
        expansions = [(i, root_actions if i == 0 else _actions(live[i])) for i in frontier]
        needed = sum(len(actions) for _, actions in expansions)
        if not needed or transitions + needed > budget.transitions:
            break
        next_frontier = []
        for parent, actions in expansions:
            source = live[parent]
            for action in actions:
                if time.perf_counter() - started >= budget.seconds:
                    return result('timeout', transitions=transitions)
                branch = source.fork_run()
                before = branch.thick.run_stats()
                obs, _, done, cut, info = branch.step(action)
                transitions += 1
                if cut or info.get('dead_screen'):
                    return result('censored', transitions=transitions)
                if not done and branch._phase() == 'combat':
                    raise RuntimeError('public upper action unexpectedly entered combat')
                reward = strategic_reward(before['act'], before['floor'], info['run_stats'],
                                          done=done, censored=False)
                child = len(nodes)
                nodes[parent]['children'].append(child)
                nodes.append(dict(action=action, reward=reward, terminal=bool(done), children=[],
                    observation=None if done else {k: np.array(v, copy=True)
                                                  for k, v in obs.items() if k in UPPER_KEYS}))
                if not done:
                    live[child] = branch
                    next_frontier.append(child)
            if actions:
                nodes[parent]['observation'] = None
                del live[parent]
        frontier = next_frontier
    if time.perf_counter() - started >= budget.seconds:
        return result('timeout', transitions=transitions)
    return result('ok', nodes, transitions)


def backup(tree, leaf_values, gamma=1.):
    """Max backup over the same audited action subset; no terminal bootstrap."""
    if tree['reason'] != 'ok':
        raise ValueError('discarded tree has no training target')
    nodes = tree['nodes']
    values = np.zeros(len(nodes), dtype=np.float64)
    expected = {i for i, n in enumerate(nodes) if not n['terminal'] and not n['children']}
    if expected != set(leaf_values):
        raise ValueError('leaf values must cover exactly the nonterminal frontier')
    for i in range(len(nodes) - 1, -1, -1):
        n = nodes[i]
        tail = (max(values[c] for c in n['children']) if n['children']
                else 0. if n['terminal'] else float(leaf_values[i]))
        values[i] = n['reward'] + gamma * tail
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite upper search value')
    children = nodes[0]['children']
    return np.array([nodes[c]['action'] for c in children], dtype=np.int64), values[children]
