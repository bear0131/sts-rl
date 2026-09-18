"""DecisionStrategy adapter for the model-free native turn planner."""
from dataclasses import asdict

import numpy as np

from env.core.native_actions import complete_actions
from env.search.turn_planner import NativeTurnPlanner, PLANNER_VERSION, TurnPlannerConfig
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
from rl.evaluation.strategies import Decision


class TurnPlannerStrategy:
    name = 'turn-planner'
    version = PLANNER_VERSION
    combat_action_contract = 'native-v1'

    def __init__(self, config=TurnPlannerConfig(), *, trace=None):
        self.config = config
        self.planner = NativeTurnPlanner(config)
        self.fallback = CombatRuleStrategy()
        self.trace = trace

    def manifest(self):
        return dict(config=asdict(self.config), horizon='current-player-turn-plus-enemy-phase',
            information='independent-NOSL-futures', dome='visible-rule-fallback',
            leaf='handwritten-state-value', draw_selection='redeterminize-then-remap-native-entity',
            resources='conservative-no-output-guard-with-public-relics',
            enemy_progress='native-threshold-slime-phase-potential',
            value_horizon='shared-root-public-enemy-hp-and-strength',
            terminal_priority='per-world-death-live-win',
            world_aggregation='lexicographic-fewer-deaths-more-wins-mean-heuristic',
            winning_tie_break='fewest-actions-to-equally-valued-sampled-wins')

    def decide(self, requests, *, check=lambda: None):
        decisions = []
        for request in requests:
            check()
            env = request.env.unwrapped
            if env.combat_action_contract != self.combat_action_contract:
                raise ValueError('turn planner requires native-v1 environment')
            # The native clone contains the actual current hidden intent with
            # Dome. Do not expose it to transition search or leaf evaluation.
            public_relics = tuple(r['id'] for r in env.thick.gc.relic_state())
            dome = 'Runic Dome' in public_relics
            if dome:
                if np.any(env.action_masks()):
                    chosen = self.fallback.decide([request], check=check)[0]
                    decision = Decision(chosen.action, chosen.source,
                        dict(chosen.diagnostics, planner_dome_fallback=1))
                else:
                    decision = Decision(None, self.name, {'planner_dome_fallback': 1},
                        native_action=complete_actions(env.thick.bc)[0])
            else:
                # Per-episode step is an opaque decision counter, not game seed
                # or simulator RNG. Identical public inputs at the same counter
                # get identical hidden samples regardless of the actual future.
                plan = self.planner.choose(env.thick.bc, state_nonce=request.step, check=check,
                                           public_relics=public_relics)
                decision = Decision(None, self.name,
                    {key: value for key, value in plan.diagnostics.items()
                     if isinstance(value, (int, float))}, native_action=plan.action_bits)
            if self.trace:
                raw = env.thick.bc.to_dict()
                raw['draw'] = sorted(raw['draw'])
                if dome:
                    # Pure fallback already handles hidden intent; trace only
                    # player/cards in this unsupported search configuration.
                    raw['monsters'] = []
                self.trace(dict(kind='turn-planner-decision', seed=request.seed, step=request.step,
                    floor=int(env.thick.gc.floor_num), state=raw,
                    action=decision.action, native_action=decision.native_action,
                    source=decision.source, diagnostics=(decision.diagnostics if dome else plan.diagnostics)))
            decisions.append(decision)
        return decisions
