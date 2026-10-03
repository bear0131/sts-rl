"""DecisionStrategy adapters for bounded native turn planning."""
from dataclasses import asdict, replace

import numpy as np

from env.core.native_actions import complete_actions
from env.search.turn_planner import NativeTurnPlanner, PLANNER_VERSION, TurnPlannerConfig
import slaythespire as sts
from schemas.combat_state import normalize_sim_global
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
from rl.evaluation.strategies import Decision, PhaseStrategy


class RiskTurnPlannerStrategy(PhaseStrategy):
    """Frozen public encounter routing; dispatch remains owned by PhaseStrategy."""
    elite_encounters = ('GREMLIN_NOB', 'LAGAVULIN', 'NEMESIS')

    def __init__(self, config, *, trace=None):
        if config.lookahead_turns != 2:
            raise ValueError('boss-elite-v1 requires planner-turns 2')
        super().__init__(TurnPlannerStrategy(replace(config, lookahead_turns=1)),
                         {'combat': TurnPlannerStrategy(config)},
                         predicates={'combat': self.is_risk_combat})
        self.name = 'turn-planner:boss-elite-v1'
        self.trace = trace

    @classmethod
    def is_risk_combat(cls, request):
        gc = request.env.unwrapped.thick.gc
        return gc.cur_room == sts.Room.BOSS or (
            gc.cur_room == sts.Room.ELITE and
            gc.encounter in tuple(getattr(sts.MonsterEncounter, name)
                                  for name in cls.elite_encounters))

    def manifest(self):
        return dict(turn_scope='boss-elite-v1', default_lookahead_turns=1,
                    selected_lookahead_turns=2, predicate=dict(
                        source='public-gc.cur_room-and-gc.encounter',
                        all_rooms=['BOSS'], encounter_room='ELITE',
                        encounters=list(self.elite_encounters)))

    @property
    def trace(self):
        return self.default.trace

    @trace.setter
    def trace(self, trace):
        for strategy in self._unique_strategies():
            strategy.trace = trace


class TurnPlannerStrategy:
    name = 'turn-planner'
    version = PLANNER_VERSION
    combat_action_contract = 'native-v1'

    def __init__(self, config=TurnPlannerConfig(), *, trace=None, checkpoint_value=None):
        if checkpoint_value is not None and (config.leaf_boundary != 'next-decision'
                                            or config.lookahead_turns != 1):
            raise ValueError('checkpoint planner value requires one-turn next-decision leaves')
        self.config = config
        self.planner = NativeTurnPlanner(config)
        self.fallback = CombatRuleStrategy()
        self.trace = trace
        self.checkpoint_value = checkpoint_value

    def manifest(self):
        return dict(config=asdict(self.config), turn_scope='all', horizon=(
            'next-legal-player-decision' if self.config.leaf_boundary == 'next-decision' else
            'current-player-turn-plus-enemy-phase' if self.config.lookahead_turns == 1
            else 'two-player-turns-through-second-enemy-phase'),
            information='independent-NOSL-futures', dome='visible-rule-fallback',
            leaf=self.config.leaf_profile, draw_selection='redeterminize-then-remap-native-entity',
            leaf_value=('heuristic' if self.checkpoint_value is None else self.checkpoint_value.manifest()),
            resources='conservative-no-output-guard-with-public-relics',
            enemy_progress='public-slime-and-awakened-phase-potential',
            value_horizon='shared-root-public-enemy-hp-and-strength',
            terminal_priority='per-world-death-live-win',
            world_aggregation=('lexicographic-fewer-deaths-more-wins-mean-heuristic'
                               if self.checkpoint_value is None else
                               'lexicographic-fewer-deaths-more-wins-mean-combat-end-hp-value'),
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
                value_options = {}
                if self.checkpoint_value is not None:
                    root_global = normalize_sim_global(env.thick.gc)
                    value_options['leaf_evaluator'] = lambda battles: self.checkpoint_value.evaluate(
                        battles, root_global=root_global)
                plan = self.planner.choose(env.thick.bc, state_nonce=request.step, check=check,
                                           public_relics=public_relics, **value_options)
                decision = Decision(None, self.name,
                    {key: value for key, value in plan.diagnostics.items()
                     if isinstance(value, (int, float)) and key != 'max_action_depth'},
                    native_action=plan.action_bits)
            # Only additive counts belong in runner diagnostics. Chosen horizon
            # is per-request trace metadata, including unsupported Dome search.
            decision.diagnostics.update(
                planner_turn1_decisions=int(not dome and self.config.lookahead_turns == 1),
                planner_turn2_decisions=int(not dome and self.config.lookahead_turns == 2))
            if dome:
                decision.diagnostics.update(effective_node_budget=0, effective_total_budget=0,
                                            native_transitions=0, second_turn_entries=0)
            if self.trace:
                raw = env.thick.bc.to_dict()
                raw['draw'] = sorted(raw['draw'])
                if dome:
                    # Pure fallback already handles hidden intent; trace only
                    # player/cards in this unsupported search configuration.
                    raw['monsters'] = []
                self.trace(dict(kind='turn-planner-decision', seed=request.seed, step=request.step,
                    floor=int(env.thick.gc.floor_num), state=raw,
                    room=str(env.thick.gc.cur_room).split('.')[-1],
                    encounter=str(env.thick.gc.encounter).split('.')[-1],
                    lookahead_turns=self.config.lookahead_turns,
                    leaf_boundary=self.config.leaf_boundary,
                    leaf_value='checkpoint' if self.checkpoint_value is not None else 'heuristic',
                    turn2_applied=not dome and self.config.lookahead_turns == 2,
                    action=decision.action, native_action=decision.native_action,
                    source=decision.source, diagnostics=(decision.diagnostics if dome else
                        dict(plan.diagnostics, **decision.diagnostics))))
            decisions.append(decision)
        return decisions
