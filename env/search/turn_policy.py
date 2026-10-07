"""Fixed model-free combat policy used by upper-only training experiments."""
from __future__ import annotations

from dataclasses import asdict

from env.search.turn_planner import NativeTurnPlanner, PLANNER_VERSION, TurnPlannerConfig
from rl.evaluation.baselines.combat_rules import visible_rule_native_action


VALIDATED_RULEBASE_VERSION = 'ironclad-turn-v8-awakened-phase'


def rulebase_turn_policy_version(config: TurnPlannerConfig) -> str:
    """Return the audited policy identity, rather than the planner code version."""
    if (
        config.worlds == 2
        and config.node_budget == 8192
        and config.beam_width == 8
        and config.max_depth == 16
        and config.lookahead_turns == 1
        and config.leaf_profile == 'legacy-v8'
        and config.master_nonce == 20260918
    ):
        return VALIDATED_RULEBASE_VERSION
    return f'{PLANNER_VERSION}+custom'


class RulebaseTurnPolicy:
    """The validated v8 turn planner with a public-only Runic Dome fallback."""

    def __init__(self, config: TurnPlannerConfig):
        self.config = config
        self.planner = NativeTurnPlanner(config)

    def choose(self, env, *, state_nonce: int, selected=()):
        public_relics = tuple(item['id'] for item in env.thick.gc.relic_state())
        if 'Runic Dome' in public_relics:
            bits, diagnostics = visible_rule_native_action(
                env, env._info(), selected=selected,
            )
            return bits, dict(diagnostics, planner_dome_fallback=1)
        plan = self.planner.choose(
            env.thick.bc,
            state_nonce=int(state_nonce),
            public_relics=public_relics,
        )
        return int(plan.action_bits), dict(plan.diagnostics, planner_dome_fallback=0)

    def manifest(self):
        return {
            'name': 'turn-planner',
            'version': rulebase_turn_policy_version(self.config),
            'config': asdict(self.config),
            'dome': 'visible-rule-fallback',
        }
