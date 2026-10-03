"""Eager checkpoint values for legal native branches; no search or transitions."""
from itertools import islice

import numpy as np
from gymnasium import spaces

from env.core import encode, native_actions as C
from env.core.combat_value import VALUE_CONTRACT
from env.core.native_observation import encode_native_observation
from rl.policies.entity_combat import EntityCombatCore
from rl.search.native import forward_leaves
from rl.training.environments import observation_flags


class CheckpointLeafValue:
    """Frozen Entity AR combat-end-HP inference with at most 64 leaves per call.

    The caller supplies legal, nonterminal branches from its sampled world and
    root run metadata. This adapter neither advances nor redeterminizes them.
    Existing model-owned card-table lazy initialization remains ordinary eager
    inference; the adapter never installs or changes an inference backend.
    """
    batch_cap = 64

    def __init__(self, model):
        core = getattr(getattr(model, 'policy', None), 'core', None)
        if not isinstance(core, EntityCombatCore) or not core.combat_only:
            raise ValueError('checkpoint leaf value requires Entity combat-only policy')
        self.contract = C.policy_contract(model.observation_space)
        if self.contract != C.AUTOREGRESSIVE_CONTRACT or not core.autoregressive_actions:
            raise ValueError('checkpoint leaf value requires autoregressive-v2')
        if getattr(model.action_space, 'n', None) != C.AUTOREGRESSIVE_SIZE:
            raise ValueError('checkpoint leaf value action schema does not match autoregressive-v2')
        required = dict(nosl_value_contract=VALUE_CONTRACT,
            nosl_supervision_contract=core.training_objective,
            nosl_is_az=True, nosl_is_az_inner_only=True, hierarchical_combat_hp=True,
            nosl_is_az_end_turn_as_leaf=False, nosl_is_az_boss_transition_target=False)
        for name, expected in required.items():
            # Missing target metadata is unknown, not an implicit compatible
            # default. The supported TRAIN28 checkpoint saves all these fields.
            if not hasattr(model, name) or getattr(model, name) != expected:
                raise ValueError(f'checkpoint leaf value requires {name}={expected!r}')
        self.model = model
        self.flags = observation_flags(model)
        expected_space = encode.obs_space(**self.flags)
        expected_space.spaces['combat_legal'] = spaces.Box(
            0, 1, shape=(C.AUTOREGRESSIVE_WIDTH,), dtype=bool)
        if model.observation_space != expected_space:
            raise ValueError('checkpoint leaf value requires the supported observation schema')
        self._require_eval_mode()

    def _require_eval_mode(self):
        if self.model.policy.training or self.model.policy.core.training:
            raise ValueError('checkpoint leaf value requires eval mode; caller must freeze the policy')

    def manifest(self):
        return dict(adapter='entity-ar-checkpoint-leaf-v1',
            combat_action_contract=self.contract, value_contract=VALUE_CONTRACT,
            supervision_contract=self.model.policy.core.training_objective,
            combat_only=True, end_turn_as_leaf=False, boss_transition_target=False,
            observation_flags=dict(self.flags), leaf_inference='eager', batch_cap=self.batch_cap,
            selection_prefix=0, boundary='legal-next-decision', terminal='caller-owned')

    def evaluate(self, battles, *, root_global):
        """Return one finite normalized HP value per battle, in input order."""
        self._require_eval_mode()
        iterator = iter(battles)
        values = []
        while chunk := list(islice(iterator, self.batch_cap)):
            observations = []
            for battle in chunk:
                raw = battle.to_dict()
                if raw.get('post_enemy_turn_boundary', False):
                    raise ValueError('checkpoint leaf value rejects projected enemy boundaries')
                if int(raw['outcome']) != 0 or raw.get('escaped', False):
                    raise ValueError('checkpoint leaf value requires nonterminal branches')
                policy_ids = [aid for aid, _ in battle.autoregressive_policy_actions(0)]
                if not policy_ids:
                    raise ValueError('checkpoint leaf value requires a legal decision')
                raw['selection_prefix'] = 0
                observations.append(encode_native_observation(raw,
                    root_global=root_global, combat_action_contract=self.contract,
                    policy_ids=policy_ids, combat_only=True, **self.flags))
            batch = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
            _, result = forward_leaves(self.model, batch, compact=True, leaf_backend=None)
            result = np.asarray(result)
            if result.shape != (len(chunk),) or not np.isfinite(result).all():
                raise ValueError('checkpoint leaf value returned invalid shape or nonfinite values')
            if np.any((result < 0) | (result > 1)):
                raise ValueError('checkpoint leaf values must satisfy combat-end-hp-v1 range [0, 1]')
            values.extend(float(value) for value in result)
        return values
