"""Trainable combat actor/critic and an immutable upper teacher."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from stable_baselines3.common.save_util import json_to_data

from env.core import vocab as V
from env.core.outer_return import OBJECTIVE, CONTEXT_SCHEMA, CONTEXT_KEYS, SCALAR_OBJECTIVES, FINITE_OBJECTIVE
from env.core.finite_curves import FiniteCurveBank
from rl.policies.entity_combat import (
    EntityCombatConfig, EntityCombatCore, EntityCombatNetwork, EntityCombatPolicy,
)
from rl.policies.entity_joint import EntityJointCore, UpperConfig
from rl.policies.frozen_upper import FrozenUpperTeacher
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS
from rl.algorithms.combat_distribution import CONTRACT as COMBAT_OBJECTIVE, MEAN_CONTRACT, OUTCOME_CONTRACTS
from rl.policies.combat_distribution import CombatMeanHead


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_initialization_material(directory, *, manifest_sha256, source_sha256):
    """Validate immutable files before decoding the source SB3 space records."""
    directory = Path(directory)
    if _sha(directory / 'manifest.json') != manifest_sha256:
        raise ValueError('initialization manifest SHA256 mismatch')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['schema'] != 'entity-joint-initialization-material-v1' or
            manifest['source_model_sha256'] != source_sha256 or
            manifest['optimizer_imported'] or manifest['replay_imported']):
        raise ValueError('initialization source/objective contract mismatch')
    for name in ('constructor.json', 'upper.pt', 'lower-init.pt', 'frozen-card-vectors.pt'):
        if _sha(directory / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'initialization file SHA256 mismatch: {name}')
    constructor = json_to_data((directory / 'constructor.json').read_text())
    kwargs = constructor['policy_kwargs']
    config = dict(kwargs['entity_config'], card_vector_checkpoint=str(directory / 'frozen-card-vectors.pt'))
    source = EntityJointCore(constructor['observation_space'], EntityCombatConfig(**config),
                             UpperConfig(**kwargs['upper_config']))
    upper = torch.load(directory / 'upper.pt', map_location='cpu', weights_only=True)
    lower = torch.load(directory / 'lower-init.pt', map_location='cpu', weights_only=True)
    state = {key.removeprefix('core.'): value for key, value in dict(upper, **lower).items()}
    missing, unexpected = source.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith(('combat.distribution.', 'combat.transition.')) for key in missing):
        raise ValueError(f'initialization architecture mismatch: {missing}, {unexpected}')
    return source, constructor['observation_space']


class OuterCombatNetwork(EntityCombatNetwork):
    def __init__(self, source, source_space, space, *, initialization='random', objective=OBJECTIVE):
        if objective not in (*SCALAR_OBJECTIVES, *OUTCOME_CONTRACTS):
            raise ValueError('unsupported frozen-upper lower objective')
        if objective in OUTCOME_CONTRACTS and initialization != 'random':
            raise ValueError('TRAIN47 context control requires a fresh random lower')
        self.objective = objective
        if initialization not in ('random', '876'):
            raise ValueError('lower initialization must be random or 876')
        config = replace(source.config, auxiliary_loss_coef=0.) if objective in SCALAR_OBJECTIVES else source.config
        if objective == MEAN_CONTRACT:
            config = replace(config, auxiliary_loss_coef=0., mean_loss_coef=0.)
        # Scalar objectives only encode the boss flag. Outcome objectives retain
        # the source's fight-local Boss utility when combining their predictions.
        super().__init__(source_space, config, card_embedding=deepcopy(source.card))
        if initialization == '876':
            state = {key: value for key, value in source.state_dict().items()
                     if not key.startswith(('distribution.', 'transition.'))}
            missing, unexpected = self.load_state_dict(state, strict=False)
            if unexpected or any(not key.startswith(('distribution.', 'transition.')) for key in missing):
                raise ValueError('lower warm-start architecture mismatch')
        if objective in (*SCALAR_OBJECTIVES, MEAN_CONTRACT):
            del self.distribution, self.transition
        width = config.width
        self.context_boss = nn.Embedding(int(space['next_boss'].high.max()) + 1, 16)
        self.context_character = nn.Embedding(int(space['character'].high.max()) + 1, 8)
        self.context_deck = nn.Linear(int(self.card.embedding_dim) + 2, width)
        # Public act/floor/run HP/ascension/keys plus boss/character/deck summary.
        self.context = nn.Sequential(nn.Linear(width + 33, width), nn.GELU(), nn.Linear(width, width))
        if initialization == '876':
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)
        if objective in SCALAR_OBJECTIVES:
            self.scalar = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        elif objective == MEAN_CONTRACT:
            self.means = CombatMeanHead(width)

    def encode(self, obs, active_relic_width=None, *, card_table=None):
        state, blocks = super().encode(obs, active_relic_width, card_table=card_table)
        context = self.context_features(obs, state.dtype, card_table=card_table)
        return state + self.context(context), blocks

    def context_features(self, obs, dtype, *, card_table=None):
        table = self._encode_card_table() if card_table is None else card_table
        ids = obs['deck_ids'].long()
        vectors = table[ids, obs['deck_upgrades'].gt(0).long()]
        cards = self.context_deck(torch.cat((vectors, obs['deck_upgrades'][..., None],
                                            obs['outer_deck_misc'][..., None]), -1).to(dtype))
        mask = ids.ne(0)[..., None]
        pooled = (cards * mask).sum(1) / mask.sum(1).clamp_min(1)
        # Public pending GRID choices remain in the source pile until the last
        # pick. Encode them separately so actor and critic see selection progress.
        selected = obs['selected_card_ids'].long()
        upgrades = obs['selected_card_upgrades'][..., None]
        chosen = self.context_deck(torch.cat((table[selected, upgrades.squeeze(-1).gt(0).long()],
                                             upgrades, torch.zeros_like(upgrades)), -1).to(dtype))
        pooled = pooled + (chosen * selected.ne(0)[..., None]).sum(1)
        scalars = torch.cat([obs[key].float() for key in ('act', 'floor', 'run_hp', 'ascension', 'run_keys')], -1)
        context = torch.cat((pooled, scalars, self.context_boss(obs['next_boss'].long().flatten()),
                             self.context_character(obs['character'].long().flatten())), -1)
        return context.to(dtype)

    def _value(self, state, obs):
        if self.objective == COMBAT_OBJECTIVE:
            return super()._value(state, obs)
        if self.objective == MEAN_CONTRACT:
            from env.core.boss_value import win_hp_utility
            _, outputs = self.means(state)
            weight = obs['combat_is_boss'].float() * self.config.boss_win_weight
            outputs['boss_win_weight'] = weight
            return win_hp_utility(outputs['win_logits'].sigmoid(), outputs['hp_mean'], weight), outputs
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.scalar(state.float()), {}


class OuterReturnCore(EntityCombatCore):
    training_objective = OBJECTIVE
    context_schema = CONTEXT_SCHEMA
    combat_observation_keys = COMBAT_OBSERVATION_KEYS | CONTEXT_KEYS
    combat_only = False
    joint_upper = False
    frozen_upper = True

    def __init__(self, source, source_space, space, *, source_sha256, lower_initialization='random',
                 objective=OBJECTIVE, finite_curve_bank=None):
        nn.Module.__init__(self)
        if not CONTEXT_KEYS <= space.spaces.keys():
            raise ValueError('outer-return policy requires public run context')
        self.lower_initialization = lower_initialization
        self.training_objective = objective
        self.combat = OuterCombatNetwork(source.combat, source_space, space,
            initialization=lower_initialization, objective=objective)
        self.config = self.combat.config
        self.autoregressive_actions = True
        self.teacher = FrozenUpperTeacher(source, source_space, source_sha256=source_sha256)
        self.finite_curve_bank = FiniteCurveBank(finite_curve_bank) if finite_curve_bank is not None else None
        if (objective == FINITE_OBJECTIVE) != (self.finite_curve_bank is not None):
            raise ValueError('finite curve objective requires exactly one embedded bank')
        if self.finite_curve_bank is not None and self.finite_curve_bank.identity['teacher_sha256'] != self.teacher.assert_integrity():
            raise ValueError('finite curve bank teacher mismatch')
        self.map_dag_enabled = source.map_dag_enabled

    def disable_transition_auxiliary(self):
        if self.training_objective == COMBAT_OBJECTIVE:
            super().disable_transition_auxiliary()

    def actor_critic(self, obs, **kwargs):
        phase = obs['phase'].long().flatten()
        combat = phase == V.PHASE_VOCAB_INDEX['combat']
        logits = obs['player_scalars'].new_zeros((len(phase), 198), dtype=torch.float32)
        values = logits.new_zeros((len(phase), 1))
        for mask, upper in ((combat, False), (~combat, True)):
            indices = mask.nonzero().flatten()
            if indices.numel():
                batch = {key: value[indices] for key, value in obs.items()}
                out, value = (self.teacher(batch) if upper else self.combat_actor_critic(batch,
                    active_relic_width=kwargs.get('combat_relic_width')))
                logits[indices], values[indices] = out.float(), value.float()
        return logits, values

    def value(self, obs):
        return self.actor_critic(obs)[1]


class OuterReturnPolicy(EntityCombatPolicy):
    def obs_to_tensor(self, observation):
        if not isinstance(observation, dict) or not CONTEXT_KEYS <= observation.keys():
            raise ValueError('outer-return checkpoint requires the complete public run context; use its train/eval environment')
        return super().obs_to_tensor(observation)

    def __init__(self, *args, material_directory, material_sha256, source_sha256,
                 lower_initialization='random', objective=OBJECTIVE, finite_curve_bank=None, **kwargs):
        self.objective = objective
        self.finite_curve_bank_data = deepcopy(finite_curve_bank)
        self.material_directory = str(material_directory)
        self.material_sha256 = material_sha256
        self.source_sha256 = source_sha256
        self.lower_initialization = lower_initialization
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        source, source_space = load_initialization_material(self.material_directory,
            manifest_sha256=self.material_sha256, source_sha256=self.source_sha256)
        self.core = OuterReturnCore(source, source_space, self.observation_space,
            source_sha256=self.source_sha256, lower_initialization=self.lower_initialization,
            objective=self.objective, finite_curve_bank=self.finite_curve_bank_data)
        self.entity_config = asdict(self.core.config)
        self.optimizer = self.optimizer_class(self.core.combat.parameters(),
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), material_directory=self.material_directory,
                    material_sha256=self.material_sha256, source_sha256=self.source_sha256,
                    lower_initialization=self.lower_initialization, objective=self.objective,
                    finite_curve_bank=self.finite_curve_bank_data)
