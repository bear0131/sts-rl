"""One shared Transformer trained by whole-run PPO, without search teachers."""
import math

import torch
from torch import nn

from env.core import native_actions as C
from rl.policies.entity_combat import EntityCombatConfig, EntityCombatNetwork
from rl.policies.entity_joint import EntityJointCore, UpperConfig
from rl.policies.split_policy import (
    SharedSemanticTokenizer, OverworldEncoder, _SpecialistActorCritic,
    SplitStSPointerPolicy,
)


CONTRACT = 'shared-wholerun-ppo-room-v1'


class LearnedCardTable(nn.Module):
    """Random, trainable base/upgraded IDs; no pretrained rule encoder."""
    def __init__(self, count, width=64):
        super().__init__()
        self.num_embeddings, self.embedding_dim = count, width
        self.weight = nn.Parameter(torch.randn(count, 2, width) * .02)

    def versioned_forward(self, ids, upgrades):
        result = self.weight[ids.long(), upgrades.gt(0).long()]
        return result * ids.ne(0)[..., None]


class ScalarCombatNetwork(EntityCombatNetwork):
    def __init__(self, space, config, card_embedding):
        super().__init__(space, config, card_embedding=card_embedding)
        # The old distribution predicts fight-local HP. Whole-run PPO requires
        # an unrestricted scalar in the unchanged strategic reward units.
        del self.distribution, self.transition
        self.value_head = nn.Sequential(nn.Linear(config.width, config.width),
                                       nn.GELU(), nn.Linear(config.width, 1))

    def _value(self, state, obs):
        return self.value_head(state), {}


class SharedWholeRunCore(EntityJointCore):
    joint_upper = False
    combat_only = False
    training_objective = CONTRACT
    environment_value_objective = None

    def actor_critic(self, observations, **kwargs):
        runtime = getattr(self, '_collection_graphs', None)
        if runtime is not None:
            return runtime(observations)
        return super().actor_critic(observations, **kwargs)

    def __init__(self, space, *, width, layers, heads):
        nn.Module.__init__(self)
        if C.policy_contract(space) != C.AUTOREGRESSIVE_CONTRACT:
            raise ValueError('shared whole-run PPO requires autoregressive-v2')
        self.autoregressive_actions = True
        self.config = EntityCombatConfig(width=width, layers=layers, heads=heads)
        self.upper_config = UpperConfig(width=width, layers=layers, heads=heads)
        cards = LearnedCardTable(int(space['hand_ids'].high.max()) + 1)
        self.combat = ScalarCombatNetwork(space, self.config, cards)
        self.upper_tokenizer = SharedSemanticTokenizer(
            space, width, card_language=False, card_embedding=cards)
        encoder = OverworldEncoder(space, d_model=width, n_layers=layers,
                                  n_heads=heads, features_dim=width, map_dag=True)
        # Both phases call the very same four layers, and one optimizer owns
        # each parameter once. Observation adapters retain their slot semantics.
        encoder.transformer = self.combat.transformer
        self.overworld = _SpecialistActorCritic(
            encoder, d_model=width, features_dim=width,
            vf_arch=(width,), kind='overworld')
        self.map_dag_enabled = encoder.map_dag_enabled


class SharedWholeRunPolicy(SplitStSPointerPolicy):
    # The original learner normalized each 256-row microbatch. Fusing their
    # model computation must preserve those statistical groups and sample order.
    ppo_advantage_group_size = 256

    def __init__(self, *args, room_lambda=.9, **kwargs):
        if not math.isfinite(room_lambda) or not 0 <= room_lambda <= 1:
            raise ValueError('room lambda must be in [0, 1]')
        self.room_lambda = float(room_lambda)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        cfg = self._core_cfg
        if (cfg['card_vector_checkpoint'] is not None or cfg['combat_arch'] != 'transformer'
                or self.action_space.n != C.AUTOREGRESSIVE_SIZE):
            raise ValueError('pure PPO rejects pretrained cards or incompatible actions')
        self.core = SharedWholeRunCore(self.observation_space, width=cfg['d_model'],
                                      layers=cfg['n_layers'], heads=cfg['n_heads'])
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), **self._core_cfg,
                    room_lambda=self.room_lambda)
