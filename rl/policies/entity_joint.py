"""Independent strategic PPO network beside the unchanged entity combat core."""
from dataclasses import asdict, dataclass
import math

import torch

from env.core import native_actions as C, vocab as V, encode
from rl.policies.entity_combat import EntityCombatCore, EntityCombatConfig, EntityCombatPolicy
from rl.policies.split_policy import (
    SharedSemanticTokenizer, OverworldEncoder, _SpecialistActorCritic,
    OVERWORLD_OBSERVATION_KEYS, OVERWORLD_PHASE_OBSERVATION_KEYS,
)


@dataclass(frozen=True)
class UpperConfig:
    width: int = 256
    layers: int = 4
    heads: int = 8
    learning_rate: float = 1e-4
    epochs: int = 4
    micro_batch_size: int = 256
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02  # None records KL without early stopping.
    minibatch_size: int = 0
    optimizer: str = 'inherit'
    value_learning_rate: float = 0.
    weight_decay: float = 0.
    adam_eps: float = 1e-8
    architecture: str = 'legacy'
    latents: int = 16
    critic_epochs: int = 4
    effective_batch_size: int = 6144
    critic_effective_batch_size: int = 0  # 0 inherits effective_batch_size for old checkpoints.
    critic_target: str = 'mc'
    measure_value_loss: bool = False

    def __post_init__(self):
        if (not all(math.isfinite(value) for key, value in asdict(self).items()
                    if key not in ('optimizer', 'target_kl', 'architecture', 'critic_target', 'measure_value_loss'))
                or min(self.width, self.layers, self.heads, self.epochs, self.micro_batch_size) < 1
                or self.width % self.heads or self.learning_rate <= 0
                or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or not 0 < self.clip < 1
                or (self.target_kl is not None and
                    (not math.isfinite(self.target_kl) or self.target_kl <= 0))
                or self.max_grad_norm <= 0 or min(self.entropy_coef, self.value_coef) < 0
                or type(self.minibatch_size) is not int or self.minibatch_size < 0
                or self.optimizer not in ('inherit', 'adamw') or self.adam_eps <= 0
                or min(self.weight_decay, self.value_learning_rate) < 0
                or (self.optimizer == 'inherit' and (self.weight_decay or self.value_learning_rate))):
            raise ValueError('invalid strategic PPO configuration')
        if self.architecture not in ('legacy', 'upper-latent-actions-v1'):
            raise ValueError('unknown upper architecture contract')
        if self.critic_target not in ('mc', 'frozen-lambda') or (
                self.architecture == 'legacy' and self.critic_target != 'mc'):
            raise ValueError('critic target override requires unified upper')
        if type(self.measure_value_loss) is not bool or (
                self.measure_value_loss and self.architecture == 'legacy'):
            raise ValueError('value loss measurement requires unified upper and a boolean option')
        if (type(self.critic_effective_batch_size) is not int or self.critic_effective_batch_size < 0
                or (self.architecture == 'legacy' and self.critic_effective_batch_size)):
            raise ValueError('critic effective batch requires unified upper and a nonnegative integer')
        if self.architecture != 'legacy' and (
                self.gamma != 1 or self.target_kl is not None or self.optimizer != 'adamw'
                or min(self.latents, self.critic_epochs, self.effective_batch_size) < 1):
            raise ValueError('unified upper requires complete-game gamma1, AdamW and monitor-only KL')


class EntityJointCore(EntityCombatCore):
    combat_only = False
    joint_upper = True

    def __init__(self, space, config, upper_config):
        # Build combat first: its architecture, initialization and leaf path are identical.
        super().__init__(space, config)
        self.upper_config = upper_config
        if upper_config.architecture != 'legacy':
            from rl.policies.unified_upper import UnifiedUpperNetwork
            self.upper_actor = UnifiedUpperNetwork(space, upper_config, actor=True)
            self.upper_critic = UnifiedUpperNetwork(space, upper_config, actor=False)
            self.map_dag_enabled = True
            return
        if config.card_vector_checkpoint is None:
            # Preserve the established entity-joint initialization exactly
            # when no frozen dependency is requested.
            self.upper_tokenizer = SharedSemanticTokenizer(
                space, upper_config.width, card_language=True)
        else:
            card_dim = int(self.combat.card.embedding_dim)
            self.upper_tokenizer = SharedSemanticTokenizer(
                space, upper_config.width, card_language=False,
                card_dim=card_dim, card_embedding=self.combat.card)
        encoder = OverworldEncoder(space, d_model=upper_config.width,
            n_layers=upper_config.layers, n_heads=upper_config.heads,
            features_dim=upper_config.width, map_dag=True)
        self.overworld = _SpecialistActorCritic(encoder, d_model=upper_config.width,
            features_dim=upper_config.width, vf_arch=(upper_config.width,), kind='overworld')
        self.map_dag_enabled = encoder.map_dag_enabled

    def upper_parameters(self):
        if self.upper_config.architecture != 'legacy':
            yield from self.upper_actor.parameters()
            yield from self.upper_critic.parameters()
            return
        yield from self.upper_tokenizer.parameters()
        yield from self.overworld.parameters()

    def actor_critic(self, obs, **kwargs):
        phase = obs['phase'].long().flatten()
        if self.upper_config.architecture != 'legacy':
            upper = phase != V.PHASE_VOCAB_INDEX['combat']
            logits = torch.zeros(len(phase), C.AUTOREGRESSIVE_SIZE, device=phase.device)
            values = torch.zeros(len(phase), 1, device=phase.device)
            for selected, is_upper in ((upper.nonzero().flatten(), True), ((~upper).nonzero().flatten(), False)):
                if selected.numel():
                    batch = {k: v[selected] for k, v in obs.items()}
                    out, value = ((self.upper_actor(batch), self.upper_critic(batch)) if is_upper
                                  else self.combat_actor_critic(batch))
                    logits[selected], values[selected] = out.float(), value.float()
            return logits, values
        indices, logits, values = [], [], []
        for phase_id in torch.unique(phase).tolist():
            selected = (phase == phase_id).nonzero().flatten()
            if phase_id == V.PHASE_VOCAB_INDEX['combat']:
                batch = {k: v[selected] for k, v in obs.items()}
                out, value = self.combat_actor_critic(batch,
                    active_relic_width=kwargs.get('combat_relic_width'))
            else:
                keys = encode.upper_observation_keys(phase_id, obs)
                batch = {k: v[selected] for k, v in obs.items() if k in keys}
                out, value = self.overworld(batch, self.upper_tokenizer, phase_id)
            indices.append(selected)
            logits.append(out.float())
            values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    def value(self, obs):
        return self.actor_critic(obs)[1]


class EntityJointPolicy(EntityCombatPolicy):
    def __init__(self, *args, upper_config=None, small_playout_config=None,
                 upper_auxiliary_config=None, **kwargs):
        self.upper_config = asdict(UpperConfig(**(upper_config or {})))
        from rl.algorithms.upper_auxiliary import UpperAuxiliaryConfig
        self.upper_auxiliary_config = (asdict(UpperAuxiliaryConfig(**upper_auxiliary_config))
                                       if upper_auxiliary_config is not None else None)
        from rl.policies.small_playout import OnlinePlayoutConfig
        self.small_playout_config = (asdict(OnlinePlayoutConfig(**small_playout_config))
                                    if small_playout_config is not None else None)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if (C.policy_contract(self.observation_space) != C.AUTOREGRESSIVE_CONTRACT
                or self.action_space.n != C.policy_size(C.AUTOREGRESSIVE_CONTRACT)):
            raise ValueError('entity-joint requires fresh autoregressive-v2 whole-run training')
        cfg = UpperConfig(**self.upper_config)
        self.core = EntityJointCore(self.observation_space,
                                    EntityCombatConfig(**self.entity_config), cfg)
        self.optimizer = self.optimizer_class(self.core.combat.parameters(),
            lr=lr_schedule(1), **self.optimizer_kwargs)
        if cfg.architecture != 'legacy':
            self.upper_optimizer = torch.optim.AdamW([
                dict(params=self.core.upper_actor.parameters(), lr=cfg.learning_rate),
                dict(params=self.core.upper_critic.parameters(), lr=cfg.value_learning_rate or cfg.learning_rate),
            ], weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
        elif cfg.optimizer == 'inherit':
            self.upper_optimizer = self.optimizer_class(self.core.upper_parameters(),
                lr=cfg.learning_rate, **self.optimizer_kwargs)
        else:
            value_parameters = list(self.core.overworld.value_mlp.parameters()) + list(self.core.overworld.value_head.parameters())
            value_ids = {id(p) for p in value_parameters}
            self.upper_optimizer = torch.optim.AdamW([
                dict(params=[p for p in self.core.upper_parameters() if id(p) not in value_ids],
                     lr=cfg.learning_rate),
                dict(params=value_parameters, lr=cfg.value_learning_rate or cfg.learning_rate),
            ], weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
        if self.small_playout_config is not None:
            from rl.policies.small_playout import SmallPlayoutPolicy
            context = self.small_playout_config['context_mode']
            if context == 'none':
                self.small_playout = SmallPlayoutPolicy(self.small_playout_config['key_count'])
            elif context == 'token-root-64x2':
                from rl.policies.token_regret import OnlineTokenRegretPolicy
                self.small_playout = OnlineTokenRegretPolicy(self.small_playout_config['key_count'])
            else:
                from rl.policies.public_playout import PublicPlayoutPolicy
                self.small_playout = PublicPlayoutPolicy(self.small_playout_config['key_count'],context)
            optimizer = (torch.optim.AdamW if self.small_playout_config['optimizer'] == 'adamw'
                         else torch.optim.Adam)
            self.small_playout_optimizer = optimizer(self.small_playout.parameters(),
                lr=self.small_playout_config['learning_rate'],
                weight_decay=self.small_playout_config['weight_decay'])

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), upper_config=self.upper_config,
                    small_playout_config=self.small_playout_config,
                    upper_auxiliary_config=self.upper_auxiliary_config)
