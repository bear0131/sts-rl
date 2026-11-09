"""One shared Transformer trained by whole-run PPO, without search teachers."""
import math

import torch
from torch import nn
from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableCategoricalDistribution

from env.core import native_actions as C, vocab
from rl.policies.entity_combat import EntityCombatConfig, EntityCombatNetwork
from rl.policies.entity_joint import EntityJointCore, UpperConfig
from rl.policies.split_policy import (
    SharedSemanticTokenizer, OverworldEncoder, _SpecialistActorCritic,
    SplitStSPointerPolicy,
    _active_prefix_width,
)


CONTRACT = 'shared-wholerun-ppo-room-v1'
# Internal dispatch only; the observation keeps its real screen identity.
MERGED_UPPER_PHASE = -1


class LearnedCardTable(nn.Module):
    """Random, trainable base/upgraded IDs; no pretrained rule encoder."""
    def __init__(self, count, width=64):
        super().__init__()
        self.num_embeddings, self.embedding_dim = count, width
        self.weight = nn.Parameter(torch.randn(count, 2, width) * .02)

    def versioned_forward(self, ids, upgrades):
        result = self.weight[ids.long(), upgrades.gt(0).long()]
        return result * ids.ne(0)[..., None]


class SharedOverworldEncoder(OverworldEncoder):
    """Remove only masked queries that no existing action/value head reads."""

    @staticmethod
    def merged_token_counts(obs, rows=None):
        """Retained key/query counts for common upper screens, before padding."""
        def take(key):
            return obs[key] if rows is None else obs[key][rows]
        phase = take('phase').reshape(-1)
        kept = (2 + obs['potions'].shape[1] + (take('deck_ids') != 0).sum(axis=1)
                + (take('relics') != 0).sum(axis=1))
        kept += ((take('map_full_rooms') != 0)
                 | (take('map_full_features')[..., 3] > .5)).sum(axis=1)
        counts = {
            'combat_reward': sum(obs[k].shape[1] for k in
                ('offer_card_ids', 'reward_gold', 'reward_relic_ids', 'reward_potion_ids'))
                + (1 if 'reward_keys' in obs else 0),
            'event': sum(obs[k].shape[1] for k in ('neow_bonus_ids', 'event_option_ids')
                         if k in obs),
            'shop': sum(obs[k].shape[1] for k in
                        ('shop_card_ids', 'shop_relic_ids', 'shop_potion_ids')),
            'boss_relic': obs['boss_relic_ids'].shape[1],
            'map': obs['map_node_rooms'].shape[1],
        }
        for name, extra in counts.items():
            kept += (phase == vocab.PHASE_VOCAB_INDEX[name]) * extra
        return kept

    @classmethod
    def merged_width(cls, obs, rows=None, *, multiple=32):
        """Physical width for common upper screens; accepts host arrays or tensors."""
        kept = cls.merged_token_counts(obs, rows)
        return (int(kept.max().item()) + multiple - 1) // multiple * multiple

    def forward(self, obs, shared, phase_id, active_widths=None, map_encoding=None):
        if phase_id != MERGED_UPPER_PHASE:
            return super().forward(obs, shared, phase_id,
                                   active_widths=active_widths, map_encoding=map_encoding)
        # A shared Transformer has no cross-row attention. Pack each screen's
        # original role sequence, then run the independent rows together.
        phase = obs['phase'].long().flatten()
        blocks = self._context_blocks(obs, shared, active_widths)
        active_roles = {}
        # Absent training branches must remain absent from autograd: a zero
        # gradient would still advance their existing Adam momentum.
        present = set(torch.unique(phase).tolist()) if self.training else None
        for name in ('combat_reward', 'map', 'shop', 'boss_relic', 'event'):
            screen = vocab.PHASE_VOCAB_INDEX[name]
            if present is not None and screen not in present:
                continue
            added = self._screen_blocks(obs, shared, screen)
            blocks.extend(added)
            for role, _, _ in added:
                active_roles[role] = phase == screen
        blocks.extend(self._map_blocks(obs, MERGED_UPPER_PHASE, map_encoding))
        count, device = len(phase), phase.device
        tokens = [self._role(self.cls.expand(count, -1, -1), 'cls')]
        masks = [torch.zeros(count, 1, dtype=torch.bool, device=device)]
        discards = [torch.zeros_like(masks[0])]
        spans, offset = {}, 1
        for role, token, pad in blocks:
            spans[role] = offset, offset + token.shape[1]
            offset += token.shape[1]
            active = active_roles.get(role)
            inactive = (torch.zeros_like(pad) if active is None
                        else (~active[:, None]).expand_as(pad))
            discard = inactive.clone()
            if role in ('deck', 'relic', 'selected'):
                discard = discard | pad
            if role == 'map_full':
                discard = discard | (pad & ~(obs['map_full_features'][..., 3] > .5))
            tokens.append(self._role(token, role))
            masks.append(pad | inactive)
            discards.append(discard)
        source, pad_mask = torch.cat(tokens, 1), torch.cat(masks, 1)
        discard = torch.cat(discards, 1)
        width = min(source.shape[1], active_widths[2])
        positions = torch.argsort(discard.to(torch.int32), dim=1, stable=True)[:, :width]
        gather = positions[:, :, None].expand(-1, -1, source.shape[-1])
        encoded = self.transformer(source.gather(1, gather),
                                   src_key_padding_mask=pad_mask.gather(1, positions))
        output = torch.zeros_like(source).scatter(1, gather, encoded)
        contextual = {role: output[:, begin:end] for role, (begin, end) in spans.items()}
        contextual['__active_roles__'] = active_roles
        return output[:, 0], contextual

    def _transform_tokens(self, source, pad_mask, spans, *, map_features=None):
        if not self.training:
            return super()._transform_tokens(source, pad_mask, spans, map_features=map_features)
        # A key mask does NOT imply an unused query: a PAD card may still be
        # the explicit skip candidate. Preserve all pointer-head roles in full.
        discard = torch.zeros_like(pad_mask)
        for role in ('deck', 'relic', 'selected'):
            if role in spans:
                begin, end = spans[role]
                discard[:, begin:end] = pad_mask[:, begin:end]
        if 'map_full' in spans and map_features is not None:
            begin, end = spans['map_full']
            # Map scores consume choice flags, not room IDs. Preserve a choice
            # even when its ID is PAD; never infer action validity from identity.
            discard[:, begin:end] = pad_mask[:, begin:end] & ~(map_features[..., 3] > .5)
        width = int((~discard).sum(1).max().item())
        positions = torch.argsort(discard.to(torch.int32), dim=1, stable=True)[:, :width]
        gather = positions[:, :, None].expand(-1, -1, source.shape[-1])
        packed = source.gather(1, gather)
        packed_mask = pad_mask.gather(1, positions)
        encoded = self.transformer(packed, src_key_padding_mask=packed_mask)
        # The original role spans and native candidate slots remain unchanged.
        return torch.zeros_like(source).scatter(1, gather, encoded)


class SharedTrainingEncoder(nn.TransformerEncoder):
    """Update-only autocast; master parameters and collection remain FP32."""
    update_precision = 'fp32'

    def forward(self, src, *args, **kwargs):
        if self.training and self.update_precision == 'fp16':
            if not src.is_cuda or src.dtype != torch.float32:
                raise ValueError('shared FP16 updates require CUDA FP32 activations')
            with torch.autocast('cuda', dtype=torch.float16):
                return super().forward(src, *args, **kwargs).float()
        return super().forward(src, *args, **kwargs)


class ScalarCombatNetwork(EntityCombatNetwork):
    encoder_type = SharedTrainingEncoder
    compact_training_tokens = False

    def __init__(self, space, config, card_embedding):
        super().__init__(space, config, card_embedding=card_embedding)
        # The old distribution predicts fight-local HP. Whole-run PPO requires
        # an unrestricted scalar in the unchanged strategic reward units.
        del self.distribution, self.transition
        self.value_head = nn.Sequential(nn.Linear(config.width, config.width),
                                       nn.GELU(), nn.Linear(config.width, 1))

    def _value(self, state, obs):
        return self.value_head(state), {}

    @staticmethod
    def required_entity_queries(obs):
        # A masked key can still be a legal query or a selected prefix member.
        # Keep native slot identities; gather/scatter only inside attention.
        legal = obs['combat_legal'].bool()
        targeted = legal[:, 1:91].reshape(-1, 15, 6)
        sources = targeted.any(-1)
        multi = C.JOINT_MULTI - C.JOINT_END
        return dict(
            hand=(obs['hand_ids'] != 0) | sources[:, :10]
                 | obs['select_hand_mask'].bool() | legal[:, multi:multi + 10],
            monster=(obs['monster_id'] != 0) | targeted.any(1)[:, :5],
            potion=(obs['potions'] != 0) | sources[:, 10:],
            relic=obs['relics'] != 0)

    def training_token_counts(self, obs):
        return 1 + 4 * self.config.pile_queries + sum(
            mask.sum(1) for mask in self.required_entity_queries(obs).values())

    def _encode_tokens(self, source, pad_mask, spans, obs):
        if not self.training or not self.compact_training_tokens:
            return super()._encode_tokens(source, pad_mask, spans, obs)
        keep = ~pad_mask
        for name, mask in self.required_entity_queries(obs).items():
            begin, end = spans[name]
            keep[:, begin:end] |= mask[:, :end - begin]
        width = int(keep.sum(1).max().item())
        positions = torch.argsort((~keep).int(), dim=1, stable=True)[:, :width]
        gather = positions[:, :, None].expand(-1, -1, source.shape[-1])
        encoded = self.norm(self.transformer(source.gather(1, gather),
            src_key_padding_mask=pad_mask.gather(1, positions)))
        return torch.zeros_like(source).scatter(1, gather, encoded)


class SharedWholeRunCore(EntityJointCore):
    joint_upper = False
    combat_only = False
    training_objective = CONTRACT
    environment_value_objective = None

    def combat_actor_critic(self, observations, **kwargs):
        if self.training and kwargs.get('active_relic_width') is None:
            # Masked trailing queries never reach a head; keeping interior
            # positions preserves every actual relic and its counter.
            kwargs['active_relic_width'] = _active_prefix_width(observations['relics'])
        return super().combat_actor_critic(observations, **kwargs)

    def actor_critic(self, observations, **kwargs):
        runtime = getattr(self, '_collection_graphs', None)
        if runtime is not None:
            return runtime(observations)
        if (self.training and self.map_dag_enabled
                and 'upper_public_context' in observations):
            phase = observations['phase'].long().flatten()
            selected = (phase != vocab.PHASE_VOCAB_INDEX['combat']).nonzero().flatten()
            if selected.numel():
                encoder = self.overworld.encoder
                # Map encoding is independent of the active upper screen. Run
                # its recurrent passes once for the entire upper sub-batch;
                # each phase still receives its own rows, tokens and action head.
                encoded = encoder.map_dag(
                    encoder.room_embedding(observations['map_full_rooms'][selected].long()),
                    observations['map_full_features'][selected].float())
                positions = torch.zeros_like(phase)
                positions[selected] = torch.arange(len(selected), device=phase.device)
                kwargs['upper_map_encoding'] = (positions, encoded)
            return self._merged_upper_actor_critic(observations, kwargs.get('upper_map_encoding'))
        return super().actor_critic(observations, **kwargs)

    def _merged_upper_actor_critic(self, observations, map_encoding):
        phase = observations['phase'].long().flatten()
        combat, selection = (vocab.PHASE_VOCAB_INDEX[name] for name in ('combat', 'card_select'))
        indices, logits, values = [], [], []
        groups = ((phase == combat, combat), (phase == selection, selection),
                  ((phase != combat) & (phase != selection), MERGED_UPPER_PHASE))
        for mask, screen in groups:
            rows = mask.nonzero().flatten()
            if not len(rows):
                continue
            batch = {key: value[rows] for key, value in observations.items()}
            if screen == combat:
                out, value = self.combat_actor_critic(batch)
            else:
                widths = None if screen != MERGED_UPPER_PHASE else (
                    _active_prefix_width(batch['deck_ids']),
                    _active_prefix_width(batch['relics']),
                    self.overworld.encoder.merged_width(batch, multiple=1))
                out, value = self.overworld(batch, self.upper_tokenizer, screen,
                    upper_active_widths=widths,
                    upper_map_encoding=map_encoding[1][map_encoding[0][rows]])
            indices.append(rows)
            logits.append(out.float())
            values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

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
        encoder = SharedOverworldEncoder(space, d_model=width, n_layers=layers,
                                  n_heads=heads, features_dim=width, map_dag=True)
        # Both phases call the very same four layers, and one optimizer owns
        # each parameter once. Observation adapters retain their slot semantics.
        encoder.transformer = self.combat.transformer
        self.overworld = _SpecialistActorCritic(
            encoder, d_model=width, features_dim=width,
            vf_arch=(width,), kind='overworld')
        self.map_dag_enabled = encoder.map_dag_enabled


class _SharedCategoricalDistribution(MaskableCategoricalDistribution):
    def proba_distribution(self, action_logits):
        if action_logits.is_cuda and not torch.is_grad_enabled():
            logits = action_logits.view(-1, self.action_dim)
            # Keep a device-side finite check without synchronizing the host
            # solely to validate each already-produced distribution.
            torch._assert_async(torch.isfinite(logits).all(), 'nonfinite shared-policy logits')
            self.distribution = MaskableCategorical(logits=logits, validate_args=False)
            return self
        return super().proba_distribution(action_logits)


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
        self.action_dist = _SharedCategoricalDistribution(self.action_space.n)
        self.core = SharedWholeRunCore(self.observation_space, width=cfg['d_model'],
                                      layers=cfg['n_layers'], heads=cfg['n_heads'])
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), **self._core_cfg,
                    room_lambda=self.room_lambda)
