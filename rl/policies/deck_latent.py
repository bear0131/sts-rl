"""Large cached run context with small combat and pre-reward anchor networks.

Fresh checkpoint geometry, autoregressive-v2 actions. Search may call
``combat_with_context`` / ``anchor_with_context`` with shared root context;
neither interface invokes the large encoder. PPO recomputes context with
gradients, deduplicating equal contexts only within the current physical batch.
"""
from math import prod
from dataclasses import dataclass

import torch
from torch import nn

from env.core import encode, native_actions as C, vocab
from rl.policies.combat_summary import AdditiveCombatInputs
from rl.policies.entity_combat import EntityCombatConfig, EntityEncoderLayer
from rl.policies.entity_joint import UpperConfig
from rl.policies.shared_wholerun import (
    LearnedCardTable, ScalarCombatNetwork, SharedOverworldEncoder,
    SharedWholeRunCore, SharedWholeRunPolicy, _SharedCategoricalDistribution,
)
from rl.policies.split_policy import SharedSemanticTokenizer, _SpecialistActorCritic, _active_prefix_width


MODEL_CONTRACT = 'deck-latent-combat-v1'


@dataclass(frozen=True)
class SearchContext:
    """Root-owned, read-only projections; many branch lanes may index a row."""
    combat: torch.Tensor
    anchor: torch.Tensor

    def index(self, rows):
        return SearchContext(self.combat[rows], self.anchor[rows])


class RunContextEncoder(nn.Module):
    """No HP, potions, gold or relic counters in the expensive cache key."""
    keys = ('deck_ids', 'deck_upgrades', 'outer_deck_misc', 'relics',
            'map_full_rooms', 'map_full_features', 'act', 'floor', 'next_boss',
            'character', 'ascension')

    def __init__(self, space, cards, width, layers, heads):
        super().__init__()
        self.card = cards
        self.card_projection = nn.Linear(cards.embedding_dim + 2, width)
        self.relic = nn.Embedding(int(space['relics'].high.max()) + 1, width, padding_idx=0)
        self.room = nn.Embedding(int(space['map_full_rooms'].high.max()) + 1, width, padding_idx=0)
        self.map_projection = nn.Linear(space['map_full_features'].shape[-1], width)
        self.global_keys = ('act', 'floor', 'next_boss', 'character', 'ascension')
        self.global_projection = nn.Linear(sum(prod(space[k].shape) for k in self.global_keys), width)
        self.roles = nn.Parameter(torch.randn(4, width) * .02)
        layer = EntityEncoderLayer(width, heads, width * 4, dropout=0., activation='gelu',
                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def key_tensor(self, obs):
        return torch.cat([obs[k].reshape(len(obs['phase']), -1).float() for k in self.keys], 1)

    @staticmethod
    def map_inputs(rooms, features):
        # Coordinates/edges are explicit features; there are no sequence
        # position embeddings. Remove only masked grid holes, preserving every
        # visible node (including room-PAD nodes with public feature values).
        keep = rooms.ne(0) | features.bool().any(-1)
        width = max(1, int(keep.sum(1).max().item()))
        indices = torch.argsort(~keep, dim=1, stable=True)[:, :width]
        return rooms.gather(1, indices), features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1]))

    def forward(self, obs):
        width = _active_prefix_width(obs['deck_ids'])
        ids, upgrades = obs['deck_ids'][:, :width], obs['deck_upgrades'][:, :width]
        deck = self.card_projection(torch.cat((self.card.versioned_forward(ids, upgrades),
            upgrades[..., None], obs['outer_deck_misc'][:, :width, None]), -1))
        relic_ids = obs['relics'][:, :_active_prefix_width(obs['relics'])]
        rooms, features = self.map_inputs(obs['map_full_rooms'], obs['map_full_features'].float())
        maps = self.room(rooms.long()) + self.map_projection(features)
        global_token = self.global_projection(torch.cat([obs[k].float() for k in self.global_keys], -1))
        blocks = (global_token[:, None], deck, self.relic(relic_ids.long()), maps)
        mask = torch.cat((torch.zeros(len(ids), 1, dtype=torch.bool, device=ids.device),
                          ids == 0, relic_ids == 0,
                          (rooms == 0) & ~features.any(-1)), 1)
        tokens = torch.cat([block + self.roles[i] for i, block in enumerate(blocks)], 1)
        return self.norm(self.transformer(tokens, src_key_padding_mask=mask))[:, 0]


class ContextOverworldEncoder(SharedOverworldEncoder):
    def _context_blocks(self, obs, shared, active_widths):
        blocks = super()._context_blocks(obs, shared, active_widths)
        name, value, mask = blocks[0]
        blocks[0] = (name, value + obs['_run_context'][:, None], mask)
        return blocks


def _mean(value, keep):
    return (value * keep[..., None]).sum(1) / keep.sum(1).clamp_min(1)[:, None]


class FastCombatNetwork(ScalarCombatNetwork):
    """64-wide entities, 128-wide shared state, no all-entity attention."""
    def __init__(self, space, cards, context_width):
        super().__init__(space, EntityCombatConfig(width=64, layers=1, heads=4, pile_queries=1), cards)
        del self.transformer, self.pool_queries, self.pool_count, self.pile_role, self.role
        del self.relic, self.relic_projection
        self.summaries = AdditiveCombatInputs(cards, int(space['relics'].high.max()) + 1, 64)
        self.context_projection = nn.Linear(context_width, 128)
        self.state_mlp = nn.Sequential(nn.Linear(64 * 6 + 3 * 65, 128), nn.GELU(), nn.Linear(128, 128))
        self.state_output = nn.Sequential(nn.GELU(), nn.Linear(128, 64), nn.LayerNorm(64))
        self.entity_film = nn.Linear(64, 128)

    def training_token_counts(self, obs):
        # Conservative physical memory proxy, not an attention FLOP claim.
        return (obs['deck_ids'] != 0).sum(1) + (obs['relics'] != 0).sum(1) + 128

    def encode(self, obs, active_relic_width=None, *, card_table=None):
        table = (self._encode_card_table() if self.training or torch.is_grad_enabled()
                 else self.inference_card_table()) if card_table is None else card_table
        hp = obs['player_scalars'].float(); batch = len(hp)
        turn = hp[:, 4:5] * encode.TURN_SCALE
        extra = torch.cat((hp[:, 1:2] - hp[:, :1], turn.clamp_min(0).log1p()/4,
                           torch.sigmoid((hp[:, 5:6] - self.hp_thresholds)*12)), -1)
        player = self.player_projection(torch.cat(
            [obs[k].float().reshape(batch, -1) for k in self.scalar_keys] + [extra], -1))
        player = player + self.turn_bucket(turn.long().flatten().clamp(0, 31))
        player = player + self.choice_reason(obs['select_context'].long().flatten())
        hand = self._card_tokens(table, obs['hand_ids'], obs['hand_upgrades'])
        hand = hand + self.hand_context(torch.stack([obs[k] for k in
            ('hand_costs', 'hand_damage', 'hand_block', 'select_hand_mask')], -1).float())
        monsters = self.monster_projection(torch.cat((self.monster_id(obs['monster_id'].long()),
            self.monster_move(obs['monster_move'].long()), self.monster_intent(obs['monster_intent_cat'].long()),
            obs['monster_scalars'].float(), obs['monster_buffs'].float()), -1))
        potions = self.potion(obs['potions'].long())
        choices = self._card_tokens(table, obs['select_card_ids'], obs['select_card_upgrades'])
        valid = obs['select_native_indices'] > 0
        choices = choices + self.choice_context(torch.stack((obs['select_card_costs'].float(),
            (valid & (obs['select_card_ids'] == 0)).float()), -1))
        choices = choices + self.choice_zone(obs['select_zone'].long().flatten())[:, None]
        summary = obs.get('_combat_summary')
        if summary is None:
            summary = self.summaries.rebuild(obs)
        parts = (player, _mean(hand, obs['hand_ids'] != 0), _mean(monsters, obs['monster_id'] != 0),
                 _mean(potions, obs['potions'] != 0), _mean(choices, valid), summary.relics,
                 self.summaries.features(summary))
        context = obs.get('_combat_context')
        if context is None:
            context = self.context_projection(obs['_run_context'])
        state = self.state_output(self.state_mlp(torch.cat(parts, -1)) + context)
        scale, bias = self.entity_film(state).chunk(2, -1)
        def condition(entity):
            return self.norm(entity * (1 + scale[:, None].tanh()) + bias[:, None])
        return state, dict(hand=condition(hand), monster=condition(monsters),
                           potion=condition(potions), select=condition(choices))


class FastAnchorValue(nn.Module):
    """Only continuation resources and cached context; no reward-page fields."""
    def __init__(self, space, context_width):
        super().__init__()
        self.context_projection = nn.Linear(context_width, 256, bias=False)
        self.potion = nn.Embedding(int(space['potions'].high.max()) + 1, 16, padding_idx=0)
        self.dynamic_keys = ('gold', 'run_keys', 'combat_exit_anchor')
        features = 2 + 64 + space['potions'].shape[0] * 16 + sum(prod(space[k].shape) for k in self.dynamic_keys)
        self.dynamic_projection = nn.Linear(features, 256)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, obs, relic_summary, projected_context):
        # The pre-reward observation is still combat: run_hp may be the stale
        # GameContext value; player_scalars holds the actual branch HP/maxHP.
        parts = [obs['player_scalars'][:, :2].float(), relic_summary,
                 self.potion(obs['potions'].long()).flatten(1)]
        parts.extend(obs[k].float() for k in self.dynamic_keys)
        return self.head(self.dynamic_projection(torch.cat(parts, -1)) + projected_context)


class DeckLatentCore(SharedWholeRunCore):
    model_contract = MODEL_CONTRACT
    collection_graphs_supported = False

    def __init__(self, space, *, width=512, layers=8, heads=8):
        nn.Module.__init__(self)
        self.autoregressive_actions = True
        self.config = EntityCombatConfig(width=64, layers=1, heads=4, pile_queries=1)
        self.upper_config = UpperConfig(width=width, layers=2, heads=heads)
        cards = LearnedCardTable(int(space['hand_ids'].high.max()) + 1)
        self.context = RunContextEncoder(space, cards, width, layers, heads)
        self.combat = FastCombatNetwork(space, cards, width)
        self.anchor = FastAnchorValue(space, width)
        self.upper_tokenizer = SharedSemanticTokenizer(space, width, card_language=False, card_embedding=cards)
        encoder = ContextOverworldEncoder(space, d_model=width, n_layers=2, n_heads=heads,
                                         features_dim=width, map_dag=True)
        self.overworld = _SpecialistActorCritic(encoder, d_model=width, features_dim=width,
                                              vf_arch=(width,), kind='overworld')
        self.map_dag_enabled = encoder.map_dag_enabled
        self._context_cache = None

    def _apply(self, fn, recurse=True):
        self._context_cache = None
        return super()._apply(fn, recurse=recurse)

    def _encode_unique(self, obs, keys):
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        first = torch.full((len(unique),), len(keys), dtype=torch.long, device=keys.device)
        first.scatter_reduce_(0, inverse, torch.arange(len(keys), device=keys.device), reduce='amin')
        result = self.context({k: v[first] for k, v in obs.items()})
        return result[inverse]

    def run_context(self, obs):
        replay = getattr(self, '_context_gradient_batch', None)
        if replay is not None:
            if '_deck_context_index' not in obs:
                raise RuntimeError('context gradient replay lost the original row mapping')
            return replay.latents[obs['_deck_context_index'].long().flatten()]
        # The learner has already compared complete static keys once for this
        # Adam batch. Reuse its row identities instead of sorting wide keys in
        # every physical chunk; never use these local IDs for inference caches.
        keys = (obs['_deck_context_index'][:, None] if self.training and '_deck_context_index' in obs
                else self.context.key_tensor(obs))
        if self.training or torch.is_grad_enabled():
            self._context_cache = None
            return self._encode_unique(obs, keys)
        versions = tuple((id(p), p._version) for p in self.context.parameters())
        entry = self._context_cache
        if entry is None or entry[0] != versions or entry[1].shape != keys.shape:
            result = self._encode_unique(obs, keys)
        else:
            changed = (keys != entry[1]).any(1).nonzero().flatten()
            result = entry[2].clone()
            if len(changed):
                result[changed] = self._encode_unique({k: v[changed] for k, v in obs.items()}, keys[changed])
        self._context_cache = (versions, keys.detach().clone(), result.detach())
        return result

    def combat_with_context(self, obs, context, summary=None):
        batch = (dict(obs, _combat_context=context.combat) if isinstance(context, SearchContext)
                 else dict(obs, _run_context=context))
        if summary is not None:
            batch['_combat_summary'] = summary
        return self.combat(batch)[:2]

    def anchor_with_context(self, obs, projected_context, relic_summary=None):
        if isinstance(projected_context, SearchContext):
            projected_context = projected_context.anchor
        if relic_summary is None:
            relic_summary = self.combat.summaries.relic_terms(obs['relics'], obs['relic_counters']).sum(1)
        return self.anchor(obs, relic_summary, projected_context)

    def prepare_search_context(self, obs):
        z = self.run_context(obs)
        return SearchContext(self.combat.context_projection(z), self.anchor.context_projection(z))

    def actor_critic(self, obs, **kwargs):
        context = self.run_context(obs)
        phase = obs['phase'].long().flatten()
        anchored = obs['combat_exit_anchor'].bool().any(1)
        indices, logits, values = [], [], []
        for screen in torch.unique(phase).tolist():
            for anchor in ((False, True) if screen == vocab.PHASE_VOCAB_INDEX['combat'] else (False,)):
                rows = ((phase == screen) & (anchored == anchor)).nonzero().flatten()
                if not len(rows):
                    continue
                batch = {key: value[rows] for key, value in obs.items()}
                z = context[rows]
                if anchor:
                    out = z.new_zeros(len(rows), C.AUTOREGRESSIVE_SIZE)
                    value = self.anchor_with_context(batch, self.anchor.context_projection(z))
                elif screen == vocab.PHASE_VOCAB_INDEX['combat']:
                    out, value = self.combat_with_context(batch, z)
                else:
                    out, value = self.overworld(dict(batch, _run_context=z), self.upper_tokenizer, screen)
                indices.append(rows); logits.append(out.float()); values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]


class DeckLatentPolicy(SharedWholeRunPolicy):
    def _build(self, lr_schedule):
        required = {'combat_exit_anchor', 'reward_stage', 'outer_deck_misc', 'map_full_features',
                    'pile_upgraded_counts', 'pile_upgrade_sums', 'deck_latent_context'}
        if not required <= self.observation_space.spaces.keys():
            raise ValueError('deck latent policy requires its complete public context observation contract')
        if self.room_lambda != 1 or self.reward_flow != 'sequential-v1':
            raise ValueError('deck latent PPO requires lambda1 and sequential rewards')
        if C.policy_contract(self.observation_space) != C.AUTOREGRESSIVE_CONTRACT:
            raise ValueError('deck latent policy requires autoregressive-v2')
        cfg = self._core_cfg
        if cfg['card_vector_checkpoint'] is not None or cfg['combat_arch'] != 'transformer':
            raise ValueError('deck latent geometry requires fresh learned cards and its own combat network')
        self.action_dist = _SharedCategoricalDistribution(self.action_space.n)
        self.core = DeckLatentCore(self.observation_space, width=cfg['d_model'],
                                  layers=cfg['n_layers'], heads=cfg['n_heads'])
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
