"""Combat-only entity policy; public state, no overworld network or memory."""
from dataclasses import asdict, dataclass, replace
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from env.core import action_space as A, encode, native_actions as C, vocab as V
from rl.policies.model import CardLanguageEmbedding
from rl.policies.complete_combat import CompleteCombatHead
from rl.policies.combat_distribution import CombatDistributionHead
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS, SplitStSPointerPolicy
from rl.rules.rule_vector_model import load_frozen_card_vector_table


@dataclass(frozen=True)
class EntityCombatConfig:
    width: int = 256
    layers: int = 4
    heads: int = 8
    pile_queries: int = 3
    quantiles: int = 32
    mean_loss_coef: float = .25
    auxiliary_loss_coef: float = 0.0
    boss_win_weight: float = 0.0
    card_vector_checkpoint: str | None = None

    def __post_init__(self):
        if not 0 <= self.boss_win_weight <= 1:
            raise ValueError('boss_win_weight must be finite and in [0, 1]')
        if self.card_vector_checkpoint is not None and not self.card_vector_checkpoint:
            raise ValueError('card_vector_checkpoint must be a non-empty path')
        if min(self.width, self.layers, self.heads, self.pile_queries) < 1 or self.width % self.heads:
            raise ValueError('invalid entity transformer geometry')
        if self.quantiles < 2 or min(self.mean_loss_coef, self.auxiliary_loss_coef) < 0:
            raise ValueError('invalid distribution objective configuration')


def _vocabulary(space, key):
    return int(space[key].high.max()) + 1


class EntityEncoderLayer(nn.TransformerEncoderLayer):
    """Keep training and FP16 inference projections in batch-first layout.

    Parameter names and the pre-norm/residual/MLP operations remain PyTorch's.
    Unbinding Q/K/V also gives backward one stacked projection gradient,
    instead of three full-size zero-filled select gradients and their sums.
    """

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Only the opt-in CUDA FP16 inference path uses the new layout. BF16
        # root/leaf and FP32 teacher inference keep their established rounding.
        # The FP16 path is validated with dynamic relic lengths and a complete
        # collection, including graph eviction and temporary memory release.
        fp16_inference = (x.is_cuda and not torch.is_grad_enabled()
                          and torch.is_autocast_enabled('cuda')
                          and torch.get_autocast_dtype('cuda') == torch.float16)
        if ((not self.training and not fp16_inference) or x.ndim != 3 or not self.self_attn.batch_first
                or attn_mask is not None or is_causal):
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)
        batch, length, width = x.shape
        attention = self.self_attn
        query, key, value = F.linear(x, attention.in_proj_weight,
            attention.in_proj_bias).reshape(batch, length, 3,
                attention.num_heads, width // attention.num_heads).unbind(2)
        # TransformerEncoderLayer canonicalizes padding to additive 0/-inf.
        mask = None if key_padding_mask is None else key_padding_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=mask, dropout_p=attention.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.dropout1(F.linear(attended, attention.out_proj.weight,
                                     attention.out_proj.bias))


class EntityCombatNetwork(nn.Module):
    encoder_type = nn.TransformerEncoder

    def __init__(self, space, config, *, card_embedding=None):
        super().__init__()
        self.config = config
        width = config.width
        self.n_cards = _vocabulary(space, 'hand_ids')
        self.card = (
            card_embedding
            if card_embedding is not None
            else CardLanguageEmbedding(self.n_cards, 64)
        )
        card_dim = int(getattr(self.card, 'embedding_dim', 64))
        if int(getattr(self.card, 'num_embeddings', self.n_cards)) != self.n_cards:
            raise ValueError('card embedding does not cover the environment vocabulary')
        self.card_projection = nn.Linear(card_dim + 1, width)
        self.register_buffer('card_ids', torch.arange(self.n_cards).repeat_interleave(2), persistent=False)
        self.register_buffer('card_versions', torch.arange(2).repeat(self.n_cards), persistent=False)
        self.hand_context = nn.Linear(4, width)
        self.choice_context = nn.Linear(2, width)
        self.choice_zone = nn.Embedding(_vocabulary(space, 'select_zone'), width)
        self.choice_reason = nn.Embedding(_vocabulary(space, 'select_context'), width)
        self.monster_id = nn.Embedding(_vocabulary(space, 'monster_id'), 32, padding_idx=0)
        self.monster_move = nn.Embedding(_vocabulary(space, 'monster_move'), 32, padding_idx=0)
        self.monster_intent = nn.Embedding(_vocabulary(space, 'monster_intent_cat'), 16, padding_idx=0)
        self.monster_projection = nn.Linear(80 + space['monster_scalars'].shape[-1]
                                            + space['monster_buffs'].shape[-1], width)
        self.relic = nn.Embedding(_vocabulary(space, 'relics'), 48, padding_idx=0)
        self.relic_projection = nn.Linear(49, width)
        self.potion = nn.Embedding(_vocabulary(space, 'potions'), width, padding_idx=0)
        self.scalar_keys = ('player_scalars','player_buffs','combat_counters',
                            'post_enemy_turn_boundary','ascension','gold','pick_remaining',
                            'can_pick_zero','can_pick_any')
        if 'run_keys' in space.spaces:
            self.scalar_keys += ('run_keys',)
        if 'combat_exit_anchor' in space.spaces:
            self.scalar_keys += ('combat_exit_anchor',)
        if config.boss_win_weight:
            self.scalar_keys += ('combat_is_boss',)
        # Additional HP deficit, log turn, and soft threshold features.
        self.player_projection = nn.Linear(sum(prod(space[k].shape) for k in self.scalar_keys)+6, width)
        self.turn_bucket = nn.Embedding(32, width)
        self.register_buffer('hp_thresholds', torch.tensor([.25,.5,.75,1.]), persistent=False)
        self.pool_queries = nn.Parameter(torch.randn(config.pile_queries, width)*.02)
        self.pile_role = nn.Embedding(3, width)
        self.pool_count = nn.Linear(1, width)
        self.role = nn.Embedding(8, width)
        layer = EntityEncoderLayer(width, config.heads, 4*width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.transformer = self.encoder_type(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.pointer_query = nn.Linear(width, width)
        self.pointer_key = nn.Linear(width, width)
        self.end = nn.Linear(width, 1)
        self.autoregressive_actions = C.policy_contract(space) == C.AUTOREGRESSIVE_CONTRACT
        if config.boss_win_weight and not self.autoregressive_actions:
            raise ValueError('boss utility requires autoregressive-v2 actions')
        self.complete_head = CompleteCombatHead(width, autoregressive=self.autoregressive_actions)
        self.distribution = CombatDistributionHead(width, config.quantiles)
        self.transition = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.Linear(width, 6))
        if not config.auxiliary_loss_coef:
            self.transition.requires_grad_(False)
        # Runtime only: each precision owns stable storage referenced by CUDA
        # graphs. Weight refreshes copy into it instead of replacing its address.
        self._card_inference_tables = {}

    def _apply(self, fn, recurse=True):
        self._card_inference_tables.clear()
        return super()._apply(fn, recurse=recurse)

    def _encode_card_table(self):
        if hasattr(self.card, 'vectors'):
            return self.card.vectors
        return self.card.versioned_forward(self.card_ids, self.card_versions).reshape(
            self.n_cards, 2, int(self.card.embedding_dim))

    def inference_card_table(self):
        device = self.card_ids.device
        precision = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
        tensors = tuple(self.card.parameters()) + tuple(self.card.buffers())
        dtype = tensors[0].dtype if tensors else torch.float32
        key = (device, dtype, precision)
        versions = tuple((id(p), p._version) for p in tensors)
        entry = self._card_inference_tables.get(key)
        if entry is None or entry[0] != versions:
            # Normal (non-inference) storage permits in-place refresh under
            # either no_grad or inference_mode and stays outside graph pools.
            with torch.inference_mode(False), torch.no_grad():
                table = self._encode_card_table()
                if entry is not None:
                    entry[1].copy_(table)
                    table = entry[1]
            self._card_inference_tables[key] = (versions, table)
        return self._card_inference_tables[key][1]

    def _card_tokens(self, table, ids, upgrades):
        # The two-dimensional advanced index scatters gradients into a tiny
        # card table from every one of the 96 padded choice slots.  Embedding
        # uses the dedicated table-gradient kernel for the same lookup.
        index = ids.long() * 2 + upgrades.gt(0).long()
        vectors = F.embedding(index, table.reshape(-1, table.shape[-1]))
        return self.card_projection(torch.cat((vectors, upgrades.float().unsqueeze(-1)), -1))

    def _pool(self, entities, counts, context, *, logits=None):
        if logits is None:
            query = context[:, None] + self.pool_queries[None]
            if entities.ndim == 2:
                logits = torch.einsum('bkd,nd->bkn', query, entities) * self.config.width**-.5
            else:
                logits = torch.einsum('bkd,bnd->bkn', query, entities) * self.config.width**-.5
        weights = counts.float().clamp_min(0)
        scores = logits.float() + weights.clamp_min(1e-30).log()[:, None]
        probability = scores.masked_fill(weights[:, None] == 0, -1e9).softmax(-1)
        probability = probability * (weights[:, None] > 0)
        if entities.ndim == 2:
            pooled = torch.einsum('bkn,nd->bkd', probability.to(entities.dtype), entities)
        else:
            pooled = torch.einsum('bkn,bnd->bkd', probability.to(entities.dtype), entities)
        return pooled + self.pool_count(weights.sum(-1, keepdim=True).log1p())[:, None]

    def encode(self, obs, active_relic_width=None, *, card_table=None):
        batch = obs['player_scalars'].shape[0]
        table = card_table
        if table is None:
            table = (self._encode_card_table() if self.training or torch.is_grad_enabled()
                     else self.inference_card_table())
        hp = obs['player_scalars'].float()
        turn = hp[:, 4:5] * encode.TURN_SCALE
        extra = torch.cat((hp[:, 1:2]-hp[:, :1], turn.clamp_min(0).log1p()/4,
                           torch.sigmoid((hp[:, 5:6]-self.hp_thresholds)*12)), -1)
        global_token = self.player_projection(torch.cat(
            [obs[k].float().reshape(batch,-1) for k in self.scalar_keys]+[extra],-1))
        global_token = global_token + self.turn_bucket(turn.long().flatten().clamp(0,31))
        global_token = global_token + self.choice_reason(obs['select_context'].long().flatten())
        hand = self._card_tokens(table, obs['hand_ids'], obs['hand_upgrades'])
        hand = hand + self.hand_context(torch.stack([obs[k] for k in
            ('hand_costs','hand_damage','hand_block','select_hand_mask')], -1).float())
        monster = self.monster_projection(torch.cat((
            self.monster_id(obs['monster_id'].long()), self.monster_move(obs['monster_move'].long()),
            self.monster_intent(obs['monster_intent_cat'].long()),
            obs['monster_scalars'].float(), obs['monster_buffs'].float()), -1))
        width = obs['relics'].shape[1] if active_relic_width is None else max(1,active_relic_width)
        relic_ids = obs['relics'][:, :width].long()
        relic = self.relic_projection(torch.cat((self.relic(relic_ids),
            obs['relic_counters'][:, :width, None].float()), -1))
        potion = self.potion(obs['potions'].long())
        choices = self._card_tokens(table, obs['select_card_ids'], obs['select_card_upgrades'])
        # Only handle presence is public neural input; the numeric handle is execution metadata.
        valid_choices = obs['select_native_indices'] > 0
        choices = choices + self.choice_context(torch.stack((obs['select_card_costs'].float(),
            (valid_choices & (obs['select_card_ids'] == 0)).float()), -1))
        choices = choices + self.choice_zone(obs['select_zone'].long().flatten())[:, None]
        choice_pool = self._pool(choices, valid_choices, global_token)
        pile_tokens = []
        all_ids = self.card_ids.reshape(1,-1)
        all_versions = self.card_versions.reshape(1,-1).float()/encode.UPGRADE_SCALE
        pile_entities = self._card_tokens(table, all_ids, all_versions)[0]
        pile_query = global_token[:, None] + self.pool_queries[None]
        pile_logits = torch.einsum('bkd,nd->bkn', pile_query, pile_entities) * self.config.width**-.5
        for pile_index, pile in enumerate(('draw', 'discard', 'exhaust')):
            counts = obs[pile+'_counts'].float()
            upgraded = torch.zeros_like(counts)
            if pile+'_card_upgraded_counts' in obs:
                upgraded = upgraded.scatter_add(1, obs[pile+'_card_ids'].long(),
                                                 obs[pile+'_card_upgraded_counts'].float())
            versions = torch.stack(((counts-upgraded).clamp_min(0),upgraded),-1).flatten(1)
            pile_tokens.append(self._pool(pile_entities, versions, global_token, logits=pile_logits)
                               + self.pile_role.weight[pile_index])
        piles = torch.cat(pile_tokens, 1)
        blocks = [('global',global_token[:,None],torch.zeros(batch,1,dtype=torch.bool,device=hp.device)),
                  ('hand',hand,obs['hand_ids']==0), ('monster',monster,obs['monster_id']==0),
                  ('potion',potion,obs['potions']==0), ('relic',relic,relic_ids==0),
                  ('pile',piles,torch.zeros(piles.shape[:2],dtype=torch.bool,device=hp.device)),
                  ('choice',choice_pool,torch.zeros(choice_pool.shape[:2],dtype=torch.bool,device=hp.device))]
        tokens, masks, spans, offset = [], [], {}, 0
        for role, (name, block, mask) in enumerate(blocks):
            tokens.append(block + self.role.weight[role]); masks.append(mask)
            spans[name] = (offset, offset+block.shape[1]); offset += block.shape[1]
        encoded = self._encode_tokens(torch.cat(tokens, 1), torch.cat(masks, 1), spans, obs)
        result = {name:encoded[:,lo:hi] for name,(lo,hi) in spans.items()}
        result['select'] = choices
        return result['global'][:,0], result

    def _encode_tokens(self, source, pad_mask, spans, obs):
        return self.norm(self.transformer(source, src_key_padding_mask=pad_mask))

    def _pointer(self, state, entities):
        return torch.einsum('bd,bnd->bn',self.pointer_query(state),self.pointer_key(entities))*self.config.width**-.5

    def _value(self, state, obs):
        values, outputs = self.distribution(state)
        if self.config.boss_win_weight:
            from env.core.boss_value import win_hp_utility
            weight = obs['combat_is_boss'].float() * self.config.boss_win_weight
            values = win_hp_utility(outputs['win_logits'].sigmoid(),
                                    outputs['hp_quantiles'].mean(-1, keepdim=True), weight)
            outputs['boss_win_weight'] = weight
        return values, outputs

    def forward(self, obs, *, active_relic_width=None, evaluate_subsets=True, auxiliary_actions=None,
                card_table=None):
        state, blocks = self.encode(obs, active_relic_width, card_table=card_table)
        if self.autoregressive_actions:
            pointers = [self._pointer(state, blocks[role]) for role in ('hand', 'potion', 'select')]
            if '_resident_rows' in obs and blocks['select'].shape[1] < 96:
                # Old packed heads score real queries before restoring native
                # action slots. A zero entity still has pointer_key's bias;
                # preserve even masked logits without expanding 96x64 tokens.
                padding = self._pointer(state, state.new_zeros((len(state), 1, state.shape[-1])))
                pointers[-1] = torch.cat((pointers[-1], padding.expand(-1, 96-pointers[-1].shape[1])), -1)
            source = torch.cat((self.end(state), *pointers), -1)
        else:
            source = state.new_zeros((state.shape[0], C.JOINT_END))
            source[:, A.END_TURN] = self.end(state).flatten()
            for start, role in ((A._PLAY,'hand'),(A._POTION,'potion'),(A._PICK,'select')):
                source[:,start:start+blocks[role].shape[1]] = self._pointer(state,blocks[role])
        logits = self.complete_head(source,state,torch.cat((blocks['hand'],blocks['potion']),1),
            blocks['monster'],obs['combat_legal'],evaluate_subsets=evaluate_subsets,
            selected=obs['select_hand_mask'], remaining=(obs['pick_remaining']*encode.PICK_SCALE).round())
        values, outputs = self._value(state, obs)
        if auxiliary_actions is not None and self.config.auxiliary_loss_coef:
            actions = auxiliary_actions.long() + (C.JOINT_END if self.autoregressive_actions else 0)
            row = torch.arange(len(actions),device=actions.device)
            card = (actions >= C.JOINT_PLAY) & (actions < C.JOINT_POTION)
            potion = (actions >= C.JOINT_POTION) & (actions < C.JOINT_PICK)
            pick = (actions >= C.JOINT_PICK) & (actions < C.JOINT_MULTI)
            multi = actions >= C.JOINT_MULTI
            origin = torch.where(potion,C.JOINT_POTION,C.JOINT_PLAY)
            slot = ((actions-origin)//6).clamp(0,9)
            target = ((actions-origin)%6).clamp(0,4)
            entity = blocks['hand'][row,slot]*card[:,None]
            entity = entity + blocks['potion'][row,slot.clamp_max(4)]*potion[:,None]
            entity = entity + blocks['select'][row,(actions-C.JOINT_PICK).clamp(0,95)]*pick[:,None]
            if self.autoregressive_actions:
                selected = F.one_hot((actions-C.JOINT_MULTI).clamp(0,10), 11)[:, :10].to(state.dtype)
                selected = torch.where((actions==C.AUTOREGRESSIVE_STOP+C.JOINT_END)[:,None],
                    obs['select_hand_mask'].to(state.dtype), selected)
            else:
                bits = (actions-C.JOINT_MULTI).clamp_min(0)
                selected = ((bits[:,None] >> torch.arange(10,device=actions.device)) & 1).to(state.dtype)
            entity = entity + torch.einsum('bh,bhd->bd',selected,blocks['hand'])*multi[:,None]
            targeted = (card|potion) & ((actions-origin)%6 != 5)
            destination = blocks['monster'][row,target]*targeted[:,None]
            outputs['transition'] = self.transition(torch.cat((state,entity,destination),-1))
        return logits, values, outputs


class EntityCombatCore(nn.Module):
    combat_observation_keys = COMBAT_OBSERVATION_KEYS
    training_objective = 'combat-outcome-quantiles-v1'
    combat_only = True
    joint_actions = True
    map_dag_enabled = False

    def __init__(self, space, config):
        super().__init__()
        if 'combat_legal' not in space.spaces:
            raise ValueError('entity-combat requires joint-v2 actions')
        has_card_language = 'draw_card_upgraded_counts' in space.spaces
        if config.card_vector_checkpoint is None and not has_card_language:
            raise ValueError('entity-combat requires card-language observations or frozen card vectors')
        if config.card_vector_checkpoint is not None and has_card_language:
            raise ValueError('frozen card vectors require card-language observations to be disabled')
        self.config = config
        self.autoregressive_actions = C.policy_contract(space) == C.AUTOREGRESSIVE_CONTRACT
        card_embedding = None
        if config.card_vector_checkpoint is not None:
            card_embedding = load_frozen_card_vector_table(
                config.card_vector_checkpoint,
                expected_num_cards=_vocabulary(space, 'hand_ids'),
            )
        self.combat = EntityCombatNetwork(space, config, card_embedding=card_embedding)

    def prepare_inference_cache(self):
        self.combat.inference_card_table()

    def disable_transition_auxiliary(self):
        self.config = replace(self.config, auxiliary_loss_coef=0.0)
        self.combat.config = self.config
        self.combat.transition.requires_grad_(False)

    def policy_only_parameters(self):
        """Heads exclusive to action logits; the encoder remains shared."""
        for module in (self.combat.pointer_query, self.combat.pointer_key,
                       self.combat.end, self.combat.complete_head):
            yield from module.parameters()

    def combat_actor_critic(self, obs, **kwargs):
        logits, values, _ = self.combat(obs, **kwargs)
        return logits, values

    def graph_actor_critic(self, obs, active_relic_width, card_table):
        """Pure leaf forward; the graph owner supplies refreshed stable storage.

        Keeping cache/version bookkeeping outside this function permits whole
        forward fusion. Training and ordinary prediction keep their eager path.
        """
        logits, values, _ = self.combat(obs, active_relic_width=active_relic_width,
            evaluate_subsets=False, card_table=card_table)
        return logits, values

    def combat_training_outputs(self, obs, actions, **kwargs):
        return self.combat(obs, auxiliary_actions=actions if self.config.auxiliary_loss_coef else None,
                           **kwargs)

    def actor_critic(self, obs, **kwargs):
        # The collector may batch noncombat rows before applying its rules.
        # Those rows have neutral outputs and no learned overworld parameters.
        logits, values = self.combat_actor_critic(obs,
            active_relic_width=kwargs.get('combat_relic_width'))
        combat = obs['phase'].flatten() == V.PHASE_VOCAB_INDEX['combat']
        return torch.where(combat[:,None],logits,torch.zeros_like(logits)), values*combat[:,None]

    def action_logits(self, obs):
        return self.actor_critic(obs)[0]

    def value(self, obs):
        if self.autoregressive_actions:
            indices = (obs['phase'].flatten() == V.PHASE_VOCAB_INDEX['combat']).nonzero().flatten()
            result = obs['player_scalars'].new_zeros((len(obs['phase']), 1), dtype=torch.float32)
            if indices.numel():
                _, values = self.combat_actor_critic({k: v[indices] for k, v in obs.items()})
                result[indices] = values.float()
            return result
        return self.actor_critic(obs)[1]


class EntityCombatPolicy(SplitStSPointerPolicy):
    def __init__(self,*args,entity_config=None,**kwargs):
        self.entity_config = asdict(EntityCombatConfig(**(entity_config or {})))
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        if self.action_space.n != C.policy_size(C.policy_contract(self.observation_space)):
            raise ValueError('unsupported intermediate entity action layout; use a fresh current model')
        self.core = EntityCombatCore(self.observation_space,EntityCombatConfig(**self.entity_config))
        self.optimizer = self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def disable_transition_auxiliary(self):
        self.entity_config['auxiliary_loss_coef'] = 0.0
        self.core.disable_transition_auxiliary()

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),entity_config=self.entity_config)


def disable_entity_transition_auxiliary(model):
    """Change a loaded model's objective while keeping SB3 save kwargs aligned."""
    if not isinstance(model.policy, EntityCombatPolicy):
        raise TypeError('transition auxiliary requires EntityCombatPolicy')
    model.policy.disable_transition_auxiliary()
    model.policy_kwargs = dict(model.policy_kwargs,
                               entity_config=dict(model.policy.entity_config))
