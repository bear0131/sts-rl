"""Public entity memory and uniform action queries, independently owned per head.

Contract upper-latent-actions-v1 preserves the existing observation/action ABI.
Learned latent queries bound attention cost; no tactical/resource scores enter.
"""
import torch
from torch import nn
from torch.nn import functional as F

from env.core import action_space as A, encode, vocab as V


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width, bias=False)
        self.out = nn.Linear(width, width)

    def forward(self, query, memory, valid=None):
        b, n, d = query.shape
        q = self.q(query).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, -1)
        k, v = (x.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2) for x in (k, v))
        mask = None if valid is None else valid[:, None, None, :]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.attn = Attention(width, heads)
        self.ffnorm = nn.RMSNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(2 * width, width)

    def forward(self, x):
        normalized = self.norm(x)
        x = x + self.attn(normalized, normalized)
        a, b = self.up(self.ffnorm(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class PublicTokens(nn.Module):
    """One fact codec for all screens; no rank/order embedding for set entities."""
    cards = (
        ('deck', 'deck_ids', 'deck_upgrades', None, None),
        ('offer', 'offer_card_ids', 'offer_card_upgrades', 'combat_reward', None),
        ('select', 'select_card_ids', 'select_card_upgrades', 'card_select', None),
        ('selected', 'selected_card_ids', 'selected_card_upgrades', 'card_select', None),
        ('shop_card', 'shop_card_ids', 'shop_card_upgrades', 'shop', 'shop_card_prices'),
    )
    entities = (
        ('relic', 'relics', 'relic', None, 'relic_counters'),
        ('potion', 'potions', 'potion', None, None),
        ('reward_relic', 'reward_relic_ids', 'relic', 'combat_reward', None),
        ('reward_potion', 'reward_potion_ids', 'potion', 'combat_reward', None),
        ('shop_relic', 'shop_relic_ids', 'relic', 'shop', 'shop_relic_prices'),
        ('shop_potion', 'shop_potion_ids', 'potion', 'shop', 'shop_potion_prices'),
        ('boss_relic', 'boss_relic_ids', 'relic', 'boss_relic', None),
    )
    categories = ('phase', 'arg_step', 'character', 'select_context', 'event_id', 'next_boss')
    scalars = ('run_hp', 'ascension', 'act', 'floor', 'gold', 'pick_remaining',
               'can_pick_zero', 'can_pick_any', 'shop_remove_cost', 'run_keys', 'deck_size')

    def __init__(self, space, width):
        super().__init__()
        self.space = space
        vocab = lambda key: int(space[key].high.max()) + 1
        self.ids = nn.ModuleDict({name: nn.Embedding(vocab(key), width, padding_idx=0)
            for name, key in [('card', 'deck_ids'), ('relic', 'relics'), ('potion', 'potions'),
                             ('room', 'map_full_rooms'), *[(k, k) for k in self.categories]]})
        self.scalar_keys = [k for k in self.scalars if k in space.spaces]
        self.global_values = nn.Linear(sum(space[k].shape[0] for k in self.scalar_keys), width)
        self.number = nn.Linear(2, width, bias=False)
        self.map_values = nn.Linear(12, width)
        self.map_child = nn.Linear(width, width, bias=False)
        self.map_parent = nn.Linear(width, width, bias=False)
        self.roles = nn.Embedding(24, width)
        self.potion_slot = nn.Embedding(A.P_MAX, width)
        self.event_ids = nn.Embedding(vocab('event_option_ids'), width, padding_idx=0)
        self.event_values = nn.Linear(2, width)
        self.neow_bonus = nn.Embedding(vocab('neow_bonus_ids'), width, padding_idx=0)
        self.neow_drawback = nn.Embedding(vocab('neow_drawback_ids'), width, padding_idx=0)
        self.neow_values = nn.Linear(2, width)
        self.details = 'event_choice_kinds' in space.spaces
        if self.details:
            self.event_kind = nn.Embedding(len(encode.EVENT_CHOICE_KINDS), width)
            self.event_detail = nn.Linear(4, width)

    def forward(self, obs):
        phase = obs['phase'].long().flatten()
        b, device = len(phase), phase.device
        def get(key):
            if key in obs:
                return obs[key]
            return torch.zeros((b, *self.space[key].shape), device=device)
        def active(screen):
            return torch.ones(b, device=device, dtype=torch.bool) if screen is None else phase == V.PHASE_VOCAB_INDEX[screen]
        blocks, validity, named = [], [], {}
        def add(name, token, valid, role):
            token = token + self.roles.weight[role]
            named[name] = token
            blocks.append(token)
            validity.append(valid)
        state = self.global_values(torch.cat([get(k).float() for k in self.scalar_keys], -1))
        for key in self.categories:
            state = state + self.ids[key](get(key).long().flatten())
        add('global', state[:, None], torch.ones(b, 1, device=device, dtype=torch.bool), 0)
        for role, (name, key, upgrade, screen, price) in enumerate(self.cards, 1):
            ids = get(key).long()
            values = torch.stack((get(upgrade).float(), get(price).float() if price else torch.zeros_like(ids)), -1)
            token = self.ids['card'](ids) + self.number(values)
            add(name, token, ids.ne(0) & active(screen)[:, None], role)
        for role, (name, key, kind, screen, number) in enumerate(self.entities, 6):
            ids = get(key).long()
            numeric = get(number).float() if number else torch.zeros_like(ids, dtype=torch.float32)
            token = self.ids[kind](ids) + self.number(torch.stack((numeric, torch.zeros_like(numeric)), -1))
            if name == 'potion':
                token = token + self.potion_slot.weight[None, :ids.shape[1]]
            add(name, token, ids.ne(0) & active(screen)[:, None], role)
        gold = get('reward_gold').float()
        add('reward_gold', self.number(torch.stack((gold, torch.zeros_like(gold)), -1)),
            gold.gt(0) & active('combat_reward')[:, None], 13)
        keys = get('reward_keys').float()
        add('reward_key', self.number(keys)[:, None], keys.ne(0).any(-1, keepdim=True) & active('combat_reward')[:, None], 14)
        rooms, features = get('map_full_rooms').long(), get('map_full_features').float()
        room = self.ids['room'](rooms) + self.map_values(features)
        grid = room.reshape(b, encode.MAP_ROWS, encode.MAP_COLS, -1)
        edges = features[..., 5:].reshape(b, encode.MAP_ROWS, encode.MAP_COLS, encode.MAP_COLS)
        children = F.pad(grid[:, 1:], (0, 0, 0, 0, 0, 1))
        child = torch.einsum('byxc,bycd->byxd', edges, children)
        parent = F.pad(torch.einsum('byxc,byxd->bycd', edges[:, :-1], grid[:, :-1]), (0, 0, 0, 0, 1, 0))
        room = room + self.map_child(child.flatten(1, 2)) + self.map_parent(parent.flatten(1, 2))
        add('map', room, rooms.ne(0), 15)
        # Each column action refers to the actual choice node, including boss exit.
        choice = features[..., 3:4]
        named['map_action'] = (room * choice).reshape(b, encode.MAP_ROWS, encode.MAP_COLS, -1).sum(1)
        event_ids = get('event_option_ids').long()
        event = self.event_ids(event_ids) + self.event_values(get('event_option_features').float())
        event = event + self.ids['event_id'](get('event_option_event_ids').long())
        if self.details:
            event = event + self.event_kind(get('event_choice_kinds').long()) + self.event_detail(get('event_choice_features').float())
            for key, kind in [('event_choice_card_ids', 'card'), ('event_choice_relic_ids', 'relic'), ('event_choice_potion_ids', 'potion')]:
                ids = get(key).long()
                event = event + self.ids[kind](ids) * ids.ne(0)[..., None]
            upgrades = get('event_choice_card_upgrades').float()
            event = event + self.number(torch.stack((upgrades, torch.zeros_like(upgrades)), -1))
        add('event', event, event_ids.ne(0) & active('event')[:, None], 16)
        bonus = get('neow_bonus_ids').long()
        neow = self.neow_bonus(bonus) + self.neow_drawback(get('neow_drawback_ids').long()) + self.neow_values(get('neow_features').float())
        add('neow', neow, bonus.ne(0) & active('event')[:, None], 17)
        tokens, valid = torch.cat(blocks, 1), torch.cat(validity, 1)
        # Batch-wide compaction preserves internal slots and named action references.
        keep = valid.any(0)
        return tokens[:, keep], valid[:, keep], named


class UnifiedUpperNetwork(nn.Module):
    def __init__(self, space, cfg, *, actor):
        super().__init__()
        self.actor = actor
        self.tokens = PublicTokens(space, cfg.width)
        self.latents = nn.Parameter(torch.randn(cfg.latents, cfg.width) * .02)
        self.input_norm = nn.RMSNorm(cfg.width)
        self.read = Attention(cfg.width, cfg.heads)
        self.blocks = nn.Sequential(*[Block(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.norm = nn.RMSNorm(cfg.width)
        if actor:
            # Slot-independent operation types for entity actions; fixed operations
            # (rest/smith/etc.) have distinct meanings, never a separate output head.
            operations, next_id = [], 0
            for name, count in A._SEGMENTS:
                distinct = name in ('CAMPFIRE', 'CHEST')
                operations.extend(range(next_id, next_id + count) if distinct else [next_id] * count)
                next_id += count if distinct else 1
            self.register_buffer('operations', torch.tensor(operations))
            self.operation = nn.Embedding(next_id, cfg.width)
            self.action_read = Attention(cfg.width, cfg.heads)
            self.action_norm = nn.RMSNorm(cfg.width)
            self.score = nn.Linear(cfg.width, 1, bias=False)
            self.score.weight.data.mul_(.01)
        else:
            self.score = nn.Linear(cfg.width, 1)

    def forward(self, obs):
        memory, valid, named = self.tokens(obs)
        latent = self.latents[None].expand(len(memory), -1, -1)
        latent = self.norm(self.blocks(latent + self.read(latent, self.input_norm(memory), valid)))
        if not self.actor:
            return self.score(latent.mean(1))
        actions = self.operation(self.operations)[None].expand(len(memory), -1, -1).clone()
        mapping = {'PICK_CARD':'select', 'USE_POTION':'potion', 'MAP_NODE':'map_action',
            'REWARD_GOLD':'reward_gold', 'REWARD_RELIC':'reward_relic', 'REWARD_POTION':'reward_potion',
            'REWARD_CARD':'offer', 'REWARD_KEY':'reward_key', 'BOSS_RELIC':'boss_relic',
            'SHOP_CARD':'shop_card', 'SHOP_RELIC':'shop_relic', 'SHOP_POTION':'shop_potion',
            'EVENT_OPT':'event', 'NEOW':'neow'}
        for segment, name in mapping.items():
            token = named[name]
            start = A.seg_base(segment)
            actions[:, start:start + token.shape[1]] += token
        actions = actions + self.action_read(self.action_norm(actions), latent)
        logits = self.score(self.action_norm(actions)).squeeze(-1)
        return F.pad(logits, (0, 198 - A.N_ACTIONS), value=-1e9)
