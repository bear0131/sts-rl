"""Device projection of the versioned compact battle observation.

Vocabulary, scales and printed card tables come from the canonical adapter.
Run fields remain owned by the root; live resources and permanent card growth
are replaced from the resident branch. No state transition or RNG lives here.
"""
import numpy as np
import torch

from . import encode as E
from .outer_return import MISC_SCALE
from .run_env import _nosl_compact_lut, _nosl_card_stats


def canonical_misc(misc):
    """Match the native observation's integer / scale -> FP32 conversion.

    FP32 reciprocal multiplication differs by one ULP for values such as
    5/100, spuriously invalidating an unchanged root's expensive context.
    """
    return (misc.to(torch.float64) / MISC_SCALE).to(torch.float32)


class CudaObservationDecoder:
    def __init__(self, space, device='cuda'):
        self.device, self.space = torch.device(device), space
        lut = _nosl_compact_lut()
        self.layout = layout = lut['layout']
        self.tables = {k: self.tensor(lut[k]) for k in
                       ('card', 'monster', 'move', 'potion', 'select_context')}
        self.ps, self.ms = layout['player_status_count'], layout['monster_status_count']
        self.cards = layout['card_count']
        self.hand_start = 6 + self.ps + E.COMBAT_COUNTERS + 1 + 30 + 5*self.ms
        self.pile_start = self.hand_start + 20
        self.select_start = self.pile_start + 3*self.cards
        self.status = {}
        for name in ('player', 'monster'):
            # Multiple native aliases follow the canonical decoder's last write.
            sources, targets = lut[name + '_status_columns']
            pairs = dict(zip(targets, sources))
            self.status[name] = (self.tensor(list(pairs.values())), self.tensor(list(pairs)))
        self.binary = self.tensor(lut['binary_counters'])
        stats = _nosl_card_stats()
        self.damage = self.tensor(stats['damage'][:, :2])
        self.block = self.tensor(stats['block'][:, :2])
        self.searing = E._card_id_idx('Searing Blow')
        self.bits = self.tensor(1 << np.arange(10))
        size = int(space['relics'].high.max()) + 1
        columns, add, mod, stop, scale = (np.zeros(size, np.float32) for _ in range(5))
        live = np.zeros(size, bool)
        scale.fill(E.COUNTER_SCALE)
        for rid, col, modulo, until in zip(lut['live_relic_ids'], lut['live_relic_counter_columns'],
                lut['live_relic_counter_moduli'], lut['live_relic_counter_stop_turns'], strict=True):
            live[rid] = True
            columns[rid] = (6+self.ps+col if col >= 0 else 4 if col == -3 else
                           self.select_start+200 if col == -4 else self.select_start+201
                           if col == -5 else self.select_start+197-col)
            add[rid] = col == -3
            mod[rid], stop[rid] = modulo, until
            if col in (-1, -2):
                scale[rid] = 1
        self.live, self.columns = self.tensor(live), self.tensor(columns.astype(np.int64))
        self.add, self.mod, self.stop, self.scale = map(self.tensor, (add, mod, stop, scale))

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value), device=self.device)

    def relic_counters(self, scalars, ids, original):
        ids = ids.long()
        return self._relic_counters(scalars.gather(1, self.columns[ids]), ids, original)

    def _relic_counters(self, counter, ids, original):
        counter = counter+self.add[ids]
        counter = torch.where(self.mod[ids] != 0, torch.fmod(counter, self.mod[ids].clamp_min(1)), counter)
        counter = torch.where((self.stop[ids] != 0) & (counter >= self.stop[ids]), 0, counter)/self.scale[ids]
        return torch.where(self.live[ids], counter, original)

    def anchor(self, raw, roots, owners):
        """Only the exact pre-reward head consumers, without combat encoding."""
        s = raw['scalars']
        ids = roots['relics'][owners]
        return dict(player_scalars=s[:, :2].clamp_max(E.HP_SCALE)/E.HP_SCALE,
            gold=s[:, 5:6]/E.GOLD_SCALE, run_keys=roots['run_keys'][owners],
            potions=self.tables['potion'][raw['categorical'][:, 25:30].long()],
            combat_exit_anchor=torch.stack((raw['outcome'] == 1, raw['outcome'] == 3), 1).float(),
            relics=ids, relic_counters=self.relic_counters(s, ids, roots['relic_counters'][owners]))

    def anchor_indexed(self, raw, roots, owners, rows, scalar_rows):
        """Gather only head consumers, including from a reused compact frame."""
        ids = roots['relics'][owners]
        counters = raw['scalars'][scalar_rows[:, None], self.columns[ids.long()]]
        status = raw['outcome'][rows]
        return dict(player_scalars=raw['scalars'][scalar_rows, :2].clamp_max(E.HP_SCALE)/E.HP_SCALE,
            gold=raw['scalars'][scalar_rows, 5:6]/E.GOLD_SCALE, run_keys=roots['run_keys'][owners],
            potions=self.tables['potion'][raw['categorical'][rows, 25:30].long()],
            combat_exit_anchor=torch.stack((status == 1, status == 3), 1).float(),
            relics=ids, relic_counters=self._relic_counters(counters, ids.long(), roots['relic_counters'][owners]))

    def __call__(self, raw, roots, owners, *, head_only=False, choice_width=96, include_piles=True):
        if not 1 <= choice_width <= 96 or (not head_only and (choice_width != 96 or not include_piles)):
            raise ValueError('only the frozen head may narrow trailing selection padding')
        c, s, n = raw['categorical'].long(), raw['scalars'], len(owners)
        # Only public context and the head consumers, not unused reward pages.
        keys = ('phase', 'ascension', 'act', 'floor', 'next_boss', 'character', 'run_keys',
                'combat_is_boss', 'deck_ids', 'deck_upgrades', 'outer_deck_misc', 'relics',
                'relic_counters', 'map_full_rooms', 'map_full_features')
        if head_only:
            keys = ('ascension', 'run_keys', 'combat_is_boss', 'relics', 'relic_counters')
        o = {k: roots[k][owners] for k in keys if k in roots}
        def zeros(key):
            return torch.zeros((n, *self.space[key].shape), device=s.device, dtype=torch.float32)
        def lookup(key, cols):
            return self.tables[key][c[:, cols]]
        o.update(hand_ids=lookup('card', slice(0, 10)), monster_id=lookup('monster', slice(10, 15)),
                 monster_move=lookup('move', slice(15, 20)), monster_intent_cat=c[:, 20:25].clone(),
                 potions=lookup('potion', slice(25, 30)), select_context=lookup('select_context', slice(30, 31)),
                 select_card_ids=lookup('card', slice(31, 31+choice_width)),
                 select_native_indices=c[:, 127:127+choice_width],
                 select_zone=c[:, 223:224], combat_legal=raw['mask'].bool(), gold=s[:, 5:6]/E.GOLD_SCALE)
        o['player_scalars'] = torch.stack((s[:, 0].clamp_max(E.HP_SCALE)/E.HP_SCALE,
            s[:, 1].clamp_max(E.HP_SCALE)/E.HP_SCALE, s[:, 2]/E.BLOCK_SCALE,
            s[:, 3]/E.ENERGY_SCALE, (s[:, 4]+1)/E.TURN_SCALE,
            torch.where(s[:, 1] != 0, s[:, 0]/s[:, 1], 0)), 1)
        o['player_buffs'] = zeros('player_buffs')
        src, dst = self.status['player']
        o['player_buffs'][:, dst] = s[:, 6+src]/E.BUFF_SCALE
        counters = s[:, 6+self.ps:6+self.ps+E.COMBAT_COUNTERS]
        o['combat_counters'] = counters/E.COUNTER_SCALE
        o['combat_counters'][:, self.binary] = (counters[:, self.binary] != 0).float()
        start = 6+self.ps+E.COMBAT_COUNTERS
        o['post_enemy_turn_boundary'] = s[:, start:start+1]
        m = s[:, start+1:start+31].reshape(n, 5, 6)
        o['monster_scalars'] = torch.stack((m[:, :, 0], m[:, :, 1].clamp_max(E.HP_SCALE)/E.HP_SCALE,
            torch.where(m[:, :, 2] != 0, m[:, :, 1]/m[:, :, 2], 0), m[:, :, 3]/E.BLOCK_SCALE,
            m[:, :, 4]/E.DMG_SCALE, m[:, :, 5]/E.HITS_SCALE), -1)
        o['monster_buffs'] = zeros('monster_buffs')
        src, dst = self.status['monster']
        powers = s[:, start+31:self.hand_start].reshape(n, 5, self.ms)
        o['monster_buffs'][:, :, dst] = torch.where(m[:, :, :1] > 0, powers[:, :, src]/E.BUFF_SCALE, 0)
        explosion = o['monster_move'] == E.V.MOVE_VOCAB_INDEX['EXPLODER_EXPLODE']
        o['monster_intent_cat'].masked_fill_(explosion, 0)
        o['monster_scalars'][:, :, 4:6].masked_fill_(explosion[:, :, None], 0)
        hand = s[:, self.hand_start:self.pile_start].reshape(n, 10, 2)
        up = hand[:, :, 0]
        o['hand_upgrades'] = up/E.UPGRADE_SCALE
        o['hand_costs'] = torch.where(hand[:, :, 1] == -3, -2, hand[:, :, 1])/E.ENERGY_SCALE
        ids, levels = o['hand_ids'], (up > 0).long()
        o['hand_damage'] = torch.where(ids == self.searing, 12+up*(up+7)/2, self.damage[ids, levels])/E.DMG_SCALE
        o['hand_block'] = self.block[ids, levels]/E.BLOCK_SCALE
        if include_piles:
            pile = s[:, self.pile_start:self.select_start].reshape(n, 3, self.cards)
            for i, name in enumerate(('draw', 'discard', 'exhaust')):
                o[name+'_counts'] = pile[:, i]
            o['pile_upgraded_counts'] = raw['upgrades'][:, :3]
            o['pile_upgrade_sums'] = raw['upgrades'][:, 3:]/E.UPGRADE_SCALE
        selection = s[:, self.select_start:]
        o['select_card_upgrades'] = selection[:, 4:4+choice_width]/E.UPGRADE_SCALE
        cost = selection[:, 100:100+choice_width]
        o['select_card_costs'] = torch.where(cost == -3, -2, cost)/E.ENERGY_SCALE
        o['select_hand_mask'] = ((raw['prefix'][:, None] & self.bits) != 0).float()
        o['pick_remaining'] = (selection[:, :1]-o['select_hand_mask'].sum(1, keepdim=True)).clamp_min(0)/E.PICK_SCALE
        o['can_pick_zero'], o['can_pick_any'] = selection[:, 1:2], selection[:, 2:3]
        if not head_only:
            o['selected_card_ids'] = zeros('selected_card_ids').long()
            o['selected_card_ids'][:, :2] = lookup('card', slice(224, 226))
            o['selected_card_upgrades'] = zeros('selected_card_upgrades')
            o['selected_card_upgrades'][:, :2] = selection[:, 196:198]/E.UPGRADE_SCALE
        o['relic_counters'] = self.relic_counters(s, o['relics'], o['relic_counters'])
        if not head_only:
            misc = raw['permanent_misc']
            o['outer_deck_misc'] = torch.where((misc >= 0) & (o['deck_ids'] != 0), canonical_misc(misc), o['outer_deck_misc'])
        o['combat_exit_anchor'] = torch.stack((raw['outcome'] == 1, raw['outcome'] == 3), 1).float()
        return o
