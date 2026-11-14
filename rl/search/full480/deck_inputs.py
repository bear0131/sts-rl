"""Frozen current-consumer tables for the original GPU observer and packer.

This module owns no states, search scheduling, policy sampling or rollout loop.
The canonical codec remains the source of vocabulary, scales and native fields.
"""
from pathlib import Path
import hashlib
import re

import numpy as np
import torch
from torch.nn import functional as F

from env.core import encode
from env.core.run_env import _nosl_compact_lut, _nosl_card_stats


class DeckInputTables:
    def __init__(self, core, prototypes):
        self.net = core.combat
        self.prototypes = prototypes
        self.tables = {}
        self.defines = {}
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        n = self.net
        device = next(n.parameters()).device
        if n.training or device.type != 'cuda':
            raise ValueError('full480 input tables require frozen CUDA weights')
        lut = _nosl_compact_lut()
        arrays = {}
        constants = {}
        for key in ('card', 'monster', 'move', 'potion', 'select_context',
                    'live_relic_ids', 'live_relic_counter_columns',
                    'live_relic_counter_moduli', 'live_relic_counter_stop_turns'):
            arrays[key] = torch.as_tensor(lut[key], device=device, dtype=torch.int32)
        for key in ('player_status_columns', 'monster_status_columns'):
            native, target = lut[key]
            width = self.prototypes['player_buffs' if key.startswith('player') else 'monster_buffs'].shape[-1]
            reverse = np.full(width, -1, np.int32)
            reverse[target] = native
            arrays[key] = torch.as_tensor(reverse, device=device)
        binary = np.zeros(encode.COMBAT_COUNTERS, np.int32)
        binary[lut['binary_counters']] = 1
        arrays['binary_counters'] = torch.as_tensor(binary, device=device)
        constants['LIVE_RELIC_COUNT'] = len(lut['live_relic_ids'])
        constants['RELIC_WIDTH'] = self.prototypes['relics'].shape[-1]
        constants['PLAYER_BUFF_WIDTH'] = self.prototypes['player_buffs'].shape[-1]
        constants['MONSTER_BUFF_WIDTH'] = self.prototypes['monster_buffs'].shape[-1]
        # Unbounded upgrades are checked, never clamped to this printed table.
        stats = _nosl_card_stats(128)
        arrays['damage'] = torch.as_tensor(stats['damage'], device=device)
        arrays['block'] = torch.as_tensor(stats['block'], device=device)
        constants['UPGRADE_LEVELS'] = stats['damage'].shape[1]
        for key in ('HP', 'BLOCK', 'ENERGY', 'TURN', 'BUFF', 'COUNTER', 'DMG',
                    'HITS', 'UPGRADE', 'PICK', 'GOLD', 'ASCENSION'):
            constants[key + '_SCALE'] = float(getattr(encode, key + '_SCALE'))
        constants['EXPLODER_MOVE'] = encode.V.MOVE_VOCAB_INDEX['EXPLODER_EXPLODE']
        offset = 0
        for key in n.scalar_keys:
            constants['SCALAR_' + key.upper()] = offset
            offset += int(np.prod(self.prototypes[key].shape))
        constants['SCALAR_EXTRA'] = offset
        constants['SCALAR_WIDTH'] = n.player_projection.in_features
        if offset + 6 != n.player_projection.in_features:
            raise ValueError('current player projection schema is incomplete')
        arrays['player_weight'], arrays['player_bias'] = n.player_projection.weight, n.player_projection.bias
        arrays['turn_bucket'], arrays['reason'], arrays['roles'] = n.turn_bucket.weight, n.choice_reason.weight, n.roles.weight
        card = n.inference_card_table()
        arrays['card_token'] = F.linear(card, n.card_projection.weight[:, :-1], n.card_projection.bias)
        arrays['card_magnitude'] = n.card_projection.weight[:, -1].contiguous()
        for key, layer in (('hand', n.hand_context), ('choice', n.choice_context)):
            arrays[key + '_weight'], arrays[key + '_bias'] = layer.weight, layer.bias
        arrays['zone'], arrays['potion_token'] = n.choice_zone.weight, n.potion.weight
        mw = n.monster_projection.weight
        arrays['monster_id_token'] = F.linear(n.monster_id.weight, mw[:, :32])
        arrays['monster_move_token'] = F.linear(n.monster_move.weight, mw[:, 32:64])
        arrays['monster_intent_token'] = F.linear(n.monster_intent.weight, mw[:, 64:80])
        arrays['monster_weight'] = mw[:, 80:].contiguous()
        arrays['monster_bias'] = n.monster_projection.bias
        summary = n.summaries
        ids = torch.arange(summary.card.num_embeddings, device=device)
        base = summary.card_terms(ids, torch.zeros_like(ids))
        delta = (summary.card_projection(summary.card.versioned_forward(ids, torch.ones_like(ids))) - base) * ids.ne(0)[:, None]
        pw = n.pile_projection.weight[:, :64]
        arrays['pile_base'] = F.linear(base, pw)
        arrays['pile_upgrade'] = F.linear(delta, pw)
        arrays['pile_magnitude'] = F.linear(summary.upgrade_projection, pw)
        arrays['pile_count'] = n.pile_projection.weight[:, 64].contiguous()
        arrays['pile_bias'] = n.pile_projection.bias
        arrays['relic_identity'], arrays['relic_counter'] = summary.relic_identity.weight, summary.relic_counter.weight
        for key, value in arrays.items():
            value = value.detach().contiguous()
            if key not in self.tables:
                self.tables[key] = value.clone()
            else:
                if self.tables[key].shape != value.shape:
                    raise RuntimeError('input table geometry changed across frozen collections')
                self.tables[key].copy_(value)
        self.defines = constants
        pointers = torch.tensor([x.data_ptr() for x in self.tables.values()], device=device, dtype=torch.int64)
        if hasattr(self, 'bindings'):
            self.bindings.copy_(pointers)
        else:
            self.bindings = pointers
        self.version = tuple((id(p), p._version) for p in n.parameters())

    def validate(self):
        if self.version != tuple((id(p), p._version) for p in self.net.parameters()):
            raise RuntimeError('full480 input weights changed before collection drain and refresh')

    def header(self):
        lines = ['#pragma once', 'namespace deck_input {']
        for i, (name, value) in enumerate(self.tables.items()):
            kind = 'int' if value.dtype == torch.int32 else 'float'
            if value.dtype not in (torch.int32, torch.float32):
                raise TypeError((name, value.dtype))
            lines.append(f'inline const {kind}* {name}(const unsigned long long*p){{return reinterpret_cast<const {kind}*>(p[{i}]);}}')
        for key, value in self.defines.items():
            kind = 'float' if isinstance(value, float) else 'int'
            literal = repr(value) + 'f' if isinstance(value, float) else str(value)
            lines.append(f'constexpr {kind} {key}={literal};')
        lines.append('}')
        return '\n'.join(lines) + '\n'


def portable_compact_header(source):
    """Compile the same canonical writer using the original device std subset.

    Only host includes, diagnostics and the immutable intent-name table change;
    the numeric fields and selection logic remain the canonical source text.
    """
    source = Path(source).read_text(encoding='utf-8')
    source = re.sub(r'^#include <[^>]+>\s*$', '', source, flags=re.M)
    begin = source.index('inline int intentCategoryFromName(')
    end = source.index('inline int monsterStatusValue(', begin)
    categories = list(map(int, _nosl_compact_lut()['layout']['move_intent_categories']))
    table = ','.join(map(str, categories))
    second = (f'constexpr int intentCategories[]={{'+table+'};\n'
        'static_assert(sizeof(intentCategories)/sizeof(int)==kNativeMoves,"intent vocabulary mismatch");\n'
        'inline int intentCategory(const sts::Monster &monster,int baseDamage){\n'
        ' if(baseDamage>0)return 1;const int id=static_cast<int>(monster.moveHistory[0]);\n'
        ' return id<=0||id>=kNativeMoves?0:intentCategories[id];\n}\n')
    source = source[:begin] + second + source[end:]
    source = source[:source.index('template <typename Pile>')] + '} // namespace compact_leaf\n'
    source = re.sub(r'throw std::\w+\([^;]*\);', 'gpu_fail(__LINE__);', source)
    return source
