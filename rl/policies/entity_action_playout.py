"""Token-wise action scores, with an explicit source–target interaction."""

import numpy as np
import torch
from torch import nn

from env.search.token_context import (
    TOKEN_COUNT,
    TOKEN_FEATURES,
    WIDTH,
    HEADS,
    public_tokens,
)
from rl.policies.small_playout import SmallPlayoutPolicy


def action_tokens(table):
    """Encode a batch of roots from a game shard; handles are only gather indices."""
    lens = np.asarray(table["lengths"], np.int64)
    alens = np.asarray(table["alens"], np.int64)
    features = public_tokens(table, lens)
    start = lens.cumsum() - lens
    owner = np.repeat(np.arange(len(lens)), lens)
    aowner = np.repeat(np.arange(len(lens)), alens)
    source = np.asarray(table["sources"], np.int64)
    targets = np.asarray(table["targets"], np.int64)
    if (
        source.shape != (len(aowner), 10)
        or targets.shape != (len(aowner),)
        or (source < -1).any()
        or (source >= lens[aowner, None]).any()
        or ((targets < -1) | (targets >= 5)).any()
    ):
        raise ValueError("invalid public source/target gather indices")
    actual = np.where(source >= 0, source + start[aowner, None], -1)
    ids = np.asarray(table["ids"])
    z = np.asarray(table["zones"])
    x = np.asarray(table["numbers"])
    slot = np.full(len(ids), -1, np.int64)
    hand = z == 2
    pot = z == 13
    enemy = (z == 14) & (ids // 512 == 5)
    slot[hand] = 18 + np.rint(x[hand, 8] * 40).astype(int) - 1
    slot[pot] = 13 + np.rint(x[pot, 0] * 5).astype(int) - 1
    slot[enemy] = 28 + np.asarray(table["monsters"])[enemy]
    symbols = np.zeros((len(lens), TOKEN_COUNT), np.int32)
    primary = hand | pot | enemy
    symbols[owner[primary], slot[primary]] = ids[primary] + 1
    # A selected discard/draw/exhaust/generated instance also needs its own
    # output token. Never turn such a handle into a hand slot or truncate it.
    used = np.unique(actual[actual >= 0])
    extra = used[slot[used] < 0]
    counts = np.bincount(owner[extra], minlength=len(lens))
    estart = counts.cumsum() - counts
    slot[extra] = TOKEN_COUNT + np.arange(len(extra)) - estart[owner[extra]]
    source_slots = np.full_like(source, -1)
    valid = actual >= 0
    source_slots[valid] = slot[actual[valid]]
    if len(extra):
        # Each extra row is a single-card token. Keep its true kind/zone and
        # numeric properties; a shared slot embedding avoids hidden physical indices.
        from env.search.token_context import _hash32

        er = np.zeros((len(extra), TOKEN_FEATURES), np.float32)
        ei = ids[extra] * 15 + z[extra]
        ki = (ids[extra] // 512 + 6144) * 15 + z[extra]

        def add(ix, name, val):
            h = _hash32(name)
            np.add.at(er, (ix, h % (TOKEN_FEATURES - 1)), np.where(h >> 31, -val, val))

        add(np.arange(len(extra)), ei * 17, 1.0)
        rr, cc = np.nonzero(x[extra])
        v = x[extra][rr, cc]
        add(rr, ei[rr] * 17 + cc + 1, v)
        add(rr, ki[rr] * 17 + cc + 1, v)
        er[:, -1] = np.log(2.0)
    else:
        er = np.empty((0, TOKEN_FEATURES), np.float32)
    return (
        features,
        symbols,
        er,
        (ids[extra] + 1).astype(np.int32),
        counts,
        source_slots.astype(np.int32),
        np.where(targets >= 0, 28 + targets, -1).astype(np.int32),
    )


class EntityBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.mix = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, 2 * width)
        self.ff2 = nn.Linear(2 * width, width)

    def forward(self, x, present):
        b, t, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        h = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=present[:, None, None, :]
        )
        x = x + self.mix(h.transpose(1, 2).reshape(b, t, d))
        return x + self.ff2(torch.relu(self.ff1(self.norm2(x))))


class ActionTokenPolicy(SmallPlayoutPolicy):
    def __init__(self, key_count, coupled=True, width=32, depth=1):
        super().__init__(key_count)
        if width not in (32, 64, 128) or depth not in (1, 2, 3):
            raise ValueError("unsupported token action architecture")
        self.width, self.depth, self.coupled = width, depth, coupled
        self.input = nn.Linear(TOKEN_FEATURES, width)
        self.slots = nn.Embedding(TOKEN_COUNT + 1, width)
        # Fixed lookup width across arms; the capacity comparison expands the
        # attention/FF layers, not just the number of unused embedding columns.
        self.symbol = nn.Embedding(6145, 32, padding_idx=0)
        self.symbol_lift = (
            nn.Identity() if width == 32 else nn.Linear(32, width, bias=False)
        )
        self.blocks = nn.ModuleList([EntityBlock(width, 4) for _ in range(depth)])
        self.final = nn.LayerNorm(width)
        nn.init.normal_(self.slots.weight, std=0.02)
        nn.init.normal_(self.symbol.weight, std=0.1)
        with torch.no_grad():
            self.symbol.weight[0].zero_()
        self.source_score = nn.Linear(width, 1, bias=False)
        self.target_score = nn.Linear(width, 1, bias=False)
        self.left = nn.Linear(width, 16, bias=False)
        self.right = nn.Linear(width, 16, bias=False)
        for m in (self.source_score, self.target_score, self.left):
            nn.init.zeros_(m.weight)

    @classmethod
    def from_mature(cls, parent, coupled=True, width=32, depth=1):
        net = cls(parent.key_count, coupled, width, depth).to(parent.fc1.weight.device)
        miss = net.load_state_dict(parent.state_dict(), strict=False)
        if miss.unexpected_keys or not set(miss.missing_keys).isdisjoint(
            parent.state_dict()
        ):
            raise ValueError("token action fork needs the mature compact parent")
        return net

    def representations(self, features, symbols):
        t = features.shape[1]
        present = features[:, :, -1] > 0
        present = torch.cat((torch.ones_like(present[:, :1]), present[:, 1:]), 1)
        slot = torch.arange(t, device=features.device).clamp_max(TOKEN_COUNT)
        x = (
            self.input(features)
            + self.slots(slot)
            + self.symbol_lift(self.symbol(symbols))
        )
        for block in self.blocks:
            x = block(x, present)
        return self.final(x)

    def forward_packed(
        self, state, actions, keys, slots, features, symbols, sources, targets, mask
    ):
        old = super().forward_packed(state, actions, keys, slots, mask)
        x = self.representations(features, symbols)
        owners = slots // mask.shape[1]
        sv = sources >= 0
        n = sv.sum(-1, keepdim=True)
        source = (x[owners[:, None], sources.clamp_min(0)] * sv[:, :, None]).sum(
            1
        ) / n.clamp_min(1)
        # End Turn, optional skip and selecting zero cards keep different legacy
        # features. Their learned state-dependent source is the CLS output.
        source = torch.where(n == 0, x[owners, 0], source)
        target = x[owners, targets.clamp_min(0)] * (targets >= 0)[:, None]
        bonus = (
            self.source_score(source).flatten() + self.target_score(target).flatten()
        )
        if self.coupled:
            bonus = bonus + (self.left(source) * self.right(target)).sum(-1) * 0.25
        return old + old.new_zeros(mask.numel()).index_copy(0, slots, bonus).view_as(
            mask
        )

    def forward(self, *args):
        raise ValueError(
            "token action scores require actual source and target references"
        )

    def export(self):
        raise ValueError(
            "use an explicitly versioned token action scorer; no v1 export"
        )


class ActionTokenBank:
    def __init__(self, shards, legacy):
        self.base = legacy
        data = [action_tokens(shard) for shard in shards]
        (
            self.features,
            self.symbols,
            self.extras,
            self.extra_symbols,
            self.counts,
            self.sources,
            self.targets,
        ) = [np.concatenate([row[i] for row in data]) for i in range(7)]
        self.starts = self.counts.cumsum() - self.counts
        if len(self.counts) != len(legacy.states) or len(self.targets) != len(
            legacy.actions
        ):
            raise ValueError("mismatched token and search banks")

    def cache_features(self, device):
        import torch

        for name in ("features", "symbols", "extras", "extra_symbols"):
            setattr(self, name, torch.as_tensor(getattr(self, name), device=device))

    def batch(self, indices, device):
        inputs, label, teacher = self.base.batch(indices, device)
        lengths = self.base.lengths[indices]

        def packed(starts, lens):
            owner = np.repeat(np.arange(len(lens)), lens)
            within = np.arange(lens.sum()) - np.repeat(lens.cumsum() - lens, lens)
            return starts[owner] + within, owner, within

        rows, _, _ = packed(self.base.starts[indices], lengths)
        counts = self.counts[indices]
        width = TOKEN_COUNT + int(counts.max())
        er, ow, pos = packed(self.starts[indices], counts)
        if torch.is_tensor(self.features):
            if width == TOKEN_COUNT:
                f = self.features[indices]
                sym = self.symbols[indices].long()
            else:
                f = self.features.new_zeros((len(indices), width, TOKEN_FEATURES))
                sym = torch.zeros(
                    (len(indices), width), dtype=torch.long, device=device
                )
                f[:, :TOKEN_COUNT] = self.features[indices]
                sym[:, :TOKEN_COUNT] = self.symbols[indices].long()
                f[ow, TOKEN_COUNT + pos] = self.extras[er]
                sym[ow, TOKEN_COUNT + pos] = self.extra_symbols[er].long()
        else:
            f = np.zeros((len(indices), width, TOKEN_FEATURES), np.float32)
            sym = np.zeros((len(indices), width), np.int64)
            f[:, :TOKEN_COUNT] = self.features[indices]
            sym[:, :TOKEN_COUNT] = self.symbols[indices]
            f[ow, TOKEN_COUNT + pos] = self.extras[er]
            sym[ow, TOKEN_COUNT + pos] = self.extra_symbols[er]
        extra = tuple(
            torch.as_tensor(t, device=device)
            for t in (
                f,
                sym,
                self.sources[rows].astype(np.int64),
                self.targets[rows].astype(np.int64),
            )
        )
        return (*inputs[:-1], *extra, inputs[-1]), label, teacher
