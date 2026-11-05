"""Public tokenization shared by diagnostic lower policies."""

import numpy as np


def _hash32(x):
    x = x.astype(np.uint32, copy=False)
    x = (x ^ (x >> 16)) * np.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * np.uint32(0x846CA68B)
    return x ^ (x >> 16)


TOKEN_COUNT, TOKEN_FEATURES, WIDTH, HEADS = 33, 128, 32, 4


def public_tokens(table, lengths):
    """One token per hand card, potion slot and monster; other piles are multisets.

    Position in a known draw pile remains an entity property. Hidden order and
    native execution indices never become inputs. All rows enter a token.
    """
    ids, zones, mon = (
        np.asarray(table[k], np.int32) for k in ("ids", "zones", "monsters")
    )
    x = np.asarray(table["numbers"], np.float32)
    lengths = np.asarray(lengths, np.int64)
    if (
        lengths.ndim != 1
        or np.any(lengths <= 0)
        or lengths.sum() != len(ids)
        or x.shape != (len(ids), 16)
    ):
        raise ValueError("invalid public token input shape")
    if (
        not np.isfinite(x).all()
        or np.any((zones < 0) | (zones >= 15))
        or np.any((ids < 0) | (ids >= 6144))
        or np.any((mon < -1) | (mon > 4))
    ):
        raise ValueError("invalid public token entity")
    # battle/player, hand, unknown/known draw, discard, exhaust, stasis,
    # generated, pending, deck, owned/active relics, potions, enemies.
    slot = np.array([1, 2, 18, 6, 7, 8, 9, 10, 12, 11, 5, 3, 4, 13, 28], np.int32)[
        zones
    ]
    slot[(zones == 1) & (ids // 512 == 1)] = 1
    hand = zones == 2
    hand_i = np.rint(x[hand, 8] * 40).astype(np.int32) - 1
    pot = zones == 13
    pot_i = np.rint(x[pot, 0] * 5).astype(np.int32) - 1
    if (
        np.any((hand_i < 0) | (hand_i >= 10))
        or np.any((pot_i < 0) | (pot_i >= 5))
        or np.any(mon[zones == 14] < 0)
    ):
        raise ValueError("invalid real combat token slot")
    slot[hand] += hand_i
    slot[pot] += pot_i
    slot[zones == 14] += mon[zones == 14]
    owners = np.repeat(np.arange(len(lengths)), lengths) * TOKEN_COUNT + slot
    result = np.zeros((len(lengths) * TOKEN_COUNT, TOKEN_FEATURES), np.float32)

    # Last channel is reserved for the exact entity count through log1p.
    def add(owner, name, val):
        h = _hash32(name)
        np.add.at(
            result, (owner, h % (TOKEN_FEATURES - 1)), np.where(h >> 31, -val, val)
        )

    identity = ids * 15 + zones
    kind = (ids // 512 + 6144) * 15 + zones
    add(owners, identity * 17, 1.0)
    rr, cc = np.nonzero(x)
    v = x[rr, cc]
    add(owners[rr], identity[rr] * 17 + cc + 1, v)
    add(owners[rr], kind[rr] * 17 + cc + 1, v)
    counts = np.bincount(owners, minlength=result.shape[0])
    result /= np.sqrt(np.maximum(1, counts).astype(np.float32))[:, None]
    result[:, -1] = np.log1p(counts)
    return result.reshape(len(lengths), TOKEN_COUNT, TOKEN_FEATURES)
