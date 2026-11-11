"""Batch-local context indices and homogeneous routing for the COMBAT49 FP32 learner."""
from dataclasses import dataclass

import numpy as np
import torch

from env.core import native_actions as C, vocab
from rl.policies.packed_update_graphs import PackedHeadGraphs


def _last_slot(mask):
    positions = torch.arange(1, mask.shape[1] + 1, device=mask.device)
    return (mask.long() * positions).amax(1).clamp_min(1)


@dataclass
class _Chunk:
    first: torch.Tensor
    inverse: torch.Tensor
    deck: int
    relic: int
    length: int
    phase: int
    routes: list
    order: torch.Tensor | None
    upper_kept: int


class DeckUpdatePlan:
    def __init__(self, model, reducer):
        self.core = model.policy.core
        self.reducer = reducer
        self.graphs = PackedHeadGraphs(self.core)
        self.chunks = {}
        self.chunk_count = 0
        self.eager_retries = 0

    def schedule(self, chunks):
        groups = [batch.observations for batch, _, _, _ in chunks]

        def join(key):
            return torch.cat([obs[key] for obs in groups], dim=0)

        deck, relic = join('deck_ids').ne(0), join('relics').ne(0)
        rooms, features = join('map_full_rooms').ne(0), join('map_full_features')
        tokens = deck.sum(1) + relic.sum(1)
        lengths = 1 + tokens + (rooms | features.bool().any(-1)).sum(1)
        phase = join('phase').flatten()
        tokens += torch.where(phase == vocab.PHASE_VOCAB_INDEX['card_select'],
                              join('selected_card_ids').ne(0).sum(1), 0)
        map_tokens = (rooms | (features[..., 3] > .5)).sum(1)
        if 'upper_public_context' in self.core.overworld.encoder.observation_space.spaces:
            tokens += map_tokens
        else:
            tokens += torch.where(phase == vocab.PHASE_VOCAB_INDEX['map'], map_tokens, 0)
        valid = join('select_native_indices') > 0
        pick = C.JOINT_PICK - C.JOINT_END
        required = valid | join('combat_legal')[:, pick:pick + valid.shape[1]].bool()

        # A single transfer supplies metadata for every physical chunk in this original Adam group.
        table = torch.stack((join('_deck_context_index').flatten(), phase, join('combat_exit_anchor').bool().any(1),
                             _last_slot(deck), _last_slot(relic), lengths, _last_slot(required), tokens), dim=1)
        host = table.long().cpu().numpy()
        offset = 0
        for batch, _, _, name in chunks:
            n = len(batch.advantages)
            ids, phases, anchored, dw, rw, length, choice, kept = host[offset:offset + n].T
            offset += n
            screen = vocab.PHASE_VOCAB_INDEX[name]
            if not np.all(phases == screen):
                raise ValueError(f'packed {name} chunk contains other phases')
            _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)

            def indices(rows):
                return torch.as_tensor(rows, device=deck.device)

            routes = []
            if name == 'combat':
                for anchor in (False, True):
                    rows = np.flatnonzero(anchored == anchor)
                    if len(rows):
                        routes.append((anchor, None if len(rows) == n else indices(rows), int(choice[rows].max())))
            order = indices(np.argsort(np.argsort(anchored, kind='stable'))) if len(routes) == 2 else None
            self.chunks[(batch.observations['phase'].data_ptr(), n)] = _Chunk(
                indices(first), indices(inverse), int(dw.max()), int(rw.max()), int(length.max()),
                screen, routes, order, int(kept.max()))
        self.chunk_count += len(chunks)
        try:
            yield from self.reducer.schedule(chunks)
        finally:
            self.chunks.clear()

    def cap_key(self, group, obs):
        chunk = self.chunks.get((obs['phase'].data_ptr(), len(obs['phase'])))
        return group if chunk is None else f'{group}/L{(chunk.length + 7) // 8}'

    def actor_critic(self, obs):
        chunk = self.chunks.get((obs['phase'].data_ptr(), len(obs['phase'])))
        if chunk is None:
            # OOM subdivisions have their own rows, widths, and anchor routes.
            self.eager_retries += 1
            return self.core.eager_actor_critic(obs)
        context_obs = {k: obs[k][chunk.first] for k in self.core.context.keys}
        z = self.core.context(context_obs, (chunk.deck, chunk.relic), packed_width=chunk.length)[chunk.inverse]
        if not chunk.routes:
            return self.graphs.upper(obs, z, chunk.phase, chunk.deck, chunk.relic, chunk.upper_kept)
        outputs, values = [], []
        for anchor, rows, width in chunk.routes:
            part = obs if rows is None else {k: t[rows] for k, t in obs.items()}
            ctx = z if rows is None else z[rows]
            if anchor:
                logits = z.new_zeros(len(ctx), C.AUTOREGRESSIVE_SIZE)
                value = self.core.anchor_with_context(part, self.core.anchor.context_projection(ctx))
            else:
                logits, value = self.graphs.combat(part, ctx, width)
            if rows is None:
                return logits.float(), value.float()
            outputs.append(logits.float())
            values.append(value.float())
        return torch.cat(outputs)[chunk.order], torch.cat(values)[chunk.order]
