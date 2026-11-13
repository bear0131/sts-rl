"""Legacy real-battle mirrors with versioned PPO-action replay.

Only the real CPU environment chooses actions. Each source stays in its own
device slot; search forks copy it before a later real version can advance it.
Callback queues and RNG remain owned by the original simulator module.
"""
import numpy as np
import torch


class ResidentBattleMirror:
    def __init__(self, arena, first, capacity):
        self.arena, self.first, self.capacity = arena, first, capacity
        self.versions = {}
        self.counts = dict(loads=0, actual_actions=0, roots=0)
        self.storage = torch.empty((capacity, arena.battle_bytes), dtype=torch.uint8, device=arena.device)
        if capacity:
            arena.arena[first:first+capacity].copy_(self.storage.data_ptr()+
                torch.arange(capacity, device=arena.device, dtype=torch.int64)*arena.battle_bytes)

    @staticmethod
    def groups(payloads):
        """No batch advances the same real environment past an un-cloned root."""
        group, seen = [], set()
        for i, root in enumerate(payloads):
            identity = root.get('mirror')
            env_id = None if identity is None else identity['env']
            if env_id is not None and env_id in seen:
                yield group
                group, seen = [], set()
            group.append(i)
            if env_id is not None:
                seen.add(env_id)
        if group:
            yield group

    def prepare(self, payloads, slots):
        sources, loads, prefixes, load_slots, suffixes = [], [], [], [], []
        for root, slot in zip(payloads, slots, strict=True):
            identity = root.get('mirror')
            start = 0
            if identity is not None:
                env_id, serial, version = (identity[k] for k in ('env', 'serial', 'version'))
                if not 0 <= env_id < self.capacity or version != len(root['replay']):
                    raise ValueError('invalid real battle mirror version')
                slot = self.first+env_id
                previous = self.versions.get(env_id)
                if previous is not None and previous[0] == serial:
                    start = previous[1]
                    if version < start:
                        raise ValueError('real battle mirror moved backwards')
                else:
                    if previous is not None and serial <= previous[0]:
                        raise ValueError('stale real battle identity')
                    if root['snapshot'] is None:
                        raise ValueError('real battle mirror origin was not delivered')
                    loads.append(root['snapshot']); prefixes.append(root['prefix']); load_slots.append(slot)
                    self.counts['loads'] += 1
                self.versions[env_id] = serial, version
                self.counts['roots'] += 1
            else:
                loads.append(root['snapshot']); prefixes.append(root['prefix']); load_slots.append(slot)
            sources.append(slot)
            suffixes.append(root.get('replay', ())[start:])
        if loads:
            self.arena.load_slots(loads, prefixes, load_slots)
        for step in range(max(map(len, suffixes), default=0)):
            selected = [i for i, moves in enumerate(suffixes) if step < len(moves)]
            self.arena.replay_slots([sources[i] for i in selected], [suffixes[i][step] for i in selected])
            self.counts['actual_actions'] += len(selected)
        return np.asarray(sources, np.int64)
