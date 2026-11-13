"""CPU root export with exact replay of device-incompatible callback queues.

Only portable battle states cross the device boundary. A short committed action
suffix recreates pending callbacks on CUDA. The live game is never replayed or
mutated; a restored private clone bootstraps an initially pending root.
"""
from .cuda_battle import load_artifacts
from .resident_chains import root_payload
import numpy as np


def pack_actor_roots(rows, host):
    """Adapt full PPO observations to the old FIFO's contiguous native fields."""
    keys = list(rows)
    observations = [rows[k]['observation'] for k in keys]
    groups = {}
    for name, value in observations[0].items():
        groups.setdefault(value.dtype.str, []).append(name)
    columns = [(names, [observations[0][k].shape for k in names],
                host.stack_fields(observations, names)) for names in groups.values()]
    metadata = [{k: v for k, v in rows[key].items() if k != 'observation'} for key in keys]
    return dict(contract='legacy-deck-roots-v1', keys=keys, columns=columns, metadata=metadata,
                field_order=list(observations[0]))


def unpack_actor_roots(packet, *, row_observations=True):
    if packet.get('contract') != 'legacy-deck-roots-v1':
        observations = {k: np.stack([row['observation'][k] for row in packet.values()])
                        for k in next(iter(packet.values()))['observation']}
        return packet, observations
    observations = [{} for _ in packet['keys']]
    batched = {}
    for names, shapes, storage in packet['columns']:
        offset = 0
        for name, shape in zip(names, shapes, strict=True):
            end = offset + int(np.prod(shape))
            field = storage[:, offset:end].reshape(len(observations), *shape)
            batched[name] = field
            if row_observations:
                for row, observation in enumerate(observations):
                    observation[name] = field[row]
            offset = end
        if offset != storage.shape[1]:
            raise ValueError('legacy deck root field layout changed')
    return ({key: (dict(metadata, observation=observation) if row_observations else dict(metadata))
            for key, metadata, observation
            in zip(packet['keys'], packet['metadata'], observations, strict=True)},
            {key: batched[key] for key in packet['field_order']})


def pack_roots(rows, host):
    """Columnar FIFO payload; retain exact fields, row order and root versions.

    Pickling a dictionary of dozens of small arrays per root dominates the
    collector thread. Pack each observation dtype once, like the native root
    bank, and transfer ragged simulator arrays without padding or truncation.
    """
    keys = list(rows)
    values = list(rows.values())
    observations = [r['observation'] for r in values]
    groups = {}
    for key, value in observations[0].items():
        groups.setdefault(value.dtype.str, []).append(key)
    columns = []
    for names in groups.values():
        shapes = [observations[0][k].shape for k in names]
        columns.append((names, shapes, host.stack_fields(observations, names)))
    roots = [r['root'] for r in values]
    metadata = [{k: v for k, v in r.items() if k not in ('snapshot', 'actions', 'bits', 'replay')}
                for r in roots]
    snapshots = [None if r['snapshot'] is None else
                 {k: v for k, v in r['snapshot'].items() if k != 'words'} for r in roots]
    ragged = []
    for arrays in ([np.empty(0, np.uint64) if r['snapshot'] is None else r['snapshot']['words'] for r in roots],
                   [r['actions'] for r in roots], [r['bits'] for r in roots],
                   [np.asarray(r.get('replay', ()), np.int64) for r in roots],
                   [r['worlds'] for r in values]):
        sizes = np.asarray([len(a) for a in arrays], np.int64)
        ragged.append((sizes, np.concatenate(arrays)))
    return dict(version='resident-columns-v1', keys=keys, columns=columns,
                metadata=metadata, snapshots=snapshots, ragged=ragged,
                context=tuple(np.stack([r['context'][i] for r in values]) for i in (0, 1)),
                budgets=[r['budget'] for r in values],
                submitted=np.asarray([r['submitted_at'] for r in values], np.float64))


def unpack_roots(packet):
    """Materialize owned metadata and readonly views of one received batch."""
    if packet['version'] != 'resident-columns-v1':
        raise ValueError('unsupported resident root transport')
    count = len(packet['keys'])
    observations = [{} for _ in range(count)]
    for names, shapes, storage in packet['columns']:
        offset = 0
        for name, shape in zip(names, shapes, strict=True):
            end = offset + int(np.prod(shape))
            field = storage[:, offset:end].reshape(count, *shape)
            for i in range(count):
                observations[i][name] = field[i]
            offset = end
        if offset != storage.shape[1]:
            raise ValueError('resident observation column width changed')
    ragged = []
    for sizes, storage in packet['ragged']:
        offsets = np.r_[0, sizes.cumsum()]
        if len(sizes) != count or offsets[-1] != len(storage):
            raise ValueError('resident ragged field length changed')
        ragged.append([storage[offsets[i]:offsets[i+1]] for i in range(count)])
    words, actions, bits, replay, worlds = ragged
    return {key: dict(root=dict(packet['metadata'][i],
                               snapshot=(None if packet['snapshots'][i] is None else
                                         dict(packet['snapshots'][i], words=words[i])),
                               actions=actions[i], bits=bits[i], replay=replay[i]),
                      observation=observations[i],
                      context=tuple(x[i] for x in packet['context']),
                      worlds=worlds[i], budget=packet['budgets'][i],
                      submitted_at=float(packet['submitted'][i]))
            for i, key in enumerate(packet['keys'])}


class ResidentRootExporter:
    def __init__(self, envs, artifacts, *, observations=None):
        self.envs = [e.unwrapped for e in envs]
        _, _, self.bridge = load_artifacts(artifacts)
        self.cached = [None]*len(envs)
        self.observations = observations
        self.mirrors = [None]*len(envs)
        self.serial = 0

    def _remember(self, env, previous, action):
        if env._phase() != 'combat' or env._at_combat_exit_anchor():
            return None
        state = self.bridge.snapshot(env.thick.bc)
        if state['portable']:
            previous = dict(snapshot=state, prefix=sum(1 << k for k in (env._multi or ())), replay=[])
        if previous is not None:
            previous['replay'].append(int(action)-env._policy_offset)
        return previous

    def before_step(self, actions, indices=None):
        indices = range(len(self.envs)) if indices is None else indices
        for i, action in zip(indices, actions, strict=True):
            env = self.envs[i]
            mirror = self.mirrors[i]
            if mirror is not None and env.thick is not None and mirror['battle'] is env.thick.bc:
                if env._phase() == 'combat' and not env._at_combat_exit_anchor():
                    mirror['replay'].append(int(action)-env._policy_offset)
                    continue
            self.mirrors[i] = None
            self.cached[i] = self._remember(env, self.cached[i], action)

    def export(self, indices, generation=None):
        if generation is not None:
            return self._export_mirrors(indices, generation)
        return self._export_snapshots(indices)

    def _export_mirrors(self, indices, generation):
        result = []
        for i in indices:
            env, mirror = self.envs[i], self.mirrors[i]
            if mirror is None or mirror['battle'] is not env.thick.bc:
                root = self._export_snapshots([i])[0]
                if root.get('native_fallback'):
                    result.append(root)
                    continue
                self.serial += 1
                mirror = dict(snapshot=root['snapshot'], prefix=root['prefix'],
                              replay=list(root['replay']), serial=self.serial,
                              battle=env.thick.bc, emitted=None)
                self.mirrors[i] = mirror
                # Legacy direct callers can still replay this exact origin.
                self.cached[i] = {k: mirror[k] for k in ('snapshot', 'prefix', 'replay')}
            else:
                observation = (None if self.observations is None else
                               {'select_zone': self.observations['select_zone'][i]})
                root = root_payload(env, self.bridge, observation=observation, snapshot=mirror['snapshot'])
            root.update(prefix=mirror['prefix'], replay=list(mirror['replay']),
                        mirror=dict(env=i, serial=mirror['serial'], version=len(mirror['replay'])))
            root['snapshot'] = mirror['snapshot'] if mirror['emitted'] != generation else None
            mirror['emitted'] = generation
            result.append(root)
        return result

    def _export_snapshots(self, indices):
        result = []
        for i in indices:
            env = self.envs[i]
            observation = (None if self.observations is None else
                           {'select_zone': self.observations['select_zone'][i]})
            root = root_payload(env, self.bridge, observation=observation)
            root['replay'] = []
            if not root['snapshot']['portable']:
                if self.cached[i] is None:
                    shadow = env.fork_run()
                    current = None
                    def remember(branch, record):
                        nonlocal current
                        action = (record['action'] if record['kind'] == 'thin' else
                                  {int(b): a for a, b in branch._joint_policy_actions()}[record['action_bits']])
                        current = self._remember(branch, current, action)
                    try:
                        shadow.restore_training_state(env.export_training_state(), replay_observer=remember)
                    finally:
                        shadow.close()
                    self.cached[i] = current
                if self.cached[i] is None:
                    # A battle can enter its very first choice with callbacks
                    # already queued (there is no earlier combat action to
                    # replay). Keep the exact native state in its owner worker;
                    # the learner routes this worker's paired group through the
                    # existing native-branch/GPU-inference service.
                    root['native_fallback'] = 'pending-without-portable-origin'
                else:
                    root.update(self.cached[i])
            # The following real step may extend our cached suffix after send.
            result.append(dict(root, replay=list(root['replay'])))
        return result
