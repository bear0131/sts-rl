"""Native step batching within workers, shared observations across processes.

Control, masks, terminal infos and recovery use the existing grouped protocol.
Only current observations use shared storage; public returns always own copies.
"""
from multiprocessing import shared_memory
from contextlib import contextmanager
import os
import selectors
import time

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

from rl.training import cached_mask_vec_env as cv
from rl.training.native_vec_env import NativeSearchVecEnv


class _NativeStepPipe:

    def __init__(self, remote, envs):
        self.remote = remote
        self.envs = envs
        self.native = None
        self.memory = None
        self.closed = False
        self.deck_search = None
        self.deck_frontier = None
        self.deck_resident = None
        self.anchor = None
        self.collection_context = None

    def send(self, x):
        self.remote.send(x)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.anchor is not None:
                self.anchor.close()
            if self.collection_context is not None:
                self.collection_context.__exit__(None, None, None)
            if self.deck_frontier is not None:
                self.deck_frontier.close()
            if self.native is not None:
                self.native.close()
        finally:
            if self.memory is not None:
                self.memory.close()
            self.remote.close()

    def write(self, obs):
        for key, value in obs.items():
            self.arrays[key][self.start:self.end] = value

    def recv(self):
        while True:
            cmd, data = self.remote.recv()
            if cmd == 'native_shared':
                name, layout, self.start, self.end = data
                self.memory = shared_memory.SharedMemory(name=name)
                self.arrays = {
                    key: np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offset)
                    for key, shape, dtype, offset in layout}
                self.native = NativeSearchVecEnv([lambda env=env: env for env in self.envs], threads=1)
                import slaythespire as sts
                self.observation_copy = sts.ObservationRowCopy(self.arrays, self.native.buf_obs, 1)
                self.remote.send(None)
            elif cmd == 'step' and self.native is not None:
                if self.deck_resident is not None:
                    self.deck_resident.before_step(data)
                obs, rewards, dones, infos = self.native._step_for_transport(np.asarray(data))
                self.write(obs)
                self.remote.send([(None, rewards[i], dones[i], info, self.native.reset_infos[i])
                                  for i, info in enumerate(infos)])
            elif cmd == 'reset' and self.native is not None:
                self.native._seeds = [r[0] for r in data]
                self.native._options = [r[1] for r in data]
                obs = self.native.reset()
                self.write(obs)
                self.remote.send([(None, i) for i in self.native.reset_infos])
            elif cmd == 'native_restore':
                if self.anchor is not None or self.collection_context is not None:
                    raise RuntimeError('restore requires a drained environment worker')
                self.write(self.native.restore_training_states(data))
                self.deck_resident = None
                self.remote.send(self.native.reset_infos)
            elif cmd == 'native_collection_begin':
                if self.collection_context is not None:
                    raise RuntimeError('collection already owns the environment worker')
                self.collection_context = self.native.omit_unused_combat_value_observations()
                self.collection_context.__enter__()
                self.remote.send(None)
            elif cmd == 'native_collection_end':
                if self.collection_context is None or self.anchor is not None:
                    raise RuntimeError('collection exit requires drained native frontiers')
                self.collection_context.__exit__(None, None, None)
                self.collection_context = None
                self.remote.send(None)
            elif cmd == 'native_subset':
                before = dict(self.native.step_seconds)
                if self.deck_resident is not None:
                    self.deck_resident.before_step(data.values(), indices=data.keys())
                obs, rewards, dones, infos = self.native._step_subset(data, borrow=True)
                indices = list(data)
                local = np.asarray(indices, np.intp)
                self.observation_copy.copy(self.start + local, local)
                self.remote.send(dict(rows=[(i, rewards[j], dones[j], infos[j], self.native.reset_infos[i])
                    for j, i in enumerate(indices)],
                    seconds={k: v-before[k] for k, v in self.native.step_seconds.items()}))
            elif cmd == 'native_anchor_start':
                if self.anchor is not None:
                    raise RuntimeError('native frontier already active')
                from env.search.anchor_wave import NativeAnchorFrontier
                self.anchor = NativeAnchorFrontier([self.envs[i].unwrapped for i in data['indices']],
                    data['roots'], data['worlds'], **data['options'])
                self.remote.send(self.anchor.advance())
            elif cmd == 'native_anchor_advance':
                self.remote.send(self.anchor.advance(data))
            elif cmd == 'native_anchor_close':
                if self.anchor is not None:
                    self.anchor.close()
                    self.anchor = None
                self.remote.send(None)
            elif cmd == 'deck_shadow_search':
                if self.deck_search is None:
                    from rl.training.deck_search_worker import DeckSearchWorker
                    self.deck_search = DeckSearchWorker(self.envs)
                self.remote.send(self.deck_search.run(data))
            elif cmd == 'deck_resident_roots':
                if self.deck_resident is None:
                    from env.search.resident_transport import ResidentRootExporter
                    self.deck_resident = ResidentRootExporter(self.envs, data['artifacts'],
                        observations=self.native.buf_obs if self.native is not None else None)
                self.remote.send(self.deck_resident.export(data['indices'], data.get('generation')))
            elif cmd in ('deck_frontier_start', 'deck_frontier_advance', 'deck_frontier_close'):
                if self.deck_frontier is None:
                    from env.search.frontier_transport import FrontierSession
                    self.deck_frontier = FrontierSession(self.envs)
                method = getattr(self.deck_frontier, cmd.removeprefix('deck_frontier_'))
                self.remote.send(method() if cmd == 'deck_frontier_close' else method(data))
            else:
                return (cmd, data)

def _native_group_worker(remote, parent_remote, wrappers, group_index):
    envs = []

    def make(wrapper):

        def create():
            env = wrapper.var()
            envs.append(env)
            return env
        return CloudpickleWrapper(create)
    pipe = _NativeStepPipe(remote, envs)
    try:
        cv._group_worker(pipe, parent_remote, [make(w) for w in wrappers], group_index)
    finally:
        pipe.close()

class NativeSubprocVecEnv(cv.GroupedCachedMaskSubprocVecEnv):
    """Same logical ordering/control RPCs as grouped VecEnv, native step batches."""
    _worker_target = staticmethod(_native_group_worker)

    def __init__(self, *args, native_threads=4, **kwargs):
        self.memory = None
        if native_threads < 1:
            raise ValueError('native search thread count must be positive')
        self.native_threads = native_threads
        # These are sums over parallel worker wall intervals, never additive
        # with the collector's end-to-end environment wall time.
        self.step_seconds = dict(wrappers_sum=0., encoding_sum=0.)
        try:
            super().__init__(*args, **kwargs)
            self._create_shared_observations()
        except BaseException:
            if hasattr(self, 'processes'):
                self.close()
            raise

    def _create_shared_observations(self):
        layout = []
        offset = 0
        for key, space in self.observation_space.spaces.items():
            shape = (self.num_envs, *space.shape)
            offset = (offset + 63) // 64 * 64
            layout.append((key, shape, space.dtype.str, offset))
            offset += int(np.prod(shape)) * space.dtype.itemsize
        self.memory = shared_memory.SharedMemory(create=True, size=offset)
        self.arrays = {
            key: np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offset)
            for key, shape, dtype, offset in layout}
        self.buf_obs = self.arrays
        self.shared_observation_spec = self.memory.name, layout
        for remote, group in zip(self.remotes, self._groups, strict=True):
            remote.send(('native_shared', (self.memory.name, layout, group[0], group[-1] + 1)))
        self._receive_groups({r: i for i, r in enumerate(self.remotes)})

    def restore_training_states(self, states):
        if self.waiting or len(states) != self.num_envs:
            raise ValueError('native restore requires a drained matching vector')
        for remote, group in zip(self.remotes, self._groups, strict=True):
            remote.send(('native_restore', [states[i] for i in group]))
        groups = self._receive_groups({r: i for i, r in enumerate(self.remotes)})
        self.reset_infos = [info for i in range(len(self._groups)) for info in groups[i]]
        self._cached_action_masks = [self._info_mask(info) for info in self.reset_infos]
        return self.observations()

    @contextmanager
    def omit_unused_combat_value_observations(self):
        if self.waiting:
            raise RuntimeError('collection requires a drained vector')
        for remote in self.remotes:
            remote.send(('native_collection_begin', None))
        self._receive_groups({r: i for i, r in enumerate(self.remotes)})
        try:
            yield
        finally:
            for remote in self.remotes:
                remote.send(('native_collection_end', None))
            self._receive_groups({r: i for i, r in enumerate(self.remotes)})

    def _step_subset(self, actions, *, borrow=False):
        if self.waiting or not actions or any(not 0 <= i < self.num_envs for i in actions):
            raise ValueError('subset steps require idle valid logical slots')
        pending = {}
        for group, members in self._grouped(actions).items():
            remote = self.remotes[group]
            remote.send(('native_subset', {local: actions[i] for i, local in members}))
            pending[remote] = group
        result = {}
        for group, packet in self._receive_groups(pending).items():
            for local, reward, done, info, reset in packet['rows']:
                i = self._groups[group][local]
                result[i] = reward, done, info
                self.reset_infos[i] = reset
                self._cached_action_masks[i] = self._info_mask(reset if done else info)
            for key, value in packet['seconds'].items():
                self.step_seconds[key+'_sum'] += value
        ordered = [result[i] for i in actions]
        rewards, dones, infos = zip(*ordered, strict=True)
        rows = np.asarray(list(actions), np.intp)
        observations = self.arrays if borrow else {k: v[rows] for k, v in self.arrays.items()}
        return observations, np.asarray(rewards, np.float32), np.asarray(dones, bool), list(infos)

    def anchor_frontier(self, slots, roots, worlds, options):
        return _GroupedAnchorFrontier(self, slots, roots, worlds, options)

    def observations(self):
        # SB3 retains the previous observation until after step_wait. Returning
        # views would silently overwrite its training sample on the next step.
        return {k: v.copy() for k, v in self.arrays.items()}

    def deck_shadow_search(self, request):
        """Search live roots before the already-sampled real actions advance."""
        if self.waiting:
            raise RuntimeError('cannot fork shadow roots while real transitions are in flight')
        positions = {index: j for j, index in enumerate(request['indices'])}
        pending, selected = {}, {}
        for group_index, group in enumerate(self._groups):
            local = [j for j, i in enumerate(group) if i in positions]
            if not local:
                continue
            rows = [positions[group[j]] for j in local]
            call = dict(request, indices=local, budgets=[request['budgets'][j] for j in rows],
                context=tuple(array[rows] for array in request['context']),
                seed=int(np.random.SeedSequence([request['seed'], group_index]).generate_state(1, dtype=np.uint64)[0]))
            self.remotes[group_index].send(('deck_shadow_search', call))
            pending[self.remotes[group_index]] = group_index
            selected[group_index] = rows
        responses = self._receive_groups(pending)
        results, totals = [None] * len(positions), {}
        for group_index, (group_results, stats) in responses.items():
            for row, result in zip(selected[group_index], group_results, strict=True):
                results[row] = result
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.) + value
        return results, totals

    def deck_resident_roots(self, indices, artifacts, generation=None):
        if self.waiting:
            raise RuntimeError('cannot export resident roots during real transitions')
        positions = {index: j for j, index in enumerate(indices)}
        pending, selected = {}, {}
        for group_index, group in enumerate(self._groups):
            local = [j for j, i in enumerate(group) if i in positions]
            if local:
                selected[group_index] = [positions[group[j]] for j in local]
                self.remotes[group_index].send(('deck_resident_roots',
                    dict(indices=local, artifacts=artifacts, generation=generation)))
                pending[self.remotes[group_index]] = group_index
        roots, groups = [None]*len(indices), [None]*len(indices)
        for group, response in self._receive_groups(pending).items():
            for row, root in zip(selected[group], response, strict=True):
                if 'mirror' in root:
                    root['mirror']['env'] = int(indices[row])
                roots[row], groups[row] = root, group
        return roots, groups

    def deck_batched_search(self, request, scorer):
        """Central inference with native CPU branches; never a CUDA simulator.

        Partition and world seeds match the CPU worker reference. One model
        scores the merged frontier; worker processes never own CUDA contexts.
        All workers finish a complete paired set before real actions advance.
        """
        from env.search.anchored_chains import ChainRequest
        if self.waiting:
            raise RuntimeError('cannot search while real transitions are in flight')
        positions = {index: j for j, index in enumerate(request['indices'])}
        selected, pending, active = {}, {}, set()
        results, caches = [None] * len(positions), {}
        totals = dict(scoring_seconds=0., worker_and_transport_seconds=0., frontier_batches=0.,
                      frontier_rows=0., transport_bytes=0.)
        try:
            started = time.monotonic()
            for group_index, group in enumerate(self._groups):
                local = [j for j, i in enumerate(group) if i in positions]
                if not local:
                    continue
                rows = [positions[group[j]] for j in local]
                call = dict(request, indices=local, budgets=[request['budgets'][j] for j in rows],
                    observation_keys=scorer.observation_keys, static_keys=scorer.context_keys,
                    seed=int(np.random.SeedSequence([request['seed'], group_index]).generate_state(1, dtype=np.uint64)[0]))
                self.remotes[group_index].send(('deck_frontier_start', call))
                pending[self.remotes[group_index]] = group_index
                selected[group_index] = rows
                active.add(group_index)
            while pending:
                responses = self._receive_groups(pending)
                totals['worker_and_transport_seconds'] += time.monotonic() - started
                frontier, batches = [], []
                for group_index, response in responses.items():
                    if response['done']:
                        active.remove(group_index)
                        group_results, stats = response['result']
                        for row, result in zip(selected[group_index], group_results, strict=True):
                            results[row] = result
                        for key, value in stats.items():
                            totals[key] = totals.get(key, 0.) + value
                        continue
                    begin = len(frontier)
                    for root, lane, kind, mask, delta in response['rows']:
                        row = selected[group_index][root]
                        observation = dict(scorer.roots[row], **delta)
                        cache = caches.setdefault((group_index, lane), {})
                        frontier.append(ChainRequest(row, observation, mask, kind, cache, lane))
                    totals['transport_bytes'] += response['observation_bytes']
                    batches.append((group_index, response['sequence'], begin, len(frontier)))
                if not frontier:
                    continue
                before = time.monotonic()
                outputs = scorer(frontier)  # CPU outputs imply completion of CUDA copies.
                totals['scoring_seconds'] += time.monotonic() - before
                totals['frontier_batches'] += 1
                totals['frontier_rows'] += len(frontier)
                if len(outputs) != len(frontier):
                    raise ValueError('incomplete central search scorer response')
                pending, started = {}, time.monotonic()
                for group_index, sequence, begin, end in batches:
                    self.remotes[group_index].send(('deck_frontier_advance', dict(
                        generation=request['generation'], sequence=sequence, outputs=outputs[begin:end])))
                    pending[self.remotes[group_index]] = group_index
            totals['context_refreshes'] = scorer.context_refreshes
            totals.update(scorer.cache_statistics())
            return results, totals
        finally:
            # Scorer failures must not leave a suspended search retaining forks
            # or turn the next real step into a stale frontier response.
            if active and all(self.processes[i].is_alive() for i in active):
                for i in active:
                    self.remotes[i].send(('deck_frontier_close', None))
                self._receive_groups({self.remotes[i]: i for i in active})

    def _receive_groups(self, pending):
        if os.name != 'posix':
            return super()._receive_groups(pending)
        # Register each pipe/sentinel once for this request batch. Rebuilding a
        # selector after each early worker response was quadratic in workers.
        # Windows retains multiprocessing's named-pipe-aware implementation.
        results = {}
        with selectors.PollSelector() as selector:
            for remote in pending:
                selector.register(remote, selectors.EVENT_READ)
            for sentinel in self._worker_sentinels:
                selector.register(sentinel, selectors.EVENT_READ)
            while pending:
                ready = [key.fileobj for key, events in selector.select()]
                for item in ready:
                    if item in self._worker_sentinels:
                        self._raise_worker_failure(self._worker_sentinels[item])
                for remote in ready:
                    group_index = pending.pop(remote)
                    selector.unregister(remote)
                    try:
                        payload, _timing = cv._unwrap_timed_payload(remote.recv())
                        results[group_index] = cv._unpack_leaf_responses(payload)
                    except (EOFError, BrokenPipeError, OSError) as exc:
                        self._raise_worker_failure(group_index, exc)
        return results

    def results(self):
        grouped = self._receive_groups({r: i for i, r in enumerate(self.remotes)})
        return [grouped[i][j] for i, g in enumerate(self._groups) for j, _ in enumerate(g)]

    def step_wait(self):
        try:
            results = self.results()
        finally:
            self.waiting = False
        _, rewards, dones, infos, self.reset_infos = zip(*results, strict=True)
        for i, (done, info) in enumerate(zip(dones, infos, strict=True)):
            self._cached_action_masks[i] = self._info_mask(self.reset_infos[i] if done else info)
        return (self.observations(), np.asarray(rewards), np.asarray(dones), infos)

    def reset(self):
        for remote, group in zip(self.remotes, self._groups, strict=True):
            remote.send(('reset', [(self._seeds[i], self._options[i]) for i in group]))
        _, infos = zip(*self.results(), strict=True)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        self._cached_action_masks = [self._info_mask(i) for i in infos]
        return self.observations()

    def close(self):
        try:
            super().close()
        finally:
            if self.memory is not None:
                self.memory.close()
                self.memory.unlink()
                self.memory = None


class _GroupedAnchorFrontier:
    """Same native wave in each existing environment owner, one frozen scorer."""
    def __init__(self, vector, slots, roots, worlds, options):
        self.vector = vector
        self.members = {}
        for root, slot in enumerate(slots):
            group, local = vector._env_to_group[slot]
            self.members.setdefault(group, []).append((root, local))
        self.requests = {group: dict(indices=[local for _, local in members],
            roots=[roots[root] for root, _ in members], worlds=[worlds[root] for root, _ in members],
            options=dict(options, threads=max(1, options['threads']//len(self.members))))
            for group, members in self.members.items()}
        self.results = [None]*len(slots)
        self.finished, self.row_counts = {}, {}
        self.started = self.closed = False

    def advance(self, outputs=None):
        if self.closed or (outputs is None) == self.started:
            raise RuntimeError('out-of-order native frontier response')
        if self.started and len(outputs) != sum(self.row_counts[g] for g in self.members if g not in self.finished):
            raise ValueError('native frontier output count mismatch')
        pending = {}
        at = 0
        for group in sorted(self.members):
            if group in self.finished:
                continue
            if not self.started:
                command, data = 'native_anchor_start', self.requests[group]
            else:
                count = self.row_counts[group]
                command, data = 'native_anchor_advance', outputs[at:at+count]
                at += count
            remote = self.vector.remotes[group]
            remote.send((command, data))
            pending[remote] = group
        self.started = True
        packets = self.vector._receive_groups(pending)
        rows = []
        for group in sorted(packets):
            packet, members = packets[group], self.members[group]
            if packet['done']:
                self.finished[group] = packet['stats']
                for (root, _), result in zip(members, packet['results'], strict=True):
                    self.results[root] = result
            else:
                self.row_counts[group] = len(packet['rows'])
                rows.extend((members[root][0], job, kind, mask, delta)
                            for root, job, kind, mask, delta in packet['rows'])
        if rows:
            return dict(done=False, rows=rows)
        if len(self.finished) != len(self.members) or any(r is None for r in self.results):
            raise RuntimeError('native frontier lost a root')
        stats = {}
        for counts in self.finished.values():
            for key, value in counts.items():
                # Peak concurrent lanes are group-local, as in grouped CPU
                # search. Other counters are disjoint exact integer work.
                stats[key] = max(stats.get(key, 0), value) if key == 'max_active' else stats.get(key, 0)+value
        return dict(done=True, results=self.results, stats=stats)

    def close(self):
        if self.closed:
            return
        if self.started:
            pending = {}
            for group in self.members:
                remote = self.vector.remotes[group]
                remote.send(('native_anchor_close', None))
                pending[remote] = group
            self.vector._receive_groups(pending)
        self.closed = True
