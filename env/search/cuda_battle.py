"""Owned, model-free CUDA battle states; not a training search backend.

The caller supplies a separately built and hash-verified simulator module.
Native callback queues cannot be imported from CPU. Queues created on device
stay with this module through forks, graph retirement and state destruction.
"""
from collections import OrderedDict
import ctypes as c
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def upload_arrays(host, device, arrays):
    """One owned byte transfer with typed, aligned views for existing consumers."""
    arrays = [np.ascontiguousarray(a) for a in arrays]
    packed,offsets = host.pack_arrays(arrays)
    storage = torch.from_numpy(packed).to(device)
    return [storage[offset:offset+a.nbytes].view(torch.from_numpy(a).dtype).reshape(a.shape)
            for offset,a in zip(offsets,arrays,strict=True)]


def load_artifacts(folder):
    """Verify both halves of one simulator build before importing CPU states."""
    folder = Path(folder)
    manifest = json.loads((folder / 'artifacts.json').read_text())
    if manifest['contract'] != 'deck-resident-battle-v1':
        raise ValueError('unknown resident simulator contract')
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'resident simulator artifact mismatch: {name}')
    name = 'deck_cuda_bridge.cpython-311-x86_64-linux-gnu.so'
    spec = importlib.util.spec_from_file_location('deck_cuda_bridge', folder / name)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    if getattr(bridge, 'transport_version', 1) != manifest.get('transport_version', 1):
        raise ValueError('resident simulator transport version mismatch')
    return folder, manifest, bridge


class CudaBattleArena:
    """Fixed state storage and bounded observe/apply graph pairs.

    Operations are serialized on the creating CUDA stream. Returned observation
    views are overwritten by the next observe. The module must outlive every
    state and graph; close destroys graphs, then states, then the module.
    This adapter does not sample actions, change RNG algorithms or score leaves.
    """

    _stack_owners = {}  # Primary CUDA context, shared by live simulator arenas.

    def __init__(self, cubin, sha256, capacity, *, card_count, deck_capacity=96,
                 graph_capacity=2, device='cuda', paged=False, observation_capacity=None):
        if min(capacity, card_count, deck_capacity) < 1 or graph_capacity < 0:
            raise ValueError('invalid resident battle capacity')
        data = Path(cubin).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('resident simulator module hash mismatch')
        self.host = load_artifacts(Path(cubin).parent)[2]
        if getattr(self.host,'host_runtime_version',None) != 1:
            raise ValueError('resident search requires the versioned native host runtime')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('resident battles require CUDA')
        self.capacity, self.graph_capacity = int(capacity), int(graph_capacity)
        self.paged = bool(paged)
        self.observation_capacity = int(observation_capacity or capacity)
        if not 0 < self.observation_capacity <= capacity or (not paged and self.observation_capacity != capacity):
            raise ValueError('compact observation frames require paged states')
        self.graphs = OrderedDict()
        self.closed = False
        self.loaded = np.zeros(capacity, bool)
        self.counts = dict(graph_captures=0, graph_replays=0, graph_evictions=0)
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream()
            torch.empty(1, device=self.device)  # Establish the primary context.
            self.driver = c.CDLL('libcuda.so.1')
            p = c.c_void_p
            self.driver.cuModuleLoadData.argtypes = [c.POINTER(p), p]
            self.driver.cuModuleGetFunction.argtypes = [c.POINTER(p), p, c.c_char_p]
            self.driver.cuModuleUnload.argtypes = [p]
            self.driver.cuCtxSetLimit.argtypes = [c.c_int, c.c_size_t]
            self.driver.cuCtxGetLimit.argtypes = [c.POINTER(c.c_size_t), c.c_int]
            self.driver.cuLaunchKernel.argtypes = [p] + [c.c_uint] * 7 + [p, p, p]
            self.stack_device = torch.cuda.current_device()
            self.module = p()
            self._acquire_stack_limit()
            try:
                # Dynamic monster callbacks have an unknown static stack bound.
                # The old observation kernel incidentally made the driver grow the
                # stack beyond 64 KiB; direct observation must not rely on that.
                current = c.c_size_t()
                self._check(self.driver.cuCtxGetLimit(c.byref(current), 2))
                if current.value < 512 * 1024**2:
                    self._check(self.driver.cuCtxSetLimit(2, 512 * 1024**2))
                self.module = p()
                image = c.create_string_buffer(data)
                self._check(self.driver.cuModuleLoadData(c.byref(self.module), image))
                self.functions = {}
                symbol_prefix = 'deck_paged_' if self.paged else 'deck_'
                for name in ('layout', 'load', 'observe', 'apply', 'snapshot', 'fork', 'destroy', 'first'):
                    f = p()
                    self._check(self.driver.cuModuleGetFunction(c.byref(f), self.module, (symbol_prefix + name).encode()))
                    self.functions[name] = f
                f = p()
                if self.driver.cuModuleGetFunction(c.byref(f), self.module, (symbol_prefix+'observe_live_block').encode()) == 0:
                    self.functions['observe_live'] = f
                for name, symbol in (('observe_indexed', 'deck_observe_indexed_block'),
                                     ('observe_search', 'deck_observe_search_block'),
                                     ('apply_compact', 'deck_apply_compact')):
                    f = p()
                    if self.driver.cuModuleGetFunction(c.byref(f), self.module, symbol.replace('deck_',symbol_prefix).encode()) == 0:
                        self.functions[name] = f
                if self.paged:
                    for name in ('observe_indexed', 'observe_search'):
                        f = p()
                        self._check(self.driver.cuModuleGetFunction(c.byref(f), self.module,
                            ('deck_paged_frame_'+name.removeprefix('observe_')).encode()))
                        self.functions[name] = f
                    for name in ('load_indexed','destroy_indexed'):
                        f = p()
                        self._check(self.driver.cuModuleGetFunction(c.byref(f),self.module,
                                                                  ('deck_paged_'+name).encode()))
                        self.functions[name] = f
                layout = self._empty((8,), torch.int64)
                self._launch('layout', [layout])
                self.battle_bytes, cats, floats, self.actions, cards, deck, version, words = layout.cpu().tolist()
                if (self.battle_bytes < 1 or (cats, self.actions, words) != (226, 198, 8192)
                        or version not in (1, 2)
                        or (cards, deck) != (card_count, deck_capacity)):
                    raise ValueError('unsupported resident simulator layout')
                self.transport_version = version
                if version == 2:
                    for name in ('load_variable', 'snapshot_sizes', 'snapshot_variable'):
                        f = p()
                        self._check(self.driver.cuModuleGetFunction(c.byref(f), self.module, (symbol_prefix + name).encode()))
                        self.functions[name] = f
                self.arena = (self._empty((capacity,), torch.int64).zero_() if self.paged
                              else self._empty((capacity, self.battle_bytes), torch.uint8))
                self.constructed = self._empty((capacity,)).zero_()
                self.prefix = self._empty((capacity,)).zero_()
                self.selected = self._empty((capacity,)).fill_(-1)
                self.categorical = self._empty((capacity, cats))
                self.scalars = self._empty((self.observation_capacity, floats), torch.float32)
                self.bits = self._empty((capacity, self.actions))
                self.mask = self._empty((capacity, self.actions))
                self.outcome = self._empty((capacity,))
                self.upgrades = self._empty((self.observation_capacity, 6, card_count), torch.float32)
                self.misc = self._empty((capacity, deck_capacity))
                self.observation_slots = self._empty((capacity,), torch.int64) if self.paged else None
                self.observe_args = [self.arena, self.constructed, self.prefix, self.categorical,
                    self.scalars, self.bits, self.mask, self.outcome, self.upgrades, self.misc]
                if self.paged:
                    self.observe_args.append(self.observation_slots)
                self.apply_args = [self.arena, self.constructed, self.prefix, self.selected, self.bits, self.mask]
            except BaseException:
                if self.module.value:
                    self.driver.cuModuleUnload(self.module)
                self._release_stack_limit()
                raise

    def _acquire_stack_limit(self):
        owners = type(self)._stack_owners
        entry = owners.get(self.stack_device)
        if entry is None:
            current = c.c_size_t()
            self._check(self.driver.cuCtxGetLimit(c.byref(current), 0))
            required = max(current.value, 80 * 1024)
            if required > current.value:
                # A preceding PPO update can leave the otherwise idle caching
                # allocator holding most VRAM. The driver cannot reuse it for
                # its stack reservation. Live tensors/graphs remain owned.
                torch.cuda.synchronize(self.device)
                gc.collect()
                torch.cuda.empty_cache()
            self._check(self.driver.cuCtxSetLimit(0, required))
            entry = owners[self.stack_device] = dict(original=current.value, required=required, count=0)
        entry['count'] += 1

    def _release_stack_limit(self):
        owners = type(self)._stack_owners
        entry = owners[self.stack_device]
        entry['count'] -= 1
        if entry['count'] == 0:
            current = c.c_size_t()
            self._check(self.driver.cuCtxGetLimit(c.byref(current), 0))
            # Do not overwrite a later external context-limit change.
            if current.value == entry['required']:
                self._check(self.driver.cuCtxSetLimit(0, entry['original']))
            del owners[self.stack_device]

    @staticmethod
    def _check(rc):
        if rc:
            raise RuntimeError(f'resident CUDA driver failure {rc}; do not reuse this arena')

    def _empty(self, shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _upload(self, values, dtype=np.int32):
        array = np.ascontiguousarray(values, dtype=dtype)
        if array.dtype == np.uint64:
            array = array.view(np.int64)
        return torch.from_numpy(array).to(self.device)

    def _upload_many(self, arrays):
        return upload_arrays(self.host,self.device,arrays)

    def _assert_owner(self):
        if self.closed:
            raise RuntimeError('resident battle arena is closed')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('resident battle operations must use their owning stream')

    def _launch(self, name, tensors, n=None, *, dispatch_n=None):
        values = [c.c_void_p(t.data_ptr()) for t in tensors]
        if n is not None:
            values.append(c.c_int(n))
        arguments = (c.c_void_p * len(values))(*(c.cast(c.byref(v), c.c_void_p) for v in values))
        # Indexed work consumes a compact frame, but its physical indices
        # still address the full arena. Keep launch extent and bounds separate.
        extent = n if dispatch_n is None else dispatch_n
        grid, threads = (extent, 64) if name in ('observe_live', 'observe_indexed', 'observe_search') else (((extent or 1) + 31) // 32, 32)
        self._check(self.driver.cuLaunchKernel(self.functions[name], grid,
            1, 1, threads, 1, 1, 0, c.c_void_p(torch.cuda.current_stream(self.device).cuda_stream), arguments, None))

    def _rows(self, n):
        self._assert_owner()
        if not 0 < n <= self.capacity or not self.loaded[:n].all():
            raise ValueError('observe/apply requires a constructed contiguous prefix')

    def load(self, snapshots, prefixes):
        self._assert_owner()
        n = len(snapshots)
        if not 0 < n <= self.capacity or len(prefixes) != n:
            raise ValueError('invalid imported battle count')
        width = max(len(s['words']) for s in snapshots) if self.transport_version == 2 else 8192
        words = np.zeros((n, width), np.uint64)
        lengths = []
        for i, state in enumerate(snapshots):
            if not state['portable'] or state['queue_size'] or len(state['words']) > width:
                raise ValueError('CPU callback queues cannot be imported to CUDA')
            words[i, :len(state['words'])] = state['words']
            lengths.append(len(state['words']))
        self.prefix[:n].copy_(self._upload(prefixes))
        args = [self.arena, self.constructed, self._upload(words, np.uint64), self._upload(lengths)]
        if self.transport_version == 2:
            self._launch('load_variable', args + [self._upload([width])], n)
        else:
            self._launch('load', args, n)
        self.loaded[:n] = True

    def _validate_fork(self, sources, destinations, worlds):
        self._assert_owner()
        sources, destinations = np.asarray(sources,np.int64), np.asarray(destinations,np.int64)
        if len(sources)!=len(worlds):raise ValueError('fork worlds must match sources')
        self.host.validate_fork(sources,destinations,self.loaded,self.capacity)
        return sources, destinations

    def fork(self, sources, destinations, worlds):
        sources, destinations = self._validate_fork(sources, destinations, worlds)
        self._launch('fork', [self.arena, self.constructed, self.prefix, self._upload(sources),
            self._upload(destinations), self._upload(worlds, np.uint64)], len(sources))
        self.loaded[destinations] = True

    def load_slots(self, snapshots, prefixes, slots):
        """Import only newly leased root slots, without replacing live siblings."""
        self._assert_owner()
        if self.transport_version != 2:
            raise ValueError('streaming roots require the variable state transport')
        slots = np.asarray(slots, np.int64)
        if len(slots) != len(snapshots) or len(slots) != len(prefixes):
            raise ValueError('invalid root slot lease')
        if not len(slots):
            return
        words,sizes = self.host.pack_snapshots(snapshots,slots,self.capacity)
        width = words.shape[1]
        data,lengths,stride,indices,prefix = self._upload_many([
            words.view(np.int64),sizes,
            np.asarray([width],np.int32),slots.astype(np.int32),np.asarray(prefixes,np.int32)])
        if self.paged:
            self._launch('load_indexed',[self.arena,self.constructed,data,lengths,stride,
                                        indices],len(slots))
        else:
            for run in np.split(np.arange(len(slots)),np.flatnonzero(np.diff(slots)!=1)+1):
                j, slot = int(run[0]), int(slots[run[0]])
                self._launch('load_variable', [self.arena[slot:], self.constructed[slot:],
                    data[j:], lengths[j:], stride], len(run))
        self.prefix.index_copy_(0, indices.long(), prefix)
        self.loaded[slots] = True

    def start_selected(self, sources, destinations, worlds, action_bits, draw_selection):
        """The existing fork/first sequence with one shared admission transfer."""
        sources, destinations = self._validate_fork(sources,destinations,worlds)
        if len(action_bits)!=len(sources) or len(draw_selection)!=len(sources):
            raise ValueError('initial actions must match admitted jobs')
        source,destination,seeds,bits,draw,counts,offset = self._upload_many([
            sources.astype(np.int32),destinations.astype(np.int32),
            np.asarray(worlds,np.uint64).view(np.int64),np.asarray(action_bits,np.uint32),
            np.asarray(draw_selection,np.int32),
            np.asarray([min(self.observation_capacity,len(sources)-i)
                        for i in range(0,len(sources),self.observation_capacity)],np.int32),
            np.asarray([0],np.int64)])
        self._launch('fork',[self.arena,self.constructed,self.prefix,source,destination,seeds],len(sources))
        self.loaded[destinations] = True
        for j,start in enumerate(range(0,len(sources),self.observation_capacity)):
            stop=min(len(sources),start+self.observation_capacity)
            ids=destination[start:stop];count=counts[j:j+1]
            self._launch('observe_indexed',self.observe_args+[ids,count,offset],
                self.capacity,dispatch_n=stop-start)
            self.selected.fill_(-1)
            self._launch('first',[self.arena,self.constructed,self.prefix,source[start:stop],ids,
                bits[start:stop],draw[start:stop],self.bits,self.mask,self.selected],stop-start)
            self._launch('apply_compact',self.apply_args+[ids,count,offset],
                self.capacity,dispatch_n=stop-start)
        return source,destination,seeds

    def release_slots(self, slots):
        """Destroy a retired lease before its state pages become reusable."""
        self._assert_owner()
        slots = np.sort(np.asarray(slots, np.int64))
        if (len(np.unique(slots)) != len(slots) or np.any(slots < 0)
                or np.any(slots >= self.capacity)):
            raise ValueError('invalid state retirement')
        if self.paged and len(slots):
            self._launch('destroy_indexed',[self.arena,self.constructed,self._upload(slots)],len(slots))
        else:
            for group in np.split(slots, np.flatnonzero(np.diff(slots) != 1)+1):
                if len(group):
                    self._launch('destroy', [self.arena[int(group[0]):],
                        self.constructed[int(group[0]):]], len(group))
        self.loaded[slots] = False

    def replay_slots(self, slots, actions):
        self.observe_selected(slots)
        self.selected.fill_(-1)
        self.selected[self._upload(slots, np.int64)] = self._upload(actions)
        self._launch('apply_compact', self.apply_args + [self._upload(slots),
            self._upload([len(slots)]), self._upload([0], np.int64)], self.capacity,
            dispatch_n=len(slots))

    def _graphs(self, n):
        found = self.graphs.get(n)
        if found is not None:
            self.graphs.move_to_end(n)
            return found
        while len(self.graphs) >= self.graph_capacity:
            self.stream.synchronize()
            self.graphs.popitem(last=False)
            self.counts['graph_evictions'] += 1
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(self.stream)
        with torch.cuda.stream(stream):
            self._launch('observe', self.observe_args, n)
        stream.synchronize()
        observe, apply = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        # Capture does not advance state. Each graph has its own pool and owns
        # no model/cuBLAS workspace; its only mutable inputs are arena buffers.
        with torch.cuda.graph(observe, stream=stream):
            self._launch('observe', self.observe_args, n)
        with torch.cuda.graph(apply, stream=stream):
            self._launch('apply', self.apply_args, n)
        self.graphs[n] = (observe, apply)
        self.counts['graph_captures'] += 1
        return observe, apply

    def observe(self, n):
        if self.paged:
            raise ValueError('paged states use explicit bounded observation frames')
        self._rows(n)
        if self.graph_capacity:
            self._graphs(n)[0].replay()
            self.counts['graph_replays'] += 1
        else:
            self._launch('observe', self.observe_args, n)
        return dict(categorical=self.categorical[:n], scalars=self.scalars[:n],
            mask=self.mask[:n], outcome=self.outcome[:n], prefix=self.prefix[:n],
            upgrades=self.upgrades[:n], permanent_misc=self.misc[:n])

    def apply(self, selected):
        n = len(selected)
        self._rows(n)
        if isinstance(selected, torch.Tensor):
            if selected.device != self.selected.device or selected.dtype != torch.int32:
                raise ValueError('resident action tensor must be device int32')
            self.selected[:n].copy_(selected)
        else:
            selected = np.asarray(selected, np.int32)
            if np.any(selected < -1) or np.any(selected >= self.actions):
                raise ValueError('invalid relative atomic action')
            self.selected[:n].copy_(self._upload(selected))
        if self.graph_capacity:
            self._graphs(n)[1].replay()
            self.counts['graph_replays'] += 1
        else:
            self._launch('apply', self.apply_args, n)

    def first(self, sources, destinations, action_bits, draw_selection):
        """Execute root identities after world sampling, remapping draw UIDs."""
        self._assert_owner()
        n = int(max(destinations))+1
        self.observe(n)
        self.selected[:n].fill_(-1)
        self._launch('first', [self.arena, self.constructed, self.prefix,
            self._upload(sources), self._upload(destinations), self._upload(action_bits, np.uint32),
            self._upload(draw_selection), self.bits, self.mask, self.selected], len(sources))
        self.apply(self.selected[:n])

    def observe_selected(self, indices):
        """Refresh owned slots without touching other in-flight branches."""
        self._assert_owner()
        indices = np.asarray(indices, dtype=np.int64)
        if (not len(indices) or np.any(indices < 0) or np.any(indices >= self.capacity)
                or not self.loaded[indices].all() or len(indices) > self.observation_capacity):
            raise ValueError('indexed observation requires live owned slots')
        ids, count, offset = self._upload(indices), self._upload([len(indices)]), self._upload([0], np.int64)
        self._launch('observe_indexed', self.observe_args + [ids, count, offset],
                     self.capacity, dispatch_n=len(indices))
        return dict(categorical=self.categorical, scalars=self.scalars, mask=self.mask,
                    outcome=self.outcome, prefix=self.prefix, upgrades=self.upgrades,
                    permanent_misc=self.misc)

    def first_selected(self, sources, destinations, action_bits, draw_selection):
        """Admit replacement jobs while unrelated branches retain their state."""
        if len(destinations) > self.observation_capacity:
            for start in range(0,len(destinations),self.observation_capacity):
                end=start+self.observation_capacity
                self.first_selected(sources[start:end],destinations[start:end],
                                    action_bits[start:end],draw_selection[start:end])
            return
        self.observe_selected(destinations)
        self.selected.fill_(-1)
        self._launch('first', [self.arena, self.constructed, self.prefix,
            self._upload(sources), self._upload(destinations), self._upload(action_bits, np.uint32),
            self._upload(draw_selection), self.bits, self.mask, self.selected], len(sources))
        self._launch('apply_compact', self.apply_args + [self._upload(destinations),
            self._upload([len(destinations)]), self._upload([0], np.int64)], self.capacity,
            dispatch_n=len(destinations))

    def snapshot(self, n):
        self._rows(n)
        width = 8192
        if self.transport_version == 2:
            sizes = self._empty((n,))
            self._launch('snapshot_sizes', [self.arena, self.constructed, sizes], n)
            width = int(sizes.max().item())
        words, metadata = self._empty((n, width), torch.int64), self._empty((n, 5))
        args = [self.arena, self.constructed, words, metadata]
        if self.transport_version == 2:
            self._launch('snapshot_variable', args + [self._upload([width])], n)
        else:
            self._launch('snapshot', args, n)
        return words.cpu().numpy().view(np.uint64), metadata.cpu().numpy()

    def clear_graphs(self):
        self._assert_owner()
        self.stream.synchronize()
        self.graphs.clear()

    def close(self):
        if self.closed:
            return
        self._assert_owner()
        self.clear_graphs()
        self._launch('destroy', [self.arena, self.constructed], self.capacity)
        self.stream.synchronize()
        self._check(self.driver.cuModuleUnload(self.module))
        self._release_stack_limit()
        self.closed = True
