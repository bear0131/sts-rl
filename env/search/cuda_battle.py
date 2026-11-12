"""Owned, model-free CUDA battle states; not a training search backend.

The caller supplies a separately built and hash-verified simulator module.
Native callback queues cannot be imported from CPU. Queues created on device
stay with this module through forks, graph retirement and state destruction.
"""
from collections import OrderedDict
import ctypes as c
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


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
                 graph_capacity=2, device='cuda'):
        if min(capacity, card_count, deck_capacity) < 1 or graph_capacity < 0:
            raise ValueError('invalid resident battle capacity')
        data = Path(cubin).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('resident simulator module hash mismatch')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('resident battles require CUDA')
        self.capacity, self.graph_capacity = int(capacity), int(graph_capacity)
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
                for name in ('layout', 'load', 'observe', 'apply', 'snapshot', 'fork', 'destroy', 'first'):
                    f = p()
                    self._check(self.driver.cuModuleGetFunction(c.byref(f), self.module, ('deck_' + name).encode()))
                    self.functions[name] = f
                f = p()
                if self.driver.cuModuleGetFunction(c.byref(f), self.module, b'deck_observe_live_block') == 0:
                    self.functions['observe_live'] = f
                for name, symbol in (('observe_indexed', 'deck_observe_indexed_block'),
                                     ('apply_compact', 'deck_apply_compact')):
                    f = p()
                    if self.driver.cuModuleGetFunction(c.byref(f), self.module, symbol.encode()) == 0:
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
                        self._check(self.driver.cuModuleGetFunction(c.byref(f), self.module, ('deck_' + name).encode()))
                        self.functions[name] = f
                self.arena = self._empty((capacity, self.battle_bytes), torch.uint8)
                self.constructed = self._empty((capacity,)).zero_()
                self.prefix = self._empty((capacity,)).zero_()
                self.selected = self._empty((capacity,)).fill_(-1)
                self.categorical = self._empty((capacity, cats))
                self.scalars = self._empty((capacity, floats), torch.float32)
                self.bits = self._empty((capacity, self.actions))
                self.mask = self._empty((capacity, self.actions))
                self.outcome = self._empty((capacity,))
                self.upgrades = self._empty((capacity, 6, card_count), torch.float32)
                self.misc = self._empty((capacity, deck_capacity))
                self.observe_args = [self.arena, self.constructed, self.prefix, self.categorical,
                    self.scalars, self.bits, self.mask, self.outcome, self.upgrades, self.misc]
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

    def _assert_owner(self):
        if self.closed:
            raise RuntimeError('resident battle arena is closed')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('resident battle operations must use their owning stream')

    def _launch(self, name, tensors, n=None):
        values = [c.c_void_p(t.data_ptr()) for t in tensors]
        if n is not None:
            values.append(c.c_int(n))
        arguments = (c.c_void_p * len(values))(*(c.cast(c.byref(v), c.c_void_p) for v in values))
        grid, threads = (n, 64) if name in ('observe_live', 'observe_indexed') else (((n or 1) + 31) // 32, 32)
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

    def fork(self, sources, destinations, worlds):
        self._assert_owner()
        sources, destinations = np.asarray(sources), np.asarray(destinations)
        if (not len(sources) or len(sources) != len(destinations) or len(sources) != len(worlds)
                or len(set(destinations)) != len(destinations)
                or np.any(sources < 0) or np.any(sources >= self.capacity)
                or np.any(destinations < 0) or np.any(destinations >= self.capacity)
                or not self.loaded[sources].all() or np.intersect1d(sources, destinations).size):
            raise ValueError('fork sources must be live and disjoint from unique destinations')
        self._launch('fork', [self.arena, self.constructed, self.prefix, self._upload(sources),
            self._upload(destinations), self._upload(worlds, np.uint64)], len(sources))
        self.loaded[destinations] = True

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
