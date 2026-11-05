"""Worker-side compact CPU search; no Torch or learner dependencies."""
import numpy as np
import slaythespire as sts
from env.core.boss_value import enum_name, is_boss_room

SEARCH_CONTRACT = 'terminal-hp-boss-win-turn128-v1'
PAIRED_CONTRACT = 'paired-worldlines-eps005-terminal-hp-boss-win-turn128-v1'


class SmallPlayoutBatch:
    def __init__(self, envs, pool, pair):
        public = pair['contract'] == 'sts1-small-playout-public-v1'
        mode = pair.get('context_mode', 'none')
        if ((public and mode not in ('root','turn','state')) or (not public and mode != 'none')
                or (not public and pair['contract'] != 'sts1-small-playout-v1')):
            raise ValueError('unknown small playout feature contract')
        self.paired_worldlines = pair.get('search_contract') == PAIRED_CONTRACT
        if self.paired_worldlines and (public or pair.get('prior') is not None or
                                       getattr(sts,'small_paired_search_contract',None) != PAIRED_CONTRACT):
            raise ValueError('paired worldlines require the versioned native path and one compact v1 scorer')
        if (pair.get('search_contract') not in (SEARCH_CONTRACT, PAIRED_CONTRACT)
                or getattr(sts, 'small_playout_search_contract', None) != SEARCH_CONTRACT):
            raise ValueError('incompatible small playout search objective/native build')
        self.envs, self.pool, self.pair = envs, pool, pair
        self.adapter = None
        if public:
            if (getattr(sts,'small_playout_public_contract',None) != 'sts1-playout-public-entities-v1'
                    or pair.get('prior') is not None):
                raise ValueError('public compact policy requires the public native contract and no separate prior')
            split = 4984 + 33*sts.small_playout_key_count()
            self.net = sts.SmallPlayoutNet(pair['policy'][:split])
            self.adapter = sts.PublicPlayoutAdapter(pair['policy'][split:])
        else:
            self.net = sts.SmallPlayoutNet(pair['policy'])
        self.prior = sts.SmallPlayoutNet(pair['prior']) if pair.get('prior') is not None else None
        self.stats = {}
        self.samples = []

    def tier(self, env):
        room = env.thick.gc.cur_room
        return 'boss' if is_boss_room(room) else 'elite' if 'ELITE' in enum_name(room) else 'common'

    def decisions(self, calls, *, pre_step=None, start_steps=None):
        out, jobs = self.prepare(calls)
        if jobs:
            callbacks = {}
            if pre_step is not None:
                # Inline decisions must be captured before callbacks start.
                inline = tuple(out.items())
                def started():
                    if start_steps is not None:
                        start_steps()
                    for index, decision in inline:
                        pre_step(index, decision['action'])
                def ready(row, bits):
                    index = jobs[row][0]
                    out[index] = self.envs[index]._apply_rulebase_turn_decision(bits, {})
                    pre_step(index, out[index]['action'])
                callbacks = dict(on_ready=ready, on_start=started)
            results = sts.small_playout_search_batch(self.pool, self.net,
                [self.envs[j[0]].thick.bc for j in jobs], [j[2] for j in jobs], [j[3] for j in jobs], self.prior,
                **callbacks)
            for job, result in zip(jobs, results, strict=True):
                index = job[0]
                decision = self.accept(job, result, apply_action=index not in out)
                if decision is not None:
                    out[index] = decision
        return {index: out[index] for index in calls}

    def prepare(self, calls):
        """Prepare independent roots; only the owning control thread mutates envs."""
        out, jobs = {}, []
        for index, (args, kw) in calls.items():
            if args or set(kw) != {'state_nonce'}:
                raise ValueError('small decision requires only its explicit nonce')
            env = self.envs[index]
            if env.thick is None or env._phase() != 'combat' or env.thick.bc is None:
                raise ValueError('compact search requires a live combat')
            if env._rulebase_pending_multi is not None:
                out[index] = env._apply_rulebase_turn_decision((3 << 29) | env._rulebase_pending_multi, {})
                continue
            tier = self.tier(env)
            nonce = int(kw['state_nonce']) & ((1 << 64)-1)
            relics = tuple(r['id'] for r in env.thick.gc.relic_state())
            cfg = sts.SmallPlayoutConfig()
            if self.adapter is not None:
                if self.pair['context_mode']=='turn':
                    cfg.condition_turns(self.adapter,env.thick.gc,env.thick.bc)
                else:
                    cfg.condition(self.adapter,env.thick.gc,env.thick.bc,self.pair['context_mode']=='state')
            if 'Runic Dome' in relics:
                # No search may consume the simulator's concealed current move.
                # The compact feature encoder suppresses every intent feature.
                f = sts.small_playout_features(env.thick.bc)
                raw = (self.net.conditioned_logits(env.thick.bc,cfg) if self.adapter is not None
                       else self.net.logits(env.thick.bc))
                logits = np.asarray(raw, np.float64)
                p = np.exp(logits-logits.max()); p /= p.sum()
                chosen = np.random.default_rng(nonce).choice(len(p), p=p)
                bits = f['action_bits'][chosen]
                out[index] = env._apply_rulebase_turn_decision(bits, {'small_dome_policy_decisions': 1})
                self.stats['dome_policy_decisions'] = self.stats.get('dome_policy_decisions', 0)+1
                self.stats['dome_turn_card_limit_decisions'] = self.stats.get('dome_turn_card_limit_decisions', 0)+int(sts.small_playout_turn_card_limit_reached(env.thick.bc))
                continue
            if self.paired_worldlines:
                cfg.paired_worldlines = True
            cfg.simulations = self.pair[tier+'_simulations']
            cfg.playout_depth = self.pair['playout_depth']
            cfg.tree_depth = self.pair['tree_depth']
            cfg.boss = tier == 'boss'
            cfg.use_prior = self.prior is not None and tier in ('elite', 'boss')
            jobs.append((index, tier, cfg, nonce))
        # A public, cheap cost hint avoids sending the longest bosses last.
        jobs.sort(key=lambda j: j[2].simulations, reverse=True)
        return out, jobs

    def accept(self, job, result, *, apply_action=True):
        index, tier, _, _ = job
        if self.pair.get('online') and len(result['action_bits']) > 1:
            self.samples.append({key: np.asarray(result[key], dtype=dtype) for key, dtype in
                (('state', np.float32), ('actions', np.float32), ('keys', np.int32),
                 ('action_bits', np.uint32), ('visits', np.float32))})
            self.samples[-1]['recommended_action'] = int(result['action'])
            self.samples[-1]['env'] = index
            if self.adapter is not None:
                self.samples[-1]['public_context'] = job[2].public_context
        self.stats[tier+'/decisions'] = self.stats.get(tier+'/decisions', 0)+1
        for key, value in result['stats'].items():
            name = tier+'/'+key
            if key == 'playout_lengths':
                self.stats.setdefault(name, np.zeros(201, dtype=np.int64))
                self.stats[name] += value
            else:
                self.stats[name] = self.stats.get(name, 0)+value
        if apply_action:
            return self.envs[index]._apply_rulebase_turn_decision(result['action'], {})

    def scalar_stats(self):
        out = {k: float(v) for k, v in self.stats.items() if not k.endswith('playout_lengths')}
        for tier in ('common', 'elite', 'boss'):
            hist = self.stats.get(tier+'/playout_lengths', np.zeros(201, np.int64))
            n = hist.sum()
            out[tier+'/playout_count'] = float(n)
            out[tier+'/playout_length_mean'] = float(np.dot(hist, np.arange(201))/n) if n else 0.
            for q in (.95, .99):
                out[tier+f'/playout_length_p{int(q*100)}'] = float(np.searchsorted(hist.cumsum(), q*n)) if n else 0.
        return out


class SmallPlayoutStream:
    """Diagnostic independent-root scheduling on GIL-free native searches.

    One controller submits, receives and steps environments. Each logical slot
    has at most one outstanding root; weights remain immutable until close.
    Completion order is explicit and must not be treated as learner sample order.
    This is not enabled by the production collector.
    """
    def __init__(self, batch, *, threads):
        from concurrent.futures import ThreadPoolExecutor
        if type(threads) is not int or threads < 1:
            raise ValueError('positive search thread count required')
        self.batch = batch
        self._executor = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='sts-search')
        self._pending = {}
        self._inline = {}
        self._closed = False

    @property
    def pending(self):
        return bool(self._pending or self._inline)

    def submit(self, calls):
        if self._closed:
            raise RuntimeError('search stream is closed')
        occupied = set(self._inline) | {job[0] for job in self._pending.values()}
        if occupied.intersection(calls):
            raise RuntimeError('environment already has an outstanding search')
        if any(type(i) is not int or not 0 <= i < len(self.batch.envs) for i in calls):
            raise ValueError('search slot outside environment vector')
        inline, jobs = self.batch.prepare(calls)
        self._inline.update(inline)
        for job in jobs:
            index, _, config, nonce = job
            future = self._executor.submit(self.batch.net.search,
                self.batch.envs[index].thick.bc, config, nonce, self.batch.prior)
            self._pending[future] = job

    def receive(self, timeout=0.):
        from concurrent.futures import FIRST_COMPLETED, wait
        if self._closed:
            raise RuntimeError('search stream is closed')
        ready = [future for future in self._pending if future.done()]
        if not ready and not self._inline and self._pending:
            ready = wait(self._pending, timeout=timeout, return_when=FIRST_COMPLETED).done
        out, self._inline = self._inline, {}
        try:
            for future in ready:
                job = self._pending.pop(future)
                out[job[0]] = self.batch.accept(job, future.result())
        except BaseException:
            self.close()
            raise
        return out

    def close(self):
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()
            self._inline.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
