"""Worker-side compact CPU search; no Torch or learner dependencies."""
import numpy as np
import slaythespire as sts
from env.core.boss_value import enum_name, is_boss_room


class SmallPlayoutBatch:
    def __init__(self, envs, pool, pair):
        if pair['contract'] != 'sts1-small-playout-v1':
            raise ValueError('unknown small playout feature contract')
        self.envs, self.pool, self.pair = envs, pool, pair
        self.net = sts.SmallPlayoutNet(pair['policy'])
        self.prior = sts.SmallPlayoutNet(pair['prior'])
        self.stats = {}

    def tier(self, env):
        room = env.thick.gc.cur_room
        return 'boss' if is_boss_room(room) else 'elite' if 'ELITE' in enum_name(room) else 'common'

    def decisions(self, calls):
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
            if 'Runic Dome' in relics:
                # No search may consume the simulator's concealed current move.
                # The compact feature encoder suppresses every intent feature.
                f = sts.small_playout_features(env.thick.bc)
                logits = np.asarray(self.net.logits(env.thick.bc), np.float64)
                p = np.exp(logits-logits.max()); p /= p.sum()
                chosen = np.random.default_rng(nonce).choice(len(p), p=p)
                bits = f['action_bits'][chosen]
                out[index] = env._apply_rulebase_turn_decision(bits, {'small_dome_policy_decisions': 1})
                self.stats['dome_policy_decisions'] = self.stats.get('dome_policy_decisions', 0)+1
                continue
            cfg = sts.SmallPlayoutConfig()
            cfg.simulations = self.pair[tier+'_simulations']
            cfg.playout_depth = self.pair['playout_depth']
            cfg.tree_depth = self.pair['tree_depth']
            cfg.use_prior = tier in ('elite', 'boss')
            jobs.append((index, tier, cfg, nonce))
        # A public, cheap cost hint avoids sending the longest bosses last.
        jobs.sort(key=lambda j: j[2].simulations, reverse=True)
        if jobs:
            results = sts.small_playout_search_batch(self.pool, self.net,
                [self.envs[j[0]].thick.bc for j in jobs], [j[2] for j in jobs], [j[3] for j in jobs], self.prior)
            for (index, tier, _, _), result in zip(jobs, results, strict=True):
                self.stats[tier+'/decisions'] = self.stats.get(tier+'/decisions', 0)+1
                for key, value in result['stats'].items():
                    name = tier+'/'+key
                    if key == 'playout_lengths':
                        self.stats.setdefault(name, np.zeros(201, dtype=np.int64))
                        self.stats[name] += value
                    else:
                        self.stats[name] = self.stats.get(name, 0)+value
                out[index] = self.envs[index]._apply_rulebase_turn_decision(result['action'], {})
        return {index: out[index] for index in calls}

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
