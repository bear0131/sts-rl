"""Native small-policy search adapter shared by evaluation and training."""
from dataclasses import asdict, dataclass

import env.core  # resolve the explicitly selected native build before import
import slaythespire as sts

from rl.policies.small_playout import CONTRACT


@dataclass(frozen=True)
class SmallSearchConfig:
    simulations: int = 250
    playout_depth: int = 200
    tree_depth: int = 1024
    exploration: float = -1.
    temperature: float = 1.
    use_prior: bool = False
    prior_mix: float = .5
    prior_temperature: float = 1.
    tie_epsilon: float = .02
    root_guards: bool = True

    def native(self):
        config = sts.SmallPlayoutConfig()
        for key, value in asdict(self).items():
            setattr(config, key, value)
        return config


class SmallPlayoutSearch:
    def __init__(self, policy, config=SmallSearchConfig(), threads=1, prior=None):
        if threads < 1:
            raise ValueError('native search threads must be positive')
        self.config, self.threads = config, threads
        self.pool = sts.NeuralSearchBatch(threads)
        self.prior = None if prior is None else sts.SmallPlayoutNet(prior.export())
        self.refresh(policy)

    def refresh(self, policy):
        if policy.key_count != sts.small_playout_key_count():
            raise ValueError('native/Python action vocabulary mismatch')
        self.net = sts.SmallPlayoutNet(policy.export())

    def search(self, roots, nonces, configs=None):
        return sts.small_playout_search_batch(
            self.pool, self.net, roots, [c.native() for c in configs] if configs is not None
            else [self.config.native() for _ in roots], nonces, self.prior)

    def manifest(self):
        return dict(contract=CONTRACT, config=asdict(self.config), native_threads=self.threads,
                    value_head=False, training_objective='mcts-visit-cross-entropy',
                    tree_selection='uct-or-separate-puct', root_selection='q-band-visits-guards')
