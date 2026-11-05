import numpy as np
import torch

from rl.policies.small_playout import SmallPlayoutPolicy, PreparedPlayoutSamples
from tools.analysis.fit_return_supervision import assess


def test_evaluate_all_actions_but_fit_only_measured_actions():
    policy = SmallPlayoutPolicy(8)
    for p in policy.parameters():
        torch.nn.init.zeros_(p)
    rows = []
    for n in (3, 2):
        rows.append(
            dict(
                action_bits=np.arange(n),
                state=np.zeros(32, dtype=np.float32),
                actions=np.zeros((n, 24), dtype=np.float32),
                keys=np.zeros((n, 10), dtype=np.int64),
                recommended_action=0,
                visits=np.ones(n),
            )
        )
    bank = PreparedPlayoutSamples(rows, 0.2)
    bank.coverage = np.array([True, False, True, True, True])
    y = np.array([0.5, 0, 0.5, 1.0, 0.0])
    metrics = assess(
        policy,
        bank,
        {"old": y, "new": y},
        np.array([0.1, 0.9, 0.2, 0.8, 0.2]),
        np.array([1, 0]),
        torch.device("cpu"),
    )
    np.testing.assert_allclose(metrics[:, 0], [0.5, 0.4], atol=1e-6)
    np.testing.assert_allclose(metrics[:, 1], [0.8, 0.1], atol=1e-6)
    np.testing.assert_allclose(metrics[:, 2:], [[np.log(2)] * 2, [0.0] * 2], atol=1e-6)
