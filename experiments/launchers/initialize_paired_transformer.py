"""Initialize COMBAT38 using the COMBAT37 architecture template and entirely new weights."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.independent_rollout import IndependentRolloutConfig
from rl.algorithms.upper_probe import state_digest
from rl.policies.small_playout import OnlinePlayoutConfig
from rl.policies.token_regret import TOKEN_REGRET_CONTRACT


def initialize(template, output, seed):
    if output.exists():
        raise FileExistsError(output)
    model = AccumulatingMaskablePPO.load(template, device="cpu")
    if model.num_timesteps or model._n_updates or getattr(model, "episode_rollout_config", None):
        raise ValueError("use the untrained upper architecture template")
    parent_upper = state_digest(model.policy.core.state_dict())
    parent_guide = model.policy.small_playout.export().copy()
    upper = dict(model.policy_kwargs["upper_config"], learning_rate=1.2e-4,
                 critic_effective_batch_size=32768, measure_value_loss=True)
    lower = OnlinePlayoutConfig(context_mode="token-root-64x2", optimizer="adamw",
            epochs=4, batch_size=4096, micro_batch_size=256, learning_rate=3e-4,
            weight_decay=1e-4, max_grad_norm=1.)
    model.policy_kwargs = dict(model.policy_kwargs, upper_config=upper, small_playout_config=asdict(lower))
    model.seed = seed
    model._setup_model()
    model.small_playout_search_contract = TOKEN_REGRET_CONTRACT
    model.independent_rollout_config = asdict(IndependentRolloutConfig(seed=seed))
    model.small_playout_optimizer_steps = 0
    model.episode_rollouts_completed = 0
    identity = state_digest(model.policy.state_dict())
    assert parent_upper != state_digest(model.policy.core.state_dict())
    assert (parent_guide != model.policy.small_playout.export_guide()).any()
    opts = [v for v in vars(model.policy).values() if isinstance(v, torch.optim.Optimizer)]
    assert opts and all(not opt.state for opt in opts)
    model.save(output)
    restored = AccumulatingMaskablePPO.load(output, device="cpu")
    assert state_digest(restored.policy.state_dict()) == identity
    assert restored.num_timesteps == restored.small_playout_optimizer_steps == restored._n_updates == 0
    print(json.dumps(dict(seed=seed, policy_sha256=identity, optimizers=len(opts),
                          lower_parameters=sum(v.numel() for v in model.policy.small_playout.parameters()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    initialize(args.template, args.output, args.seed)
