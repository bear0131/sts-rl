"""Create a fresh, width-reduced split-pointer checkpoint.

The source checkpoint supplies only the environment and trainer metadata.  Its
policy tensors, optimizer state, replay, and progress are deliberately dropped.
This is the supported path for changing Transformer width because SB3 cannot
load tensors whose shapes no longer match the policy geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy


def create_fresh_model(source: Path, *, d_model: int, n_layers: int,
                       n_heads: int, seed: int = 0):
    if d_model < 1 or n_layers < 2 or n_heads < 1:
        raise ValueError("invalid policy geometry")
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")

    model = AccumulatingMaskablePPO.load(source, device="cpu")
    if type(model.policy) is not SplitStSPointerPolicy:
        raise ValueError("source checkpoint must use SplitStSPointerPolicy")
    if not (model.nosl_is_az and model.nosl_is_az_inner_only):
        raise ValueError("source checkpoint must be inner-only NOSL")

    kwargs = dict(model.policy_kwargs or {})
    kwargs.update(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    kwargs.pop("card_vector_checkpoint", None)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = SplitStSPointerPolicy(
            model.observation_space,
            model.action_space,
            model.lr_schedule,
            **kwargs,
        )

    model.policy = policy
    model.policy_kwargs = kwargs
    model.seed = seed
    model.num_timesteps = model._n_updates = model._total_timesteps = 0
    model._num_timesteps_at_start = 0
    model._last_obs = model._last_episode_starts = None
    model._recovery_environment_restored = False
    model._nosl_is_az_replay = None
    model._nosl_is_az_pending_fights = []
    model._nosl_is_az_samples = []
    model._nosl_is_az_fresh_samples = []
    model.configure_nosl_is_az_inner_only()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.suffix != ".zip":
        args.out = Path(str(args.out) + ".zip")
    if args.out.exists():
        raise FileExistsError(args.out)

    model = create_fresh_model(
        args.source,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    combat_count = sum(p.numel() for p in model.policy.core.combat.parameters())
    total_count = sum(p.numel() for p in model.policy.parameters())
    print(f"saved={args.out} step=0 replay=empty Adam=fresh seed={args.seed}")
    print(
        f"policy_kwargs={model.policy_kwargs} "
        f"combat_parameters={combat_count:,} total_parameters={total_count:,}"
    )


if __name__ == "__main__":
    main()
