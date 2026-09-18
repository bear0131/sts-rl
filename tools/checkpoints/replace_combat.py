"""Create a fresh inner-only MLP experiment while retaining the frozen upper.

This is not a resume: old combat weights, Adam, replay and progress are discarded.
The semantic-checkpoint creator cannot serve this operation because it replaces
the entire policy, including the frozen overworld specialist.
"""

import argparse
from pathlib import Path

import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.checkpoint import custom_objects
from rl.policies.split_policy import SplitStSPointerPolicy


def replace_with_mlp(model, *, seed=0):
    if type(model.policy) is not SplitStSPointerPolicy:
        raise ValueError("combat replacement requires SplitStSPointerPolicy")
    if not (model.nosl_is_az and model.nosl_is_az_inner_only):
        raise ValueError("combat replacement requires inner-only NOSL")
    if model.policy_kwargs.get("combat_arch", "transformer") != "transformer":
        raise ValueError("source combat architecture must be transformer")
    if model.device.type != "cpu":
        raise ValueError("construct the replacement checkpoint on CPU")
    frozen = {key: value for key, value in model.policy.state_dict().items()
              if not key.startswith("core.combat.")}
    kwargs = dict(model.policy_kwargs, combat_arch="mlp")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = SplitStSPointerPolicy(
            model.observation_space, model.action_space, model.lr_schedule, **kwargs,
        )
    missing, unexpected = policy.load_state_dict(frozen, strict=False)
    if unexpected or any(not key.startswith("core.combat.") for key in missing):
        raise ValueError(f"frozen policy contract changed: missing={missing}, unexpected={unexpected}")
    assert all(torch.equal(policy.state_dict()[key], value) for key, value in frozen.items())
    model.policy = policy
    model.policy_kwargs = kwargs
    model.seed = seed
    model.num_timesteps = model._n_updates = model._total_timesteps = 0
    model._num_timesteps_at_start = 0
    model._last_obs = model._last_episode_starts = None
    model._nosl_is_az_replay = None
    model._nosl_is_az_pending_fights = []
    model._nosl_is_az_samples = []
    model._nosl_is_az_fresh_samples = []
    model.configure_nosl_is_az_inner_only()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.suffix != ".zip":
        args.out = Path(str(args.out) + ".zip")
    if args.out.exists():
        raise FileExistsError(args.out)
    model = AccumulatingMaskablePPO.load(
        args.source, device="cpu", custom_objects=custom_objects(args.source),
    )
    old_count = sum(parameter.numel() for parameter in model.policy.core.combat.parameters())
    replace_with_mlp(model, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    new_count = sum(parameter.numel() for parameter in model.policy.core.combat.parameters())
    print(f"saved={args.out} seed={args.seed} step=0 replay=empty Adam=fresh")
    print(f"combat_parameters={old_count}->{new_count} input_width={model.policy.core.combat.input_width}")


if __name__ == "__main__":
    main()
