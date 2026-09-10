"""Create a fresh semantic whole-run checkpoint with an explicit model width.

This is intentionally separate from checkpoint loading: changing ``d_model``
changes parameter shapes, so an existing SB3 checkpoint cannot be reshaped by
``load``.  The source checkpoint is used only to recover the observation and
hierarchical configuration; policy parameters are initialized from scratch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.save_util import load_from_zip_file

from rl.accumulating_ppo import AccumulatingMaskablePPO
from rl import semantic_reasoning_policy as _semantic_reasoning_policy  # noqa: F401
from rl import split_policy as _split_policy  # noqa: F401
from rl import pointer_policy as _pointer_policy  # noqa: F401
from rl import distributed_sb3
from rl.semantic_reasoning_policy import SemanticReasoningPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.d_model < 1:
        raise ValueError("--d-model must be positive")

    # Read only checkpoint metadata.  Loading old policy weights would fail
    # after the reasoning vocabulary changes the decoder output shape.
    source_data, source_params, source_variables = load_from_zip_file(
        args.source, device="cpu"
    )
    source_cfg = dict(source_data.get("policy_kwargs", {}))
    if source_data.get("policy_class") is not SemanticReasoningPolicy:
        raise ValueError("source checkpoint is not a semantic reasoning policy")
    source_space = source_data["observation_space"]

    class EnvArgs:
        pass

    env_args = EnvArgs()
    env_args.ascension = 0
    env_args.global_map_view = distributed_sb3.model_utils.uses_global_map_observation(source_space)
    env_args.card_language = True
    env_args.combat_language = distributed_sb3.model_utils.uses_combat_language(source_space)
    env_args.deck_size_observation = distributed_sb3.model_utils.uses_deck_size_observation(source_space)
    env_args.neow_observation = distributed_sb3.model_utils.uses_neow_observation(source_space)
    env_args.event_option_observation = distributed_sb3.model_utils.uses_event_option_observation(source_space)
    env_args.arch = "semantic-reasoning"
    env_args.interactive_reasoning = True
    env_args.map_dag = bool(source_cfg.get("map_dag", True))

    env = distributed_sb3.WholeRunTask().make_env_fn(env_args)()
    policy_kwargs = {
        "d_model": args.d_model,
        "n_layers": int(
            source_cfg.get("n_layers", 2)
            if args.n_layers is None else args.n_layers
        ),
        "n_heads": int(
            source_cfg.get("n_heads", 4)
            if args.n_heads is None else args.n_heads
        ),
        "card_program_layers": int(source_cfg.get("card_program_layers", 1)),
        "reasoning_layers": int(source_cfg.get("reasoning_layers", 1)),
        "max_reasoning_tokens": source_cfg.get("max_reasoning_tokens"),
        "combat_language": bool(source_cfg.get("combat_language", False)),
        "map_dag": bool(source_cfg.get("map_dag", True)),
    }
    model = AccumulatingMaskablePPO(
        SemanticReasoningPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=args.seed,
        device=args.device,
        amp_mode="off",
        hierarchical_combat_hp=bool(source_data.get("hierarchical_combat_hp", False)),
        combat_hp_gamma=float(source_data.get("combat_hp_gamma", 1.0)),
        combat_hp_gae_lambda=float(source_data.get("combat_hp_gae_lambda", 0.95)),
        reasoning_coef=float(source_data.get("reasoning_coef", 0.2)),
        reasoning_batch_size=int(source_data.get("reasoning_batch_size", 128)),
        reasoning_updates=int(source_data.get("reasoning_updates", 4)),
        verbose=0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    env.close()
    del source_data, source_params, source_variables
    print(f"saved {args.out}")
    print(f"policy_kwargs={policy_kwargs}")
    print(f"parameters={sum(parameter.numel() for parameter in model.policy.parameters()):,}")


if __name__ == "__main__":
    main()
