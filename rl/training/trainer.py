"""Unified MaskablePPO trainer for Slay the Spire.

Two orthogonal axes: the *training* (this file — PPO algorithm, VecEnv,
checkpoints, wandb, resume/init-from, the learn() loop) and the *task* (rl/tasks/
— which env/reward/metrics/policy). Pick a task with --task; every shared PPO
hyperparameter is configurable from the CLI.

Run from repo root:

    python -m rl.training.trainer --task combat
    python -m rl.training.trainer --task combat --ent-coef 0.03 --steps 5000000 --wandb
    python -m rl.training.trainer --task wholerun --arch split-pointer \
        --init-from runs/combat/<ts>/final.zip --freeze-encoder
"""

from __future__ import annotations

import argparse
from env.core import native_actions as C
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
)

from rl.tasks import TASKS
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.training.cached_mask_vec_env import (
    CachedMaskSubprocVecEnv,
    GroupedCachedMaskSubprocVecEnv,
)


class WandbStepMetricCallback(BaseCallback):
    """Expose SB3's TensorBoard global step as an explicit W&B scalar."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {"time/total_timesteps": int(self.num_timesteps)},
                step=int(self.num_timesteps),
            )


class RollingCheckpointCallback(CheckpointCallback):
    """Keep only the newest checkpoints for this callback's name prefix."""

    def __init__(self, *args, max_checkpoints: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1")
        self.max_checkpoints = max_checkpoints

    def _prune_checkpoints(self, limit: int) -> None:
        prefix = f"{self.name_prefix}_"
        suffix = "_steps"
        checkpoints = []
        for path in Path(self.save_path).glob(f"{prefix}*{suffix}.zip"):
            step_text = path.stem[len(prefix):-len(suffix)]
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))

        checkpoints.sort()
        excess = len(checkpoints) - limit
        for _step, path in checkpoints[:max(excess, 0)]:
            path.unlink()

    def _on_step(self) -> bool:
        saving = self.n_calls % self.save_freq == 0
        if saving:
            # Make room before saving so a full disk cannot prevent rotation.
            self._prune_checkpoints(self.max_checkpoints - 1)
        result = super()._on_step()
        if saving:
            self._prune_checkpoints(self.max_checkpoints)
        return result


def _checkpoint_policy_arch(model: AccumulatingMaskablePPO) -> str | None:
    """Return the trainer architecture name represented by a loaded policy."""
    from rl.policies.model import StSMLPExtractor, StSTransformerExtractor
    from rl.policies.pointer_policy import StSPointerPolicy
    from rl.policies.semantic_reasoning_policy import SemanticReasoningPolicy
    from rl.policies.split_policy import CombatStSPointerPolicy, SplitStSPointerPolicy
    from rl.policies.entity_combat import EntityCombatPolicy

    policy = model.policy
    if isinstance(policy, EntityCombatPolicy):
        return "entity-combat"
    if isinstance(policy, SemanticReasoningPolicy):
        return "semantic-reasoning"
    if isinstance(policy, CombatStSPointerPolicy):
        return "combat-pointer"
    if isinstance(policy, SplitStSPointerPolicy):
        return "split-pointer"
    if isinstance(policy, StSPointerPolicy):
        return "transformer-pointer"
    extractor = getattr(policy, "features_extractor", None)
    if isinstance(extractor, StSTransformerExtractor):
        return "transformer-flat"
    if isinstance(extractor, StSMLPExtractor):
        return "mlp-flat"
    return None


def _checkpoint_hierarchical_settings(path) -> tuple[bool, float, float]:
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _pytorch_variables = load_from_zip_file(path, device="cpu")
    settings = (
        bool(data.get("hierarchical_combat_hp", False)),
        float(data.get("combat_hp_gamma", 1.0)),
        float(data.get("combat_hp_gae_lambda", 0.95)),
    )
    del data, _params, _pytorch_variables
    return settings


def _checkpoint_nosl_is_az(path) -> bool:
    """Read the optional combat-search mode from a saved trainer config."""
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _pytorch_variables = load_from_zip_file(path, device="cpu")
    enabled = bool(data.get("nosl_is_az", False))
    del data, _params, _pytorch_variables
    return enabled


def _checkpoint_nosl_is_az_replay_enabled(path) -> bool:
    """Read the replay toggle, defaulting on for legacy checkpoints."""
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _pytorch_variables = load_from_zip_file(path, device="cpu")
    enabled = bool(data.get("nosl_is_az_replay_enabled", True))
    del data, _params, _pytorch_variables
    return enabled


def _checkpoint_nosl_is_az_boss_transition_target(path) -> bool:
    """Read the opt-in boss terminal objective from a checkpoint.

    The field is additive, so checkpoints written before this option existed
    deliberately default to the legacy objective.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _pytorch_variables = load_from_zip_file(path, device="cpu")
    enabled = bool(data.get("nosl_is_az_boss_transition_target", False))
    del data, _params, _pytorch_variables
    return enabled


def _warm_start_policy(
    model,
    pretrained,
    *,
    allow_global_map_extension=False,
    allow_card_language_extension=False,
    allow_combat_language_extension=False,
    allow_map_dag_extension=False,
    allow_deck_size_extension=False,
    allow_neow_extension=True,
    allow_event_option_extension=True,
    skip_combat_mismatch=False,
    allow_combat_contract_migration=False,
):
    """Copy pretrained weights, optionally initializing new semantic params.

    Global-map models append one encoder role. Old checkpoints therefore lack
    only ``map_full`` tokenizer/position parameters and one role-embedding row;
    every pre-existing tensor must still match exactly.
    """
    source = pretrained.policy.state_dict()
    if not any((allow_global_map_extension, allow_card_language_extension,
                allow_combat_language_extension, allow_map_dag_extension,
                allow_deck_size_extension, allow_neow_extension,
                allow_event_option_extension, skip_combat_mismatch, allow_combat_contract_migration)):
        model.policy.load_state_dict(source)
        return 0

    target = model.policy.state_dict()
    incompatible = []
    initialized = []
    source_core = getattr(getattr(pretrained, "policy", None), "core", None)
    target_core = getattr(getattr(model, "policy", None), "core", None)

    def copy_joint_global(key, value):
        if not allow_combat_contract_migration or key != "core.combat.encoder.global_projection.weight":
            return False
        old = source_core.combat.encoder
        new = target_core.combat.encoder
        def segments(encoder):
            widths = [(name, int(np.prod(encoder.observation_space[name].shape)))
                      for name in encoder.global_scalar_keys]
            widths += [(name, encoder.global_categories[name].embedding_dim)
                       for name in encoder.global_category_keys]
            offset, result = 0, {}
            for name, width in widths:
                result[name] = slice(offset, offset + width)
                offset += width
            return result
        before, after = segments(old), segments(new)
        mapped = target[key].detach().clone().zero_()
        for name in before.keys() & after.keys():
            if value[:, before[name]].shape != mapped[:, after[name]].shape:
                raise ValueError("incompatible combat feature migration: " + name)
            mapped[:, after[name]] = value[:, before[name]]
        target[key] = mapped
        initialized.append(key + " (named feature migration)")
        return True

    def _copy_inserted_combat_boundary(key, value):
        """Transfer an old combat global projection with no boundary column.

        The boundary feature is inserted between ``ascension`` and ``gold``.
        Copying by raw prefix would shift every subsequent scalar weight, so
        map segments by their named scalar keys and leave the new column at
        zero (the neutral value for legacy states).
        """
        if key != "core.combat.encoder.global_projection.weight":
            return False
        source_encoder = getattr(source_core, "combat", None)
        source_encoder = getattr(source_encoder, "encoder", None)
        target_encoder = getattr(target_core, "combat", None)
        target_encoder = getattr(target_encoder, "encoder", None)
        if source_encoder is None or target_encoder is None:
            return False
        old_keys = tuple(getattr(source_encoder, "global_scalar_keys", ()))
        new_keys = tuple(getattr(target_encoder, "global_scalar_keys", ()))
        if "post_enemy_turn_boundary" not in new_keys or (
            "post_enemy_turn_boundary" in old_keys
        ):
            return False
        old_space = getattr(source_encoder, "observation_space", None)
        new_space = getattr(target_encoder, "observation_space", None)
        if old_space is None or new_space is None:
            return False

        def width(space, name):
            return int(np.prod(space[name].shape))

        old_offsets = {}
        offset = 0
        for name in old_keys:
            old_offsets[name] = (offset, offset + width(old_space, name))
            offset += width(old_space, name)
        new_offsets = {}
        offset = 0
        for name in new_keys:
            new_offsets[name] = (offset, offset + width(new_space, name))
            offset += width(new_space, name)
        # Category embeddings follow scalar features and have not changed.
        if value.shape[1] > target[key].shape[1]:
            return False
        mapped = target[key].detach().clone()
        for name, (lo, hi) in old_offsets.items():
            nlo, nhi = new_offsets.get(name, (None, None))
            if nlo is None or (hi - lo) != (nhi - nlo):
                return False
            mapped[:, nlo:nhi] = value[:, lo:hi]
        old_category_start = max((hi for _lo, hi in old_offsets.values()), default=0)
        new_category_start = max((hi for _lo, hi in new_offsets.values()), default=0)
        category_width = value.shape[1] - old_category_start
        if category_width != target[key].shape[1] - new_category_start:
            return False
        mapped[:, new_category_start:] = value[:, old_category_start:]
        target[key] = mapped
        initialized.append(f"{key} (inserted post_enemy_turn_boundary column)")
        return True

    for key, value in source.items():
        if copy_joint_global(key, value):
            continue
        if key not in target:
            if allow_combat_contract_migration and key == "core.combat.encoder.global_categories.arg_step.weight":
                initialized.append("retired pending-step board feature")
                continue
            is_old_map = any(marker in key for marker in (
                ".tok.map_full.", ".pos.map_full", ".map_dag.", ".map_dag"
            ))
            if allow_global_map_extension and (is_old_map or allow_map_dag_extension):
                initialized.append(f"skipped source map parameter {key}")
                continue
            incompatible.append(f"unexpected source parameter {key}")
            continue
        if value.shape == target[key].shape:
            target[key] = value
            continue
        if _copy_inserted_combat_boundary(key, value):
            continue
        is_appended_role = (
            key.endswith("role_emb.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] >= value.shape[0]
            and target[key].shape[1] == value.shape[1]
        )
        is_appended_global_scalar = (
            allow_deck_size_extension
            and key.endswith("tok.global.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[1] == value.shape[1] + 1
        )
        is_appended_neow_role = (
            allow_neow_extension
            and key.endswith("role_emb.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] == value.shape[0] + 1
            and target[key].shape[1] == value.shape[1]
        )
        if is_appended_role:
            target[key][:value.shape[0]] = value
            initialized.append(f"{key}[{value.shape[0]}:]")
        elif is_appended_neow_role:
            target[key][:value.shape[0]] = value
            initialized.append(f"{key}[-1]")
        elif is_appended_global_scalar:
            target[key][:, :value.shape[1]] = value
            initialized.append(f"{key}[:, -1]")
        else:
            if skip_combat_mismatch and (
                key.startswith("core.combat.") or ".core.combat." in key
            ):
                initialized.append(f"skipped incompatible combat parameter {key}")
                continue
            incompatible.append(
                f"shape mismatch for {key}: checkpoint {tuple(value.shape)}, "
                f"target {tuple(target[key].shape)}"
            )

    for key in target.keys() - source.keys():
        is_global_map = ".map_full." in key or key.endswith(".map_full")
        is_map_dag = ".map_dag." in key or key.endswith(".map_dag")
        is_card_language = any(marker in key for marker in (
            ".shared.card.token_embedding.",
            ".shared.card.local_encoder.",
            ".shared.card.program_position",
            ".shared.card.effect_scale",
            ".pile_card_language.",
            ".exhaust_card_language.",
        ))
        is_combat_language = ".combat_language" in key
        is_neow = any(marker in key for marker in (
            ".shared.neow_bonus.",
            ".shared.neow_drawback.",
            ".tok.neow.",
            ".pos.neow",
        ))
        is_event_option = any(marker in key for marker in (
            ".shared.event_option.",
            ".tok.event_option.",
            ".pos.event_option",
        ))
        is_joint = allow_combat_contract_migration and (
            key.startswith("core.combat.complete_head.")
            or key.startswith("core.combat.encoder.choice_")
            or key == "core.combat.encoder.global_categories.select_zone.weight")
        if (
            is_joint
        ) or (
            allow_global_map_extension and is_global_map
        ) or (
            allow_map_dag_extension and is_map_dag
        ) or (
            allow_card_language_extension and is_card_language
        ) or (
            allow_combat_language_extension and is_combat_language
        ) or (
            allow_neow_extension and is_neow
        ) or (
            allow_event_option_extension and is_event_option
        ):
            initialized.append(key)
        else:
            incompatible.append(f"missing source parameter {key}")
    if incompatible:
        details = "\n  - ".join(incompatible)
        raise ValueError(
            "--init-from is not compatible with the requested policy:\n  - "
            + details
        )
    model.policy.load_state_dict(target)
    return len(initialized)


def build_parser(task) -> argparse.ArgumentParser:
    """Full parser for a chosen task: shared PPO/infra args + the task's own
    args, with the task's preferred defaults applied last so an explicit CLI
    flag still wins over them."""
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@",
        description=f"Train MaskablePPO ({task.name} task).",
    )
    # --task is consumed in the pre-parse (see main); declared here so it shows
    # in --help and survives the final parse.
    parser.add_argument("--task", choices=sorted(TASKS), required=True)

    # --- absolute step budget --------------------------------------------------
    parser.add_argument(
        "--steps",
        type=int,
        help="Absolute total-steps target: training runs until num_timesteps "
             "reaches this. On --resume, the already-trained steps count toward "
             "it (so resuming a 5M ckpt with --steps 10000000 trains 5M more).",
    )

    # --- rollout / parallelism -------------------------------------------------
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of PPO micro-batches to average before one optimizer "
             "step. Effective batch size is --batch-size times this value; "
             "GPU peak memory is governed by --batch-size.",
    )
    parser.add_argument(
        "--vec",
        choices=["subproc", "dummy"],
        default="subproc",
        help="VecEnv backend. subproc = multiple worker processes (true "
             "multi-core sampling; default). dummy = all envs in this process, serial "
             "(single core — use only for debugging).",
    )

    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=1,
        help="Independent environments/search trees hosted by one subprocess. "
             "1 preserves one-env-per-worker; values above 1 reduce IPC "
             "overhead (for example 4 with 24 envs uses 6 workers).",
    )

    # --- PPO hyperparameters (all CLI-configurable) ----------------------------
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.999,
        help="Discount. <1 so non-terminating infinite combos (0 per-step "
             "reward) discount to value 0; 0.999 keeps act1-boss credit ~0.71 "
             "(~20 steps/floor), while whole-run reward puts most progress "
             "credit on boss clears.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy bonus coefficient. Raise to slow premature convergence / "
             "encourage exploration.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)

    # --- policy geometry -------------------------------------------------------
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--card-language",
        action="store_true",
        help="Enrich Ironclad card ID embeddings with tokenized base/upgraded "
             "effect programs. Off by default for checkpoint compatibility.",
    )
    parser.add_argument(
        "--combat-language",
        action="store_true",
        help="Tokenize every monster intent and each live entity status bar "
             "(HP, max HP, Block, powers). Off for checkpoint compatibility.",
    )

    # --- run / io --------------------------------------------------------------
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="PyTorch float32 matmul precision. 'high' enables TF32 tensor cores "
             "on supported NVIDIA GPUs; use 'highest' for strict FP32.",
    )
    parser.add_argument(
        "--amp-mode",
        choices=("off", "bf16"),
        default="off",
        help="Optional CUDA autocast for policy inference and PPO backward. "
             "BF16 keeps parameters and accumulated gradients in FP32.",
    )
    parser.add_argument(
        "--out",
        help="Base directory; each run lands in <out>/<timestamp>/ "
             "(checkpoints, final model, optional TensorBoard logs).",
    )
    parser.add_argument(
        "--save-every", type=int, default=100_000,
        help="Save a checkpoint about every N environment transitions.",
    )
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--profile-performance",
        action="store_true",
        help="Measure GPU policy, CPU env, action-mask IPC, rollout overhead, "
             "update time, and peak GPU memory. CUDA synchronization makes "
             "these timings accurate but adds profiling overhead.",
    )
    parser.add_argument(
        "--metrics-window", type=int,
        help="Rolling window (episodes) for game/* and rollout/* metrics.",
    )

    # --- resume / warm-start ---------------------------------------------------
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a .zip checkpoint to continue training from. Network "
             "geometry is restored from the checkpoint and must not be changed; "
             "--learning-rate still applies. num_timesteps/TensorBoard step/"
             "checkpoint counters continue from where the checkpoint left off.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Path to a .zip checkpoint to warm-start the POLICY WEIGHTS from "
             "(e.g. a combat-pretrained model) WITHOUT inheriting its step "
             "counter or optimizer state. The arch must match. Unlike --resume, "
             "training starts fresh at step 0.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the shared semantic tokenizers and combat encoder. The "
             "combat/overworld heads and overworld encoder remain trainable, "
             "so combat pretraining can be retained while the new overworld "
             "specialist learns.",
    )

    # --- wandb -----------------------------------------------------------------
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-grad-freq", type=int, default=0)
    parser.add_argument("--wandb-save-freq", type=int, default=50_000)

    # Task-specific args, then task defaults (so explicit flags still win).
    task.add_args(parser)
    parser.set_defaults(**task.defaults)
    return parser


def main() -> None:
    # Pre-parse only --task so we can build the task-aware full parser.
    pre = argparse.ArgumentParser(fromfile_prefix_chars="@", add_help=False)
    pre.add_argument("--task", choices=sorted(TASKS))
    known, _ = pre.parse_known_args()
    if known.task is None:
        # Defer to the full parser for a proper error + usage message.
        build_parser(next(iter(TASKS.values()))).parse_args()
        return
    task = TASKS[known.task]

    args = build_parser(task).parse_args()
    th.set_float32_matmul_precision(args.float32_matmul_precision)
    if hasattr(args, "nosl_is_az_upper_strategy"):
        resume_settings = {}
        if args.resume:
            from stable_baselines3.common.save_util import load_from_zip_file
            resume_settings, _, _ = load_from_zip_file(args.resume, device="cpu")
            from rl.policies.entity_combat import EntityCombatPolicy
            if resume_settings.get("policy_class") is EntityCombatPolicy and args.arch == "auto":
                args.arch = "entity-combat"
                args.nosl_is_az_inner_only = resume_settings["nosl_is_az_inner_only"]
                args.nosl_is_az_cpp = resume_settings["nosl_is_az_cpp"]
        if args.nosl_is_az_upper_strategy is None:
            args.nosl_is_az_upper_strategy = resume_settings.get("nosl_is_az_upper_strategy", "model")
        if args.arch == "entity-combat":
            args.card_language = True
            if args.combat_action_contract is None and not args.resume:
                args.combat_action_contract = C.AUTOREGRESSIVE_CONTRACT
            if args.combat_action_contract not in (*C.NEURAL_CONTRACTS, None) or not (
                    (args.nosl_is_az or resume_settings.get("nosl_is_az"))
                    and args.nosl_is_az_inner_only and args.nosl_is_az_cpp
                    and args.nosl_is_az_upper_strategy == "rules"):
                raise ValueError("entity-combat requires joint-v2, C++ inner-only NOSL and rules upper strategy")
            if args.interactive_reasoning or args.card_vector_checkpoint:
                raise ValueError("entity-combat has its own card encoder and no history/reasoning decoder")

    source_hierarchical = False
    source_combat_gamma = 1.0
    source_combat_gae_lambda = 0.95
    if args.resume and hasattr(args, "hierarchical_combat_hp"):
        (
            source_hierarchical,
            source_combat_gamma,
            source_combat_gae_lambda,
        ) = _checkpoint_hierarchical_settings(args.resume)
    if hasattr(args, "hierarchical_combat_hp"):
        if args.hierarchical_combat_hp is None:
            args.hierarchical_combat_hp = (
                source_hierarchical if args.resume else False
            )
        if args.combat_hp_gamma is None:
            args.combat_hp_gamma = (
                source_combat_gamma if args.resume else 1.0
            )
        if args.combat_hp_gae_lambda is None:
            args.combat_hp_gae_lambda = (
                source_combat_gae_lambda if args.resume else 0.95
            )
        if not 0.0 <= args.combat_hp_gamma <= 1.0:
            raise ValueError("--combat-hp-gamma must be in [0, 1]")
        if not 0.0 <= args.combat_hp_gae_lambda <= 1.0:
            raise ValueError("--combat-hp-gae-lambda must be in [0, 1]")

    if hasattr(args, "nosl_is_az") and args.nosl_is_az is None:
        args.nosl_is_az = (
            _checkpoint_nosl_is_az(args.resume) if args.resume else False
        )

    if hasattr(args, "nosl_is_az_replay_enabled"):
        if args.nosl_is_az_replay_enabled is None:
            args.nosl_is_az_replay_enabled = (
                _checkpoint_nosl_is_az_replay_enabled(args.resume)
                if args.resume and getattr(args, "nosl_is_az", False)
                else True
            )

    if hasattr(args, "nosl_is_az_boss_transition_target"):
        if args.nosl_is_az_boss_transition_target is None:
            args.nosl_is_az_boss_transition_target = (
                _checkpoint_nosl_is_az_boss_transition_target(args.resume)
                if args.resume else False
            )

    if getattr(args, "nosl_is_az", False):
        if args.card_language and args.arch != "entity-combat":
            raise ValueError(
                "--nosl-is-az requires card_language=false so ordered hidden "
                "draw-pile information is not present in the policy observation"
            )
        if hasattr(args, "hierarchical_combat_hp") and not args.hierarchical_combat_hp:
            print(
                "[trainer] enabled hierarchical combat returns because "
                "--nosl-is-az replaces the combat PPO objective"
            )
            args.hierarchical_combat_hp = True
        for name in (
            "nosl_is_az_simulations",
            "nosl_is_az_batch_size",
            "nosl_is_az_micro_batch_size",
            "nosl_is_az_replay_capacity",
            "nosl_is_az_replay_min_size",
            "nosl_is_az_leaf_batch_size",
            "nosl_is_az_inference_batch_size",
            "nosl_is_az_updates",
        ):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if (
            not np.isfinite(args.nosl_is_az_inference_max_wait_ms)
            or args.nosl_is_az_inference_max_wait_ms < 0.0
        ):
            raise ValueError(
                "--nosl-is-az-inference-max-wait-ms must be finite and non-negative"
            )
        if args.nosl_is_az_inference_schedule not in ("fixed_wait", "adaptive"):
            raise ValueError(
                "--nosl-is-az-inference-schedule must be fixed_wait or adaptive"
            )
        if args.nosl_is_az_replay_min_size > args.nosl_is_az_replay_capacity:
            raise ValueError(
                "--nosl-is-az-replay-min-size cannot exceed replay capacity"
            )
        if (
            not np.isfinite(args.nosl_is_az_replay_max_old_fraction)
            or not 0.0 <= args.nosl_is_az_replay_max_old_fraction <= 1.0
        ):
            raise ValueError(
                "--nosl-is-az-replay-max-old-fraction must be in [0, 1]"
            )
        if args.nosl_is_az_depth < 0:
            raise ValueError("--nosl-is-az-depth must be non-negative")
        if args.nosl_is_az_temperature < 0.0 or args.nosl_is_az_c_puct < 0.0:
            raise ValueError("NOSL-IS-AZ c_puct and temperature must be non-negative")
        if not np.isfinite(args.nosl_is_az_prior_temperature) or args.nosl_is_az_prior_temperature <= 0.0:
            raise ValueError("--nosl-is-az-prior-temperature must be positive and finite")
        if args.nosl_is_az_gumbel_candidates < 0:
            raise ValueError("--nosl-is-az-gumbel-candidates must be non-negative")
        for name in (
            "nosl_is_az_gumbel_scale",
            "nosl_is_az_gumbel_value_scale",
            "nosl_is_az_gumbel_max_visit_init",
        ):
            value = getattr(args, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be non-negative and finite"
                )
        if not 0.0 <= args.nosl_is_az_gumbel_interior_uniformity <= 1.0:
            raise ValueError(
                "--nosl-is-az-gumbel-interior-uniformity must be in [0, 1]"
            )
        if not 0.0 <= args.nosl_is_az_gumbel_interior_coverage_fraction <= 1.0:
            raise ValueError(
                "--nosl-is-az-gumbel-interior-coverage-fraction must be in [0, 1]"
            )
        if args.nosl_is_az_gumbel_interior_value_mode not in (0, 1, 2):
            raise ValueError(
                "--nosl-is-az-gumbel-interior-value-mode must be 0, 1, or 2"
            )
        if (
            not np.isfinite(args.nosl_is_az_gumbel_interior_max_mix_visits)
            or args.nosl_is_az_gumbel_interior_max_mix_visits <= 0.0
        ):
            raise ValueError(
                "--nosl-is-az-gumbel-interior-max-mix-visits must be positive and finite"
            )
        if args.nosl_is_az_search_mode != "puct" and not args.nosl_is_az_cpp:
            raise ValueError("Gumbel NOSL-IS-AZ modes require --nosl-is-az-cpp")
        if getattr(args, "nosl_is_az_paper_full_gumbel", False):
            if args.nosl_is_az_search_mode != "full_gumbel":
                raise ValueError(
                    "--nosl-is-az-paper-full-gumbel requires "
                    "--nosl-is-az-search-mode full_gumbel"
                )
            if not args.nosl_is_az_cpp:
                raise ValueError(
                    "--nosl-is-az-paper-full-gumbel requires --nosl-is-az-cpp"
                )
        if args.nosl_is_az_dirichlet_alpha < 0.0:
            raise ValueError("--nosl-is-az-dirichlet-alpha must be non-negative")
        if not 0.0 <= args.nosl_is_az_dirichlet_epsilon <= 1.0:
            raise ValueError("--nosl-is-az-dirichlet-epsilon must be in [0, 1]")
        if args.nosl_is_az_dirichlet_epsilon > 0.0 and args.nosl_is_az_dirichlet_alpha <= 0.0:
            raise ValueError("Dirichlet alpha must be positive when epsilon > 0")
        if args.nosl_is_az_rollout_steps < 0:
            raise ValueError("--nosl-is-az-rollout-steps must be non-negative")
    if getattr(args, "nosl_is_az_cpp", False) and not getattr(
        args, "nosl_is_az", False
    ):
        raise ValueError("--nosl-is-az-cpp requires --nosl-is-az")
    if getattr(args, "nosl_is_az_inner_only", False):
        if not getattr(args, "nosl_is_az", False):
            raise ValueError("--nosl-is-az-inner-only requires --nosl-is-az")
        if not (args.init_from or args.resume) and args.arch != "entity-combat":
            raise ValueError(
                "--nosl-is-az-inner-only requires --init-from or --resume "
                "to provide the frozen outer policy"
            )
        if args.freeze_encoder:
            raise ValueError(
                "--nosl-is-az-inner-only and --freeze-encoder are mutually exclusive"
            )
    if getattr(args, "nosl_is_az_reset_combat", False):
        if not getattr(args, "nosl_is_az_inner_only", False):
            raise ValueError(
                "--nosl-is-az-reset-combat requires --nosl-is-az-inner-only"
            )
        if not args.init_from:
            raise ValueError("--nosl-is-az-reset-combat requires --init-from")

    # Whole-run training uses the split specialist architecture. When warm-
    # starting, a combat-only specialist is copied into the split model's
    # shared tokenizer and combat branch. Keep the loaded object for transfer.
    pretrained = None
    if args.init_from:
        if args.resume:
            raise ValueError(
                "--init-from and --resume are mutually exclusive: --resume "
                "already restores weights + step counter."
            )
        pretrained = AccumulatingMaskablePPO.load(
            args.init_from, device=args.device
        )
        if hasattr(args, "hierarchical_combat_hp"):
            source_hierarchical = bool(
                getattr(pretrained, "hierarchical_combat_hp", False)
            )

    if hasattr(args, "global_map_view"):
        from rl.policies.model import uses_global_map_observation

        source_has_global_map = (
            pretrained is not None
            and uses_global_map_observation(pretrained)
        )
        if args.resume and not args.global_map_view:
            from stable_baselines3.common.save_util import load_from_zip_file

            resume_data, _params, _pytorch_variables = load_from_zip_file(
                args.resume, device="cpu"
            )
            source_has_global_map = uses_global_map_observation(
                resume_data["observation_space"]
            )
            del resume_data, _params, _pytorch_variables
        if source_has_global_map and not args.global_map_view:
            args.global_map_view = True
            source = args.resume or args.init_from
            print(f"[trainer] enabled --global-map-view from {source}")

    if hasattr(args, "deck_size_observation"):
        from rl.policies.model import uses_deck_size_observation

        if args.resume:
            from stable_baselines3.common.save_util import load_from_zip_file

            resume_data, _params, _pytorch_variables = load_from_zip_file(
                args.resume, device="cpu"
            )
            args.deck_size_observation = uses_deck_size_observation(
                resume_data["observation_space"]
            )
            del resume_data, _params, _pytorch_variables

    if hasattr(args, "neow_observation"):
        from rl.policies.model import uses_neow_observation

        if args.resume:
            from stable_baselines3.common.save_util import load_from_zip_file

            resume_data, _params, _pytorch_variables = load_from_zip_file(
                args.resume, device="cpu"
            )
            args.neow_observation = uses_neow_observation(
                resume_data["observation_space"]
            )
            del resume_data, _params, _pytorch_variables

    if hasattr(args, "event_option_observation"):
        from rl.policies.model import uses_event_option_observation

        if args.resume:
            from stable_baselines3.common.save_util import load_from_zip_file

            resume_data, _params, _pytorch_variables = load_from_zip_file(
                args.resume, device="cpu"
            )
            args.event_option_observation = uses_event_option_observation(
                resume_data["observation_space"]
            )
            del resume_data, _params, _pytorch_variables

    if pretrained is not None:
        from rl.policies.model import uses_card_language, uses_combat_language

        if uses_card_language(pretrained) and not args.card_language:
            args.card_language = True
            print(f"[trainer] enabled --card-language from {args.init_from}")
        if uses_combat_language(pretrained) and not args.combat_language:
            args.combat_language = True
            print(f"[trainer] enabled --combat-language from {args.init_from}")

    if args.resume and (not args.card_language or not args.combat_language):
        from stable_baselines3.common.save_util import load_from_zip_file
        from rl.policies.model import uses_card_language, uses_combat_language

        resume_data, _params, _pytorch_variables = load_from_zip_file(
            args.resume, device="cpu"
        )
        resume_space = resume_data["observation_space"]
        if uses_card_language(resume_space) and not args.card_language:
            args.card_language = True
            print(f"[trainer] enabled --card-language from {args.resume}")
        if uses_combat_language(resume_space) and not args.combat_language:
            args.combat_language = True
            print(f"[trainer] enabled --combat-language from {args.resume}")
        del resume_data, _params, _pytorch_variables

    if hasattr(args, "arch") and args.arch == "auto":
        if pretrained is None:
            args.arch = "split-pointer"
            print("[trainer] selected split-pointer")
        else:
            source_arch = _checkpoint_policy_arch(pretrained)
            if source_arch is None:
                raise ValueError(
                    "Could not infer the policy architecture in --init-from. "
                    "Pass --arch explicitly."
                )
            args.arch = (
                "split-pointer" if source_arch == "combat-pointer" else source_arch
            )
            print(
                f"[trainer] inferred --arch {args.arch} from {args.init_from}"
            )
    if getattr(args, "nosl_is_az", False) and args.card_language and args.arch != "entity-combat":
        raise ValueError(
            "--nosl-is-az requires card_language=false except for the "
            "order-invariant entity-combat encoder"
        )

    if getattr(args, "nosl_is_az", False) and args.arch == "semantic-reasoning":
        raise ValueError(
            "NOSL-IS-AZ currently requires split-pointer observations; the "
            "semantic-reasoning policy exposes ordered card-language fields"
        )

    if hasattr(args, "combat_action_contract"):
        source_space = pretrained.observation_space if pretrained is not None else None
        if args.resume:
            from stable_baselines3.common.save_util import load_from_zip_file
            data, params, variables = load_from_zip_file(args.resume, device="cpu")
            source_space = data["observation_space"]
            del data, params, variables
        source_contract = C.policy_contract(source_space)
        args.combat_action_contract = args.combat_action_contract or source_contract
        if args.resume and args.combat_action_contract != source_contract:
            raise ValueError("combat action migration requires --init-from, not --resume")
        if args.combat_action_contract == C.AUTOREGRESSIVE_CONTRACT and args.arch != 'entity-combat':
            raise ValueError('autoregressive-v1 requires entity-combat')
        if args.combat_action_contract in C.NEURAL_CONTRACTS:
            if args.arch not in ("split-pointer", "entity-combat") or getattr(args, "combat_arch", "transformer") != "transformer":
                raise ValueError("joint-v2 requires split-pointer Transformer or entity-combat")
            if getattr(args, "nosl_is_az", False) and not args.nosl_is_az_cpp:
                raise ValueError("joint-v2 NOSL requires --nosl-is-az-cpp")

    if pretrained is not None and hasattr(args, "arch"):
        source_arch = _checkpoint_policy_arch(pretrained)
        combat_into_split = (
            source_arch == "combat-pointer" and args.arch == "split-pointer"
        )
        if source_arch is not None and source_arch != args.arch and not combat_into_split:
            raise ValueError(
                f"--init-from uses {source_arch}, but the requested whole-run "
                f"policy uses {args.arch}. Omit --arch to infer it, or pass "
                f"--arch {source_arch} to transfer the complete policy."
            )

    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.envs_per_worker < 1:
        raise ValueError("--envs-per-worker must be at least 1")

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            "batch_size must divide n_envs * n_steps: "
            f"{args.batch_size} does not divide {rollout_size}"
        )
    micro_batches = rollout_size // args.batch_size
    if micro_batches % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "n_envs * n_steps / batch_size must be divisible by "
            "gradient_accumulation_steps: "
            f"{micro_batches} micro-batches do not divide into groups of "
            f"{args.gradient_accumulation_steps}"
        )

    # Unique run id (start timestamp) so each run gets its own out_dir and wandb
    # name — concurrent/repeated runs never clobber each other.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    checkpoint_dir = (
        Path(args.resume).expanduser().resolve().parent
        if args.resume
        else out_dir / "checkpoints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [task.make_env_fn(args) for _ in range(args.n_envs)]
    if args.vec == "subproc":
        # SB3 selects forkserver where available and spawn on Windows. Both
        # avoid forking a process that already imported the native sim and
        # torch, which can otherwise inherit unsafe thread/CUDA state.
        if args.envs_per_worker == 1:
            venv = CachedMaskSubprocVecEnv(env_fns)
        else:
            venv = GroupedCachedMaskSubprocVecEnv(
                env_fns,
                envs_per_worker=args.envs_per_worker,
            )
        print(
            f"[trainer] subprocess workers={len(venv.remotes)} "
            f"logical_envs={args.n_envs} envs_per_worker={args.envs_per_worker}"
        )
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=str(out_dir / "monitor.csv"))
    venv.seed(args.seed)

    # Callback runs once per vectorized step, so adjust for n_envs.
    checkpoint_callback = RollingCheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=task.ckpt_prefix,
        max_checkpoints=20,
    )

    tag = task.tag(args, run_id)
    callbacks = [checkpoint_callback, task.metrics_callback(args)]
    wandb_run = None

    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        config = vars(args).copy()
        config["tag"] = tag
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or tag,
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )
        wandb.define_metric("time/total_timesteps")
        callbacks.append(
            WandbCallback(
                gradient_save_freq=args.wandb_grad_freq,
                model_save_path=str(out_dir / "wandb_models" / wandb_run.id),
                model_save_freq=args.wandb_save_freq,
                verbose=2,
            )
        )
        callbacks.append(WandbStepMetricCallback())

    callback = CallbackList(callbacks)

    # sync_tensorboard needs SB3 to actually emit TensorBoard logs, so enable the
    # TensorBoard sink whenever either --tensorboard or --wandb is set.
    tensorboard_log = (
        str(out_dir / "tensorboard") if (args.tensorboard or args.wandb) else None
    )

    random_combat_state = None
    if not args.resume:
        policy, policy_kwargs = task.build_policy(args)
        model = AccumulatingMaskablePPO(
            policy=policy,
            env=venv,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            performance_profile=args.profile_performance,
            amp_mode=args.amp_mode,
            hierarchical_combat_hp=getattr(
                args, "hierarchical_combat_hp", False
            ),
            nosl_is_az_boss_transition_target=getattr(
                args, "nosl_is_az_boss_transition_target", False
            ),
            combat_hp_gamma=getattr(args, "combat_hp_gamma", 1.0),
            combat_hp_gae_lambda=getattr(
                args, "combat_hp_gae_lambda", 0.95
            ),
            reasoning_coef=(
                getattr(args, "reasoning_coef", 0.0)
                if getattr(args, "arch", None) == "semantic-reasoning"
                else 0.0
            ),
            reasoning_batch_size=getattr(args, "reasoning_batch_size", 16),
            reasoning_updates=getattr(args, "reasoning_updates", 1),
            nosl_is_az=getattr(args, "nosl_is_az", False),
            nosl_is_az_replay_enabled=getattr(
                args, "nosl_is_az_replay_enabled", True
            ),
            nosl_is_az_inner_only=getattr(
                args, "nosl_is_az_inner_only", False
            ),
            nosl_is_az_cpp=getattr(args, "nosl_is_az_cpp", False),
            nosl_is_az_upper_strategy=getattr(args, "nosl_is_az_upper_strategy", "model"),
            nosl_is_az_simulations=getattr(args, "nosl_is_az_simulations", 8),
            nosl_is_az_depth=getattr(args, "nosl_is_az_depth", 3),
            nosl_is_az_end_turn_as_leaf=getattr(
                args, "nosl_is_az_end_turn_as_leaf", False
            ),
            nosl_is_az_rollout_steps=getattr(
                args, "nosl_is_az_rollout_steps", 0
            ),
            nosl_is_az_c_puct=getattr(args, "nosl_is_az_c_puct", 1.5),
            nosl_is_az_temperature=getattr(
                args, "nosl_is_az_temperature", 1.0
            ),
            nosl_is_az_prior_temperature=getattr(
                args, "nosl_is_az_prior_temperature", 1.0
            ),
            nosl_is_az_search_mode=getattr(
                args, "nosl_is_az_search_mode", "puct"
            ),
            nosl_is_az_paper_full_gumbel=getattr(
                args, "nosl_is_az_paper_full_gumbel", False
            ),
            nosl_is_az_gumbel_candidates=getattr(
                args, "nosl_is_az_gumbel_candidates", 0
            ),
            nosl_is_az_gumbel_scale=getattr(
                args, "nosl_is_az_gumbel_scale", 1.0
            ),
            nosl_is_az_gumbel_value_scale=getattr(
                args, "nosl_is_az_gumbel_value_scale", 0.1
            ),
            nosl_is_az_gumbel_max_visit_init=getattr(
                args, "nosl_is_az_gumbel_max_visit_init", 50.0
            ),
            nosl_is_az_gumbel_rescale_q=getattr(
                args, "nosl_is_az_gumbel_rescale_q", True
            ),
            nosl_is_az_gumbel_use_mixed_value=getattr(
                args, "nosl_is_az_gumbel_use_mixed_value", True
            ),
            nosl_is_az_gumbel_interior_uniformity=getattr(
                args, "nosl_is_az_gumbel_interior_uniformity", 0.0
            ),
            nosl_is_az_gumbel_interior_coverage_fraction=getattr(
                args, "nosl_is_az_gumbel_interior_coverage_fraction", 0.0
            ),
            nosl_is_az_gumbel_interior_value_mode=getattr(
                args, "nosl_is_az_gumbel_interior_value_mode", 2
            ),
            nosl_is_az_gumbel_interior_max_mix_visits=getattr(
                args, "nosl_is_az_gumbel_interior_max_mix_visits", 64.0
            ),
            nosl_is_az_dirichlet_alpha=getattr(
                args, "nosl_is_az_dirichlet_alpha", 0.0
            ),
            nosl_is_az_dirichlet_epsilon=getattr(
                args, "nosl_is_az_dirichlet_epsilon", 0.0
            ),
            nosl_is_az_master_nonce=getattr(
                args, "nosl_is_az_master_nonce", 0x4E4F534C4953415A
            ),
            nosl_is_az_batch_size=getattr(args, "nosl_is_az_batch_size", 64),
            nosl_is_az_micro_batch_size=getattr(
                args, "nosl_is_az_micro_batch_size", 2048
            ),
            nosl_is_az_replay_capacity=getattr(
                args, "nosl_is_az_replay_capacity", 131072
            ),
            nosl_is_az_replay_min_size=getattr(
                args, "nosl_is_az_replay_min_size", 1
            ),
            nosl_is_az_replay_max_old_fraction=getattr(
                args, "nosl_is_az_replay_max_old_fraction", 1.0
            ),
            nosl_is_az_leaf_batch_size=getattr(
                args, "nosl_is_az_leaf_batch_size", 64
            ),
            nosl_is_az_inference_batch_size=getattr(
                args, "nosl_is_az_inference_batch_size", 32
            ),
            nosl_is_az_inference_max_wait_ms=getattr(
                args, "nosl_is_az_inference_max_wait_ms", 0.5
            ),
            nosl_is_az_inference_schedule=getattr(
                args, "nosl_is_az_inference_schedule", "fixed_wait"
            ),
            nosl_is_az_updates=getattr(args, "nosl_is_az_updates", 1),
            nosl_is_az_update_schedule=getattr(args, "nosl_is_az_update_schedule", None),
            nosl_is_az_policy_coef=getattr(
                args, "nosl_is_az_policy_coef", 1.0
            ),
            nosl_is_az_value_coef=getattr(args, "nosl_is_az_value_coef", 0.5),
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )
        if getattr(args, "nosl_is_az_reset_combat", False):
            combat = getattr(getattr(model.policy, "core", None), "combat", None)
            if combat is None:
                raise ValueError(
                    "--nosl-is-az-reset-combat requires a split specialist policy"
                )
            random_combat_state = {
                key: value.detach().clone()
                for key, value in combat.state_dict().items()
            }
    else:
        # Resume: restore weights + optimizer state (continuous Adam momentum /
        # lr schedule) from the checkpoint. Network geometry and most hyper-
        # params come from the zip; env + device + tensorboard sink are rebound
        # here, and the CLI-tunable PPO knobs override via custom_objects
        # (everything else in the .zip wins).
        model = AccumulatingMaskablePPO.load(
            args.resume,
            env=venv,
            device=args.device,
            tensorboard_log=tensorboard_log,
            amp_mode=args.amp_mode,
            custom_objects={
                "learning_rate": args.learning_rate,
                "ent_coef": args.ent_coef,
                "clip_range": args.clip_range,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "vf_coef": args.vf_coef,
                "n_epochs": args.n_epochs,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "performance_profile": args.profile_performance,
                "hierarchical_combat_hp": getattr(
                    args, "hierarchical_combat_hp", False
                ),
                "nosl_is_az_boss_transition_target": getattr(
                    args, "nosl_is_az_boss_transition_target", False
                ),
                "combat_hp_gamma": getattr(args, "combat_hp_gamma", 1.0),
                "combat_hp_gae_lambda": getattr(
                    args, "combat_hp_gae_lambda", 0.95
                ),
                "reasoning_coef": (
                    getattr(args, "reasoning_coef", 0.0)
                    if getattr(args, "arch", None) == "semantic-reasoning"
                    else 0.0
                ),
                "reasoning_batch_size": getattr(
                    args, "reasoning_batch_size", 16
                ),
                "reasoning_updates": getattr(args, "reasoning_updates", 1),
                "nosl_is_az": getattr(args, "nosl_is_az", False),
                "nosl_is_az_replay_enabled": getattr(
                    args, "nosl_is_az_replay_enabled", True
                ),
                "nosl_is_az_inner_only": getattr(
                    args, "nosl_is_az_inner_only", False
                ),
                "nosl_is_az_cpp": getattr(args, "nosl_is_az_cpp", False),
                "nosl_is_az_upper_strategy": getattr(args, "nosl_is_az_upper_strategy", "model"),
                "nosl_is_az_simulations": getattr(
                    args, "nosl_is_az_simulations", 8
                ),
                "nosl_is_az_depth": getattr(args, "nosl_is_az_depth", 3),
                "nosl_is_az_end_turn_as_leaf": getattr(
                    args, "nosl_is_az_end_turn_as_leaf", False
                ),
                "nosl_is_az_rollout_steps": getattr(
                    args, "nosl_is_az_rollout_steps", 0
                ),
                "nosl_is_az_c_puct": getattr(args, "nosl_is_az_c_puct", 1.5),
                "nosl_is_az_temperature": getattr(
                    args, "nosl_is_az_temperature", 1.0
                ),
                "nosl_is_az_prior_temperature": getattr(
                    args, "nosl_is_az_prior_temperature", 1.0
                ),
                "nosl_is_az_search_mode": getattr(
                    args, "nosl_is_az_search_mode", "puct"
                ),
                "nosl_is_az_paper_full_gumbel": getattr(
                    args, "nosl_is_az_paper_full_gumbel", False
                ),
                "nosl_is_az_gumbel_candidates": getattr(
                    args, "nosl_is_az_gumbel_candidates", 0
                ),
                "nosl_is_az_gumbel_scale": getattr(
                    args, "nosl_is_az_gumbel_scale", 1.0
                ),
                    "nosl_is_az_gumbel_value_scale": getattr(
                    args, "nosl_is_az_gumbel_value_scale", 0.1
                ),
                "nosl_is_az_gumbel_max_visit_init": getattr(
                    args, "nosl_is_az_gumbel_max_visit_init", 50.0
                ),
                "nosl_is_az_gumbel_rescale_q": getattr(
                    args, "nosl_is_az_gumbel_rescale_q", True
                ),
                "nosl_is_az_gumbel_use_mixed_value": getattr(
                    args, "nosl_is_az_gumbel_use_mixed_value", True
                ),
                "nosl_is_az_gumbel_interior_uniformity": getattr(
                    args, "nosl_is_az_gumbel_interior_uniformity", 0.0
                ),
                "nosl_is_az_gumbel_interior_coverage_fraction": getattr(
                    args, "nosl_is_az_gumbel_interior_coverage_fraction", 0.0
                ),
                "nosl_is_az_gumbel_interior_value_mode": getattr(
                    args, "nosl_is_az_gumbel_interior_value_mode", 2
                ),
                "nosl_is_az_gumbel_interior_max_mix_visits": getattr(
                    args, "nosl_is_az_gumbel_interior_max_mix_visits", 64.0
                ),
                "nosl_is_az_dirichlet_alpha": getattr(
                    args, "nosl_is_az_dirichlet_alpha", 0.0
                ),
                "nosl_is_az_dirichlet_epsilon": getattr(
                    args, "nosl_is_az_dirichlet_epsilon", 0.0
                ),
                "nosl_is_az_master_nonce": getattr(
                    args, "nosl_is_az_master_nonce", 0x4E4F534C4953415A
                ),
                "nosl_is_az_batch_size": getattr(
                    args, "nosl_is_az_batch_size", 64
                ),
                "nosl_is_az_micro_batch_size": getattr(
                    args, "nosl_is_az_micro_batch_size", 2048
                ),
                "nosl_is_az_replay_capacity": getattr(
                    args, "nosl_is_az_replay_capacity", 131072
                ),
                "nosl_is_az_replay_min_size": getattr(
                    args, "nosl_is_az_replay_min_size", 1
                ),
                "nosl_is_az_replay_max_old_fraction": getattr(
                    args, "nosl_is_az_replay_max_old_fraction", 1.0
                ),
                "nosl_is_az_leaf_batch_size": getattr(
                    args, "nosl_is_az_leaf_batch_size", 64
                ),
                "nosl_is_az_inference_batch_size": getattr(
                    args, "nosl_is_az_inference_batch_size", 32
                ),
                "nosl_is_az_inference_max_wait_ms": getattr(
                    args, "nosl_is_az_inference_max_wait_ms", 0.5
                ),
                "nosl_is_az_inference_schedule": getattr(
                    args, "nosl_is_az_inference_schedule", "fixed_wait"
                ),
                "nosl_is_az_updates": getattr(args, "nosl_is_az_updates", 1),
                "nosl_is_az_update_schedule": model.nosl_is_az_update_schedule,
                "nosl_is_az_policy_coef": getattr(
                    args, "nosl_is_az_policy_coef", 1.0
                ),
                "nosl_is_az_value_coef": getattr(
                    args, "nosl_is_az_value_coef", 0.5
                ),
            },
        )
        # SB3 load restores weights/optimizer state but not the RNG state.
        # Re-apply the run seed so resume and fresh runs initialize sampling
        # the same way.
        model.set_random_seed(args.seed)
        # Each resume command creates a new run directory and metrics callback.
        # Do not carry the checkpoint's rolling episode statistics into that
        # new run: they may describe an older environment contract (notably the
        # pre-fix map-DAG boss loop) and otherwise pollute ep_len_mean until 100
        # newly completed episodes have displaced them.  Training state,
        # optimizer state, and num_timesteps remain continuous.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        model._episode_num = 0
        print(f"[trainer] resumed from {args.resume} @ {model.num_timesteps} steps")

    if args.init_from:
        # Copy only the policy weights (encoder + heads) from the pretrained
        # checkpoint; the freshly built model keeps its step-0 counter and fresh
        # optimizer state. Same obs/action space + arch => strict load.
        assert pretrained is not None
        source_arch = _checkpoint_policy_arch(pretrained)
        if source_arch == "combat-pointer" and getattr(args, "arch", None) == "split-pointer":
            source = pretrained.policy.state_dict()
            target = model.policy.state_dict()
            incompatible = [
                key for key, value in source.items()
                if key not in target or target[key].shape != value.shape
            ]
            if incompatible:
                raise ValueError(
                    "combat checkpoint does not match the split combat specialist: "
                    + ", ".join(incompatible[:8])
                )
            target.update(source)
            model.policy.load_state_dict(target)
            initialized = len(target.keys() - source.keys())
        else:
            initialized = _warm_start_policy(
                model,
                pretrained,
                allow_global_map_extension=getattr(args, "global_map_view", False),
                allow_map_dag_extension=(
                    getattr(args, "global_map_view", False)
                    and getattr(args, "map_dag", False)
                ),
                allow_deck_size_extension=getattr(
                    args, "deck_size_observation", False
                ),
                allow_card_language_extension=args.card_language,
                allow_combat_language_extension=args.combat_language,
                allow_event_option_extension=getattr(
                    args, "event_option_observation", False
                ),
                skip_combat_mismatch=bool(
                    getattr(args, "nosl_is_az_inner_only", False)
                ),
                allow_combat_contract_migration=(
                    getattr(args, "combat_action_contract", None) in C.NEURAL_CONTRACTS
                    and "combat_legal" not in pretrained.observation_space.spaces),
            )
        del pretrained
        print(f"[trainer] warm-started policy weights from {args.init_from}")
        if initialized:
            print(
                f"[trainer] initialized {initialized} new semantic parameter tensors"
            )
        if random_combat_state is not None:
            model.policy.core.combat.load_state_dict(random_combat_state)
            random_combat_state = None
            print("[trainer] reset combat specialist to its random initialization")

    if hasattr(args, "hierarchical_combat_hp"):
        has_source_checkpoint = bool(args.resume or args.init_from)
        objective_changed = (
            has_source_checkpoint
            and bool(args.hierarchical_combat_hp) != source_hierarchical
        )
        reset_count = model.configure_hierarchical_combat_hp(
            enabled=args.hierarchical_combat_hp,
            combat_gamma=args.combat_hp_gamma,
            combat_gae_lambda=args.combat_hp_gae_lambda,
            reset_combat_value=objective_changed,
            reset_seed=args.seed,
        )
        if reset_count:
            print(
                "[trainer] combat value objective changed; reset "
                f"{reset_count} combat value parameter tensors"
            )

    if getattr(args, "nosl_is_az_inner_only", False):
        frozen, trainable = model.configure_nosl_is_az_inner_only()
        print(
            "[trainer] NOSL-IS-AZ inner-only: skipped overworld PPO and froze "
            f"{frozen:,} parameters; {trainable:,} combat parameters trainable"
        )

    if args.freeze_encoder:
        core = getattr(model.policy, "core", None)
        shared = getattr(core, "shared", None)
        combat = getattr(core, "combat", None)
        combat_encoder = getattr(combat, "encoder", None)
        if shared is None or combat_encoder is None:
            raise ValueError(
                "--freeze-encoder requires a split or combat specialist policy"
            )
        frozen = 0
        for module in (shared, combat_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
                frozen += 1
        # Frozen parameters stay in Adam's parameter groups but receive no
        # gradients, so the optimizer leaves them untouched.
        print(
            f"[trainer] froze shared tokenizers and combat encoder "
            f"({frozen} parameter tensors); specialist heads and the "
            "overworld encoder remain trainable"
        )

    print(
        f"[trainer] task={task.name} steps={args.steps} n_envs={args.n_envs} "
        f"vec={args.vec} rollout={rollout_size} ascension={args.ascension} "
        f"micro_batch={args.batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} "
        f"matmul={args.float32_matmul_precision} amp={args.amp_mode} "
        f"card_language={args.card_language} "
        f"combat_language={args.combat_language} "
        f"hierarchical_combat_hp={getattr(args, 'hierarchical_combat_hp', False)} "
        f"nosl_is_az={getattr(args, 'nosl_is_az', False)} "
        f"nosl_replay_enabled={getattr(args, 'nosl_is_az_replay_enabled', True)} "
        f"nosl_inner_only={getattr(args, 'nosl_is_az_inner_only', False)} "
        f"nosl_cpp={getattr(args, 'nosl_is_az_cpp', False)} "
        f"nosl_sims={getattr(args, 'nosl_is_az_simulations', 0)} "
        f"nosl_depth={getattr(args, 'nosl_is_az_depth', 0)} "
        f"nosl_search={getattr(args, 'nosl_is_az_search_mode', 'puct')} "
        f"nosl_replay={getattr(args, 'nosl_is_az_replay_capacity', 0)} "
        f"nosl_batch={getattr(args, 'nosl_is_az_batch_size', 0)} "
        f"nosl_micro={getattr(args, 'nosl_is_az_micro_batch_size', 0)} "
        f"ent_coef={args.ent_coef} lr={args.learning_rate}"
    )

    try:
        # --steps is always the ABSOLUTE total-steps target (run until
        # num_timesteps reaches it), in both fresh and resume runs. SB3's
        # learn(total_timesteps=N) with reset_num_timesteps=False treats N as
        # "additional steps" and adds num_timesteps internally, so on resume we
        # subtract what's already been trained to keep --steps absolute.
        if args.resume:
            remaining = args.steps - model.num_timesteps
            if remaining <= 0:
                raise ValueError(
                    f"--steps {args.steps} <= checkpoint's {model.num_timesteps} "
                    f"already-trained steps; nothing to do. Pass a larger --steps."
                )
        else:
            remaining = args.steps

        model.learn(
            total_timesteps=remaining,
            callback=callback,
            progress_bar=False,
            tb_log_name=tag,
            reset_num_timesteps=not bool(args.resume),
        )
        final_path = out_dir / "final"
        model.save(str(final_path))
        print(f"[trainer] saved {final_path}.zip")
    except KeyboardInterrupt:
        # Ctrl+C: model.learn() bails out here. Save what we have — otherwise
        # we'd lose everything since the last periodic checkpoint.
        interrupted_path = out_dir / "interrupted"
        model.save(str(interrupted_path))
        print(f"[trainer] interrupted — saved {interrupted_path}.zip")
    finally:
        venv.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
