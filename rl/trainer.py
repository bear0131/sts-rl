"""Unified MaskablePPO trainer for Slay the Spire.

Two orthogonal axes: the *training* (this file — PPO algorithm, VecEnv,
checkpoints, wandb, resume/init-from, the learn() loop) and the *task* (rl/tasks/
— which env/reward/metrics/policy). Pick a task with --task; every shared PPO
hyperparameter is configurable from the CLI.

Run from repo root:

    python -m rl.trainer --task combat
    python -m rl.trainer --task combat --ent-coef 0.03 --steps 5000000 --wandb
    python -m rl.trainer --task wholerun --arch transformer-pointer \
        --init-from rl/runs/combat/<ts>/final.zip --freeze-encoder
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from rl.tasks import TASKS
from rl.accumulating_ppo import AccumulatingMaskablePPO


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


def _checkpoint_policy_arch(model: AccumulatingMaskablePPO) -> str | None:
    """Return the trainer architecture name represented by a loaded policy."""
    from rl.model import StSMLPExtractor, StSTransformerExtractor
    from rl.pointer_policy import StSPointerPolicy

    policy = model.policy
    if isinstance(policy, StSPointerPolicy):
        return "transformer-pointer"
    extractor = getattr(policy, "features_extractor", None)
    if isinstance(extractor, StSTransformerExtractor):
        return "transformer-flat"
    if isinstance(extractor, StSMLPExtractor):
        return "mlp-flat"
    return None


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
                allow_event_option_extension)):
        model.policy.load_state_dict(source)
        return 0

    target = model.policy.state_dict()
    incompatible = []
    initialized = []
    for key, value in source.items():
        if key not in target:
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
        if (
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
    parser = argparse.ArgumentParser(
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
        help="VecEnv backend. subproc = one process per env (true multi-core "
             "sampling; default). dummy = all envs in this process, serial "
             "(single core — use only for debugging).",
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
        help="Freeze the shared Transformer encoder (core.encoder) — only the "
             "policy/value heads train. Intended with --init-from to hand a "
             "combat-pretrained trunk to whole-run training without it being "
             "pulled away by meta-game gradients (the deliberately myopic-trunk "
             "tradeoff; see doc/night_0625/combat_pretrain_decision.md).",
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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", choices=sorted(TASKS))
    known, _ = pre.parse_known_args()
    if known.task is None:
        # Defer to the full parser for a proper error + usage message.
        build_parser(next(iter(TASKS.values()))).parse_args()
        return
    task = TASKS[known.task]

    args = build_parser(task).parse_args()

    # Whole-run fresh training retains transformer-flat as its default.  When
    # warm-starting, infer the architecture from the checkpoint so a combat
    # checkpoint (always transformer-pointer) works without a second, easy to
    # miss --arch flag. Keep the loaded object for the actual weight transfer.
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

    if hasattr(args, "global_map_view"):
        from rl.model import uses_global_map_observation

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
        from rl.model import uses_deck_size_observation

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
        from rl.model import uses_neow_observation

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
        from rl.model import uses_event_option_observation

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
        from rl.model import uses_card_language, uses_combat_language

        if uses_card_language(pretrained) and not args.card_language:
            args.card_language = True
            print(f"[trainer] enabled --card-language from {args.init_from}")
        if uses_combat_language(pretrained) and not args.combat_language:
            args.combat_language = True
            print(f"[trainer] enabled --combat-language from {args.init_from}")

    if args.resume and (not args.card_language or not args.combat_language):
        from stable_baselines3.common.save_util import load_from_zip_file
        from rl.model import uses_card_language, uses_combat_language

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
            args.arch = (
                "transformer-pointer"
                if getattr(args, "global_map_view", False)
                and getattr(args, "map_dag", False)
                else "transformer-flat"
            )
            if args.arch == "transformer-pointer":
                print("[trainer] selected transformer-pointer for --map-dag")
        else:
            source_arch = _checkpoint_policy_arch(pretrained)
            if source_arch is None:
                raise ValueError(
                    "Could not infer the policy architecture in --init-from. "
                    "Pass --arch explicitly."
                )
            args.arch = source_arch
            print(
                f"[trainer] inferred --arch {args.arch} from {args.init_from}"
            )

    if pretrained is not None and hasattr(args, "arch"):
        source_arch = _checkpoint_policy_arch(pretrained)
        if source_arch is not None and source_arch != args.arch:
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
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [task.make_env_fn(args) for _ in range(args.n_envs)]
    if args.vec == "subproc":
        # forkserver, not the default fork: forking a process that already
        # imported the native sts module + torch (threads, CUDA context) into a
        # child can deadlock. forkserver starts each worker from a clean-ish
        # process. spawn would also work but is slower to start.
        venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=str(out_dir / "monitor.csv"))
    venv.seed(args.seed)

    # Callback runs once per vectorized step, so adjust for n_envs.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=task.ckpt_prefix,
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
            custom_objects={
                "learning_rate": args.learning_rate,
                "ent_coef": args.ent_coef,
                "clip_range": args.clip_range,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "vf_coef": args.vf_coef,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "performance_profile": args.profile_performance,
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
        )
        del pretrained
        print(f"[trainer] warm-started policy weights from {args.init_from}")
        if initialized:
            print(
                f"[trainer] initialized {initialized} new semantic parameter tensors"
            )

    if args.freeze_encoder:
        core = getattr(model.policy, "core", None)
        if core is None or not hasattr(core, "encoder"):
            raise ValueError("--freeze-encoder requires the transformer-pointer "
                             "policy (no core.encoder to freeze on this arch).")
        n = 0
        for p in core.encoder.parameters():
            p.requires_grad = False
            n += 1
        trainable_extensions = 0
        if getattr(args, "global_map_view", False):
            for module in (
                core.encoder.tok["map_full"],
                core.encoder.role_emb,
            ):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_extensions += 1
            core.encoder.pos["map_full"].requires_grad = True
            trainable_extensions += 1
        if args.card_language:
            card_embedding = core.encoder.shared["card"]
            for name in ("token_embedding", "local_encoder"):
                for p in getattr(card_embedding, name).parameters():
                    p.requires_grad = True
                    trainable_extensions += 1
            card_embedding.program_position.requires_grad = True
            trainable_extensions += 1
            card_embedding.effect_scale.requires_grad = True
            trainable_extensions += 1
            for module in (
                core.encoder.pile_card_language,
                core.encoder.exhaust_card_language,
            ):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_extensions += 1
        if args.combat_language:
            for p in core.encoder.combat_language.parameters():
                p.requires_grad = True
                trainable_extensions += 1
            core.encoder.combat_language_scale.requires_grad = True
            trainable_extensions += 1
        # Params stay in the optimizer but get no grad, so Adam skips them.
        if trainable_extensions:
            print(
                f"[trainer] froze encoder ({n} param tensors) except "
                f"{trainable_extensions} new semantic adapter tensors; "
                "heads also train"
            )
        else:
            print(f"[trainer] froze encoder ({n} param tensors) — only heads train")

    print(
        f"[trainer] task={task.name} steps={args.steps} n_envs={args.n_envs} "
        f"vec={args.vec} rollout={rollout_size} ascension={args.ascension} "
        f"micro_batch={args.batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} "
        f"card_language={args.card_language} "
        f"combat_language={args.combat_language} "
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
