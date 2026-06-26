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

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from rl.tasks import TASKS


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
             "(~20 steps/floor), reward.floor_base then re-weights.",
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

    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            "batch_size must divide n_envs * n_steps: "
            f"{args.batch_size} does not divide {rollout_size}"
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
        callbacks.append(
            WandbCallback(
                gradient_save_freq=args.wandb_grad_freq,
                model_save_path=str(out_dir / "wandb_models" / wandb_run.id),
                model_save_freq=args.wandb_save_freq,
                verbose=2,
            )
        )

    callback = CallbackList(callbacks)

    # sync_tensorboard needs SB3 to actually emit TensorBoard logs, so enable the
    # TensorBoard sink whenever either --tensorboard or --wandb is set.
    tensorboard_log = (
        str(out_dir / "tensorboard") if (args.tensorboard or args.wandb) else None
    )

    if not args.resume:
        policy, policy_kwargs = task.build_policy(args)
        model = MaskablePPO(
            policy=policy,
            env=venv,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
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
        model = MaskablePPO.load(
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
            },
        )
        print(f"[trainer] resumed from {args.resume} @ {model.num_timesteps} steps")

    if args.init_from:
        if args.resume:
            raise ValueError("--init-from and --resume are mutually exclusive: "
                             "--resume already restores weights + step counter.")
        # Copy only the policy weights (encoder + heads) from the pretrained
        # checkpoint; the freshly built model keeps its step-0 counter and fresh
        # optimizer state. Same obs/action space + arch => strict load.
        pretrained = MaskablePPO.load(args.init_from, device=args.device)
        model.policy.load_state_dict(pretrained.policy.state_dict())
        del pretrained
        print(f"[trainer] warm-started policy weights from {args.init_from}")

    if args.freeze_encoder:
        core = getattr(model.policy, "core", None)
        if core is None or not hasattr(core, "encoder"):
            raise ValueError("--freeze-encoder requires the transformer-pointer "
                             "policy (no core.encoder to freeze on this arch).")
        n = 0
        for p in core.encoder.parameters():
            p.requires_grad = False
            n += 1
        # Params stay in the optimizer but get no grad, so Adam skips them.
        print(f"[trainer] froze encoder ({n} param tensors) — only heads train")

    print(
        f"[trainer] task={task.name} steps={args.steps} n_envs={args.n_envs} "
        f"vec={args.vec} rollout={rollout_size} ascension={args.ascension} "
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
