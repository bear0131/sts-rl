"""Minimal MaskablePPO training loop on SimCombatEnv.

Purpose right now: a trainability sanity check, not a strong agent. Two modes:

  --overfit   (default) hammer ONE fixed normal combat. A random policy on the
              starter deck wins this sometimes; a learning policy should drive
              episode reward up to a stable near-+1 and win-rate toward 1.0.
              If that doesn't happen, the obs/mask/reward wiring is wrong, not
              the hyperparameters.

  --pool P    train on the random act-1 P pool ("normal"/"elite"/"boss"). Real
              training target; convergence here is a bonus, not the gate.

Why MaskablePPO: only a subset of the 71-action space is legal each step, and
the legal set shifts as the abstraction grows. We bridge SimCombatEnv.action_mask()
to sb3-contrib via ActionMasker so the env stays masking-library-agnostic.

Run from repo root (so `env` + the sim build dir resolve):
    python -m rl.training.overfit_test                       # overfit one combat
    python -m rl.training.overfit_test --steps 200000
    python -m rl.training.overfit_test --pool normal --steps 1000000
    python -m rl.training.overfit_test --eval runs/overfit/final.zip --episodes 200
"""

import argparse
import os

import numpy as np

from env.core.combat_env import SimCombatEnv
from env.core import factories

# sb3 imports are deferred into main so `--help` and import of this module work
# on a box without torch/sb3 installed (e.g. the sim-less laptop).


# A fixed normal combat that the starter deck can plausibly win — the overfit
# target. Jaw Worm: single foe, no summons, no nasty gimmicks. seed arbitrary.
_OVERFIT_SEED = 42
_OVERFIT_ENCOUNTER = "Jaw Worm"


def _mask_fn(env):
    """ActionMasker callback: sb3-contrib calls this to get the legal mask.

    env is the (possibly wrapped) instance. SimCombatEnv exposes action_mask()
    (single combat); the whole-run SlayTheSpireEnv exposes action_masks() (the
    phase-gated autoregressive mask). Support both."""
    u = env.unwrapped
    if hasattr(u, "action_masks"):
        return u.action_masks()
    return u.action_mask()


def make_run_env(ascension=0, render_mode=None):
    """Build a masked whole-run SlayTheSpireEnv (action v2). Returns a thunk."""
    from sb3_contrib.common.wrappers import ActionMasker
    from env.core.run_env import SlayTheSpireEnv

    def _thunk():
        env = SlayTheSpireEnv(ascension=ascension, render_mode=render_mode)
        return ActionMasker(env, _mask_fn)

    return _thunk


def make_env(pool="normal", overfit=False, ascension=0, render_mode=None):
    """Build a masked SimCombatEnv. Returns a thunk for SB3's vec env makers."""
    from sb3_contrib.common.wrappers import ActionMasker

    def _thunk():
        if overfit:
            gc_factory = factories.fixed_factory(
                _OVERFIT_SEED, _OVERFIT_ENCOUNTER, floor=1, ascension=ascension
            )
            env = SimCombatEnv(gc_factory=gc_factory, render_mode=render_mode)
        else:
            env = SimCombatEnv(pool=pool, ascension=ascension, render_mode=render_mode)
        return ActionMasker(env, _mask_fn)

    return _thunk


def train(args):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv

    os.makedirs(args.out, exist_ok=True)
    if args.run:
        env_makers = [make_run_env(ascension=args.ascension) for _ in range(args.n_envs)]
    else:
        env_makers = [make_env(pool=args.pool, overfit=args.overfit, ascension=args.ascension)
                      for _ in range(args.n_envs)]
    venv = DummyVecEnv(env_makers)

    model = MaskablePPO(
        # obs is a gymnasium Dict (typed categorical + scalar groups, see
        # env/core/encode.py) -> MultiInputPolicy. NOTE: the default extractor
        # treats every Box as continuous; the categorical id groups (card/monster/
        # move/... indices) really want an embedding layer. A custom
        # features-extractor that embeds the id groups is the next model-side step
        # (decision C in obs_design.md) — MlpPolicy on a flat vector is gone.
        "MultiInputPolicy",
        venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.out if args.tensorboard else None,
    )

    tag = "whole-run" if args.run else ("overfit" if args.overfit else f"pool-{args.pool}")
    print(f"[train] {tag}: {args.steps} steps, {args.n_envs} env(s)")
    model.learn(total_timesteps=args.steps, progress_bar=False)

    final = os.path.join(args.out, "final.zip")
    model.save(final)
    print(f"[train] saved {final}")

    # quick deterministic eval right after training so the sanity signal is
    # visible without a separate command.
    eval_env = (make_run_env(ascension=args.ascension)() if args.run
                else make_env(pool=args.pool, overfit=args.overfit, ascension=args.ascension)())
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes,
        deterministic=True, use_masking=True,
    )
    print(f"[eval] {args.eval_episodes} eps: mean_reward={mean_r:.3f} +/- {std_r:.3f}")
    return model


def evaluate(args):
    from sb3_contrib import MaskablePPO
    from rl.policies import split_policy as _split_policy  # noqa: F401

    model = MaskablePPO.load(args.eval)
    if args.run:
        env = make_run_env(ascension=args.ascension)()
    else:
        env = make_env(pool=args.pool, overfit=args.overfit, ascension=args.ascension)()
    wins = 0
    rewards = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            # _mask_fn supports both the single-combat action_mask() and the
            # whole-run action_masks(); use it so --eval works on either env.
            mask = _mask_fn(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
        # terminal reward sign is the clean win/loss signal (see combat_env)
        if total > 0:
            wins += 1
    rewards = np.asarray(rewards)
    print(f"[eval] {args.episodes} eps: win_rate={wins/args.episodes:.3f} "
          f"mean_reward={rewards.mean():.3f} +/- {rewards.std():.3f}")


def build_parser():
    p = argparse.ArgumentParser(description="MaskablePPO on SimCombatEnv")
    p.add_argument("--overfit", action="store_true", default=True,
                   help="train on one fixed combat (default; trainability check)")
    p.add_argument("--run", action="store_true", default=False,
                   help="train on the whole-run SlayTheSpireEnv (action v2) instead "
                        "of single combats")
    p.add_argument("--pool", default="normal", choices=["normal", "elite", "boss"],
                   help="encounter pool when not overfitting")
    p.add_argument("--no-overfit", dest="overfit", action="store_false",
                   help="train on the random --pool distribution instead")
    p.add_argument("--steps", type=int, default=100_000, help="total timesteps")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/overfit", help="checkpoint/log dir")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="post-train eval episode count")
    # eval-only mode
    p.add_argument("--eval", metavar="CKPT", default=None,
                   help="skip training; load CKPT and report win-rate")
    p.add_argument("--episodes", type=int, default=200, help="--eval episode count")
    return p


def main():
    args = build_parser().parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
