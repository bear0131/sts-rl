"""Capture/compare numerical behavior across revisions on one CPU runtime.

Run capture in each checkout with the SAME interpreter and explicit native build.
JSON contains per-array byte hashes, not unstable zip/pickle serialization hashes.
The subprocess budget applies to the complete capture, including imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def capture(output: Path, restore: Path | None) -> None:
    import platform
    import numpy as np
    import torch
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv
    from env.core.run_env import SlayTheSpireEnv, unpack_nosl_observation_buffer
    import slaythespire as sts
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.policies.split_policy import SplitStSPointerPolicy
    from rl.training.reward import ShapedRewardWrapper

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    values = {}

    def record(name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            assert value.dtype != object, name
            values[name] = dict(dtype=value.dtype.str, shape=list(value.shape),
                                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
        elif isinstance(value, dict):
            for key, item in value.items():
                record(f"{name}/{key}", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                record(f"{name}/{index}", item)
        elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
            values[name] = value.item() if isinstance(value, np.generic) else value
        else:
            raise TypeError(f"unsupported snapshot field {name}: {type(value)}")

    class ShortCombatEnv(SlayTheSpireEnv):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            for _ in range(64):
                if self.thick.bc is not None:
                    break
                self.step(int(np.flatnonzero(self.action_masks())[0]))
            assert self.thick.bc is not None
            self.thick.bc.set_player_strength(999)
            self._invalidate_backend_cache()
            return self._obs(), self._info()

    for mode in ("ppo", "hierarchical", "puct", "strict_replay"):
        search = mode in ("puct", "strict_replay")
        env_type = ShortCombatEnv if search else SlayTheSpireEnv
        env = DummyVecEnv([lambda: TimeLimit(ShapedRewardWrapper(env_type()), 12)] * 2)
        options = {}
        if search:
            options = dict(nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_depth=0,
                           nosl_is_az_simulations=4, nosl_is_az_replay_capacity=128,
                           nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1,
                           nosl_is_az_search_mode="full_gumbel" if mode == "strict_replay" else "puct",
                           nosl_is_az_paper_full_gumbel=mode == "strict_replay",
                           nosl_is_az_replay_enabled=True,
                           nosl_is_az_replay_max_old_fraction=0.9,
                           nosl_is_az_updates=2)
        try:
            model = AccumulatingMaskablePPO(
                SplitStSPointerPolicy, env, n_steps=16, batch_size=8, n_epochs=1,
                gradient_accumulation_steps=2, hierarchical_combat_hp=mode != "ppo",
                policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), seed=42,
                device="cpu", **options)
            model.learn(64)
            assert model.num_timesteps == 64
            record(f"{mode}/weights", model.policy.state_dict())
            record(f"{mode}/adam", model.policy.optimizer.state_dict())
            for name in ("observations", "actions", "rewards", "episode_starts", "values", "log_probs"):
                record(f"{mode}/rollout/{name}", getattr(model.rollout_buffer, name))
            record(f"{mode}/torch_rng", torch.get_rng_state())
            record(f"{mode}/numpy_rng", np.random.get_state())
            if search:
                replay = model._nosl_is_az_replay
                assert replay is not None and replay.size > 0
                for name in ("size", "position", "total_added"):
                    record(f"{mode}/replay/{name}", getattr(replay, name))
                # Only initialized slots: unused np.empty capacity has no semantics.
                record(f"{mode}/replay/observations", {k: v[:replay.size] for k, v in replay.observations.items()})
                for name in ("action_masks", "policy_targets", "policy_weights", "value_targets", "sample_ids"):
                    record(f"{mode}/replay/{name}", getattr(replay, name)[:replay.size])
            checkpoint_dir = output.parent / (output.stem + "-checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            model.save(checkpoint_dir / mode)
            # On the candidate checkout, also load the pre-refactor checkpoint.
            restored = AccumulatingMaskablePPO.load(
                (restore or checkpoint_dir) / f"{mode}.zip", device="cpu")
            record(f"{mode}/restored_weights", restored.policy.state_dict())
            record(f"{mode}/restored_adam", restored.policy.optimizer.state_dict())
            obs = env.reset()
            for step in range(8):
                masks = np.stack(env.env_method("action_masks"))
                action = restored.predict(obs, deterministic=True, action_masks=masks)[0]
                tensor_obs, _ = restored.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    value = restored.policy.predict_values(tensor_obs)
                record(f"{mode}/eval/{step}", dict(obs=obs, masks=masks, action=action, value=value))
                obs, reward, done, _ = env.step(action)
                record(f"{mode}/eval/{step}/transition", dict(reward=reward, done=done))
        finally:
            env.close()

    env = ShortCombatEnv()
    try:
        env.reset(seed=57)
        for mode, strict in ((0, False), (2, False), (2, True)):
            prefix = f"native/{mode}/{strict}"
            response = env.nosl_is_az_cpp_begin(np.zeros(181), state_nonce=123,
                simulations=8, depth=0, root_search_mode=mode, paper_full_gumbel=strict,
                leaf_batch_size=1, compact_observation=True)
            for index in range(32):
                if response["done"]:
                    break
                record(f"{prefix}/leaf/{index}", unpack_nosl_observation_buffer(response))
                updates = [(leaf["request_id"], 0.4, np.zeros(181)) for leaf in response["leaves"]]
                response = env.nosl_is_az_cpp_backprop_and_leaf_batch(updates, max_leaves=1)
            assert response["done"]
            for name in ("action", "visit_policy", "root_value", "simulations"):
                record(f"{prefix}/result/{name}", response["result"][name])
    finally:
        env.close()
    native = Path(sts.__file__)
    identity = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), threads=1,
                    device="cpu", hashseed=os.environ["PYTHONHASHSEED"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.write_text(json.dumps(dict(identity=identity, commit=commit, values=values), indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(f"captured {len(values)} fields: {output}")


def compare(baseline: Path, candidate: Path) -> None:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (baseline, candidate))
    if a["identity"] != b["identity"]:
        raise SystemExit("runtime/native identities differ; cannot claim same-runtime bitwise")
    missing = object()
    differences = [k for k in sorted(a["values"].keys() | b["values"].keys())
                   if a["values"].get(k, missing) != b["values"].get(k, missing)]
    if differences:
        raise SystemExit("bitwise differences:\n" + "\n".join(differences[:80]))
    print(f"BITWISE PASS: {len(a['values'])} fields; {a['commit']} -> {b['commit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--restore", type=Path, help="baseline checkpoint directory")
    cap.add_argument("--timeout", type=float, default=180)
    cap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    comp = sub.add_parser("compare")
    comp.add_argument("baseline", type=Path)
    comp.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.baseline, args.candidate)
    elif args.worker:
        capture(args.output, args.restore)
    else:
        if args.output.exists():
            parser.error("output exists; choose a fresh evidence path")
        if args.timeout <= 0:
            parser.error("timeout must be positive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        try:
            result = subprocess.run([sys.executable, "-m", "tools.validation.bitwise", *sys.argv[1:], "--worker"],
                                    env=environment, timeout=args.timeout)
            raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            raise SystemExit(124)


if __name__ == "__main__":
    main()
