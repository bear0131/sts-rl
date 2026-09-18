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


def _record(values, name, value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        assert value.dtype != object, name
        values[name] = dict(dtype=value.dtype.str, shape=list(value.shape),
                            sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
    elif isinstance(value, dict):
        for key, item in value.items():
            _record(values, f"{name}/{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _record(values, f"{name}/{index}", item)
    elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
        values[name] = value.item() if isinstance(value, np.generic) else value
    else:
        raise TypeError(f"unsupported snapshot field {name}: {type(value)}")


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
        _record(values, name, value)

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


def capture_search(output: Path, repeats: int, record_repeats: int) -> None:
    import platform
    import time
    import numpy as np
    import torch
    from env.core.run_env import SlayTheSpireEnv
    import slaythespire as sts
    import psutil

    torch.set_num_threads(1)
    build = Path(os.environ["STS_LIGHTSPEED_BUILD_DIR"]).resolve()
    native = Path(sts.__file__).resolve()
    if native.parent != build:
        raise ValueError(f"loaded unexpected native build: {native}")
    cache = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    compiler = cache["CMAKE_CXX_COMPILER"]
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    identity = dict(kind="native-search", python=platform.python_version(),
                    platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                    native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), threads=1,
                    device="cpu", hashseed=os.environ["PYTHONHASHSEED"],
                    compiler_sha256=hashlib.sha256(compiler_version.encode()).hexdigest(),
                    build_flags={key: cache.get(key) for key in (
                        "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                        "STS_ENABLE_RULE_TRACE", "PYTHON_VERSION", "PYBIND11_INTERNALS_VERSION")})
    values, timings = {}, []
    process = psutil.Process()
    capture_started = time.perf_counter()
    cpu_before = process.cpu_times()
    available_before = psutil.virtual_memory().available
    decks = {
        "basic": ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"],
        "armaments": ["ARMAMENTS", "ANGER", "SHRUG_IT_OFF", "RAMPAGE", "POMMEL_STRIKE"] * 2,
        "long": ["STRIKE_RED"] * 70,
        "exhume": ["EXHUME"] * 5,
    }
    cases = [(deck, encounter, mode, strict, batch, depth, boundary)
             for deck, encounter in (("basic", "Jaw Worm"), ("armaments", "Two Louse"),
                                     ("long", "Gremlin Leader"), ("exhume", "Jaw Worm"))
             for mode, strict, batch, depth, boundary in (
                 (2, True, 1, 0, False), (2, False, 4, 3, False),
                 (0, False, 1, 2, True))]
    for repeat in range(repeats):
        for case_index, (deck, encounter, mode, strict, batch, depth, boundary) in enumerate(cases):
            for packed in (False, True):
                recording = repeat < record_repeats
                game = sts.GameContext(sts.CharacterClass.IRONCLAD, 19 + case_index, 0)
                game.floor_num = 1
                game.set_deck([(card, 0) for card in decks[deck]])
                battle = sts.create_battle(game, encounter)
                if deck == "exhume":
                    battle.apply_custom_state({"player": {"energy": 3}, "hand": ["EXHUME"],
                                               "draw": [], "discard": [],
                                               "exhaust": ["STRIKE_RED"] * 129})
                before = battle.to_dict()
                logits = np.linspace(-0.7, 0.9, 181, dtype=np.float64)
                started = time.perf_counter()
                session_started = started
                session = battle.nosl_neural_search_begin(
                    logits, simulations=128, depth=depth, nonce=173 + case_index,
                    root_search_mode=mode, paper_full_gumbel=strict, root_value=0.37,
                    end_turn_as_leaf=boundary, dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25)
                prefix = f"r{repeat}/c{case_index}/packed{packed}"
                elapsed = time.perf_counter() - started
                leaves = batches = 0
                while not session.done():
                    started = time.perf_counter()
                    requests = session.next_batch_packed(batch) if packed else session.next_batch(batch)
                    elapsed += time.perf_counter() - started
                    ids = requests["request_ids"] if packed else [r["request_id"] for r in requests]
                    if not ids:
                        assert session.done(), prefix
                        break
                    if recording and packed:
                        _record(values, f"{prefix}/batch{batches}", requests)
                    elif recording:
                        for leaf in requests:
                            leaf = dict(leaf)
                            state = leaf.pop("state")
                            leaf["state_sha256"] = hashlib.sha256(json.dumps(
                                state, sort_keys=True, allow_nan=False).encode()).hexdigest()
                            _record(values, f"{prefix}/batch{batches}/{leaf['request_id']}", leaf)
                    leaf_values = np.asarray([0.15 + (int(i) % 11) * 0.06 for i in ids])
                    policy = np.stack([logits + np.sin(np.arange(181) + int(i)) * 0.1 for i in ids])
                    started = time.perf_counter()
                    if packed:
                        session.backprop_batch_array(np.asarray(ids, dtype=np.uint64), leaf_values, policy)
                    else:
                        for request_id, value, row in zip(ids, leaf_values, policy):
                            session.backprop(request_id, value, row)
                    elapsed += time.perf_counter() - started
                    leaves += len(ids)
                    batches += 1
                    assert batches <= 128, prefix
                started = time.perf_counter()
                result = session.result()
                elapsed += time.perf_counter() - started
                end_to_end = time.perf_counter() - session_started
                assert result["simulations"] == 128
                assert battle.to_dict() == before, prefix
                if recording:
                    _record(values, f"{prefix}/result", result)
                    root = session.tree_snapshot([], 256)
                    _record(values, f"{prefix}/root", root)
                    for index in range(len(root["children"])):
                        _record(values, f"{prefix}/child{index}", session.tree_snapshot([index], 256))
                    # Compare future live transitions as well as the exposed parent state.
                    for step in range(3):
                        battle.execute(sts.Action(sts.ActionType.END_TURN))
                        _record(values, f"{prefix}/future{step}", battle.to_dict())
                        if battle.is_battle_over:
                            break
                timings.append(dict(repeat=repeat, case=case_index, packed=packed, simulations=128,
                                    leaves=leaves, batches=batches, elapsed_sec=elapsed,
                                    end_to_end_sec=end_to_end))
    cpu_after = process.cpu_times()
    wall = time.perf_counter() - capture_started
    memory = process.memory_info()
    resources = dict(capture_wall_sec=wall,
                     process_cpu_sec=(cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system),
                     process_rss_bytes=memory.rss,
                     peak_rss_bytes=getattr(memory, "peak_wset", None),
                     system_available_bytes=min(available_before, psutil.virtual_memory().available),
                     resource_samples=2, gpu_util=None, device_vram=None,
                     torch_allocated=None, torch_reserved=None, torch_peak=None,
                     note="CPU-only fixed callback benchmark; resources include trace recording; no GPU query")
    output.write_text(json.dumps(dict(identity=identity, values=values, timings=timings, resources=resources),
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(f"captured {len(values)} fields / {len(timings) * 128} simulations: {output}")


def compare(baseline: Path, candidate: Path, *, allow_native_rebuild=False) -> None:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (baseline, candidate))
    identities = [dict(row["identity"]) for row in (a, b)]
    if allow_native_rebuild:
        if any(row.get("kind") != "native-search" or "compiler_sha256" not in row
               or "build_flags" not in row for row in identities):
            raise SystemExit("native rebuild comparison requires capture-search build identities")
        for row in identities:
            row.pop("native_sha256")
    if identities[0] != identities[1]:
        raise SystemExit("runtime/native identities differ; cannot claim same-runtime bitwise")
    missing = object()
    differences = [k for k in sorted(a["values"].keys() | b["values"].keys())
                   if a["values"].get(k, missing) != b["values"].get(k, missing)]
    if differences:
        raise SystemExit("bitwise differences:\n" + "\n".join(differences[:80]))
    print(f"BITWISE PASS: {len(a['values'])} fields; native rebuild allowed={allow_native_rebuild}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--restore", type=Path, help="baseline checkpoint directory")
    cap.add_argument("--timeout", type=float, default=180)
    cap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    search = sub.add_parser("capture-search", help="fixed native128 traces across explicit native rebuilds")
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--repeats", type=int, default=1, choices=range(1, 11))
    search.add_argument("--record-repeats", type=int, default=1, choices=range(1, 11))
    search.add_argument("--timeout", type=float, default=180)
    search.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    comp = sub.add_parser("compare")
    comp.add_argument("baseline", type=Path)
    comp.add_argument("candidate", type=Path)
    comp.add_argument("--allow-native-rebuild", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.baseline, args.candidate, allow_native_rebuild=args.allow_native_rebuild)
    elif args.worker:
        if args.command == "capture-search":
            if args.record_repeats > args.repeats:
                parser.error("record-repeats cannot exceed repeats")
            capture_search(args.output, args.repeats, args.record_repeats)
        else:
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
