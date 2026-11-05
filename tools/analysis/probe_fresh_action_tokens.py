"""Compare revisiting one bank against never repeating a searched decision.

All arms use the same mature parent, seed, batches, update count and optimizer.
The independent final cohort is excluded from all training and model selection.
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_probe import state_digest
from rl.policies.entity_action_playout import ActionTokenPolicy
from rl.training.resource_monitor import ResourceMonitor
from tools.analysis.lower_bank import prepare_bank
from rl.policies.public_playout import PublicPlayoutPolicy
from rl.policies.small_playout import SmallPlayoutPolicy


def experiment(
    parent,
    first,
    others,
    holdout,
    output,
    seed,
    steps,
    micro,
    batch,
    device,
    pilot_models=None,
    compact_controls=False,
    token_reference=None,
):
    output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    model = AccumulatingMaskablePPO.load(
        parent / "model.zip", device=device, amp_mode="off"
    )
    mature = model.policy.small_playout
    teacher = state_digest(model.policy.state_dict())
    opt_digests = [
        state_digest(model.policy.optimizer.state_dict()),
        state_digest(model.policy.upper_optimizer.state_dict()),
    ]

    def load(path):
        _, legacy, bank, games, sha, _, meta = prepare_bank(
            path, parent, device, teacher_digest=teacher, sketch=compact_controls
        )
        return legacy, bank, games, sha, meta["cohort"]

    legacy, bank0, g0, sha, c0 = load(first)
    val = np.flatnonzero(g0 % 10 == 0)
    train = np.flatnonzero(g0 % 5 != 0)
    history = set()
    repeat_seen = set()

    def register(c):
        ids = set(range(c["first_seed"], c["last_seed"] + 1))
        if (
            len(ids) != 480
            or c["censored_games"]
            or c["completed_games"] != 480
            or history.intersection(ids)
        ):
            raise ValueError("cohorts must be complete and seed-disjoint")
        history.update(ids)

    register(c0)
    report = dict(
        seed=seed,
        parent_policy=teacher,
        batch=batch,
        micro=micro,
        steps_per_group=steps,
        lr=0.001,
        decay=0.0001,
        banks=[],
        arms={},
        pilot_tests={},
    )

    def scores(net, bank, idx):
        if type(net) is SmallPlayoutPolicy:
            bank = getattr(bank, "base", bank)
        inputs, y, t = bank.batch(idx, device)
        logit = net.forward_packed(*inputs)
        logp = logit.log_softmax(-1).masked_fill(~inputs[-1], 0)
        return (
            -(y * logp).sum(-1),
            -(y * y.clamp_min(1e-30).log()).sum(-1),
            logit.argmax(-1) == t,
        )

    @torch.no_grad()
    def evalset(net, bank, ids):
        return np.concatenate(
            [
                torch.stack(scores(net, bank, ids[i : i + micro]), 1).cpu().numpy()
                for i in range(0, len(ids), micro)
            ]
        )

    def summary(rows):
        ce, entropy, top = rows.mean(0, dtype=np.float64)
        return dict(
            ce=ce, kl=ce - entropy, top1=top, entropy=entropy, samples=len(rows)
        )

    specs = (
        [("compact", None), ("sketch", None)]
        if compact_controls
        else [(32, 1), (64, 2), (128, 3)]
    )
    learners = {}
    for width, depth in specs:
        torch.manual_seed(seed)
        if width == "compact":
            n = copy.deepcopy(mature)
        elif width == "sketch":
            n = PublicPlayoutPolicy.from_mature(mature)
        else:
            n = ActionTokenPolicy.from_mature(mature, True, width, depth)
        opt = torch.optim.AdamW(n.parameters(), lr=0.001, weight_decay=0.0001)
        learners[(width, "fresh")] = (n, opt)
        for mode in ("fresh", "repeat"):
            report["arms"][f"{width}-{mode}"] = dict(width=width, depth=depth, curve=[])
    test_folder = output / "weights"
    test_folder.mkdir()
    with ResourceMonitor(
        interval=1.0, gpu=device.startswith("cuda"), torch_device=model.device
    ) as mon:
        for group, path in enumerate([first, *others]):
            if group:
                legacy_new, current, g, sha, c = load(path)
                register(c)
                available = np.arange(len(g))
            else:
                current = bank0
                c = c0
                available = train
            size = steps * batch
            if min(len(available), len(train)) < size:
                raise ValueError("not enough unique decisions in a training bank")
            # Same order for all capacities. Group zero is shared exactly; later
            # repeated samples are shuffled anew, without replacement within a group.
            fresh_ids = np.random.default_rng(seed + group).permutation(available)[
                :size
            ]
            old_ids = np.random.default_rng(seed + group).permutation(train)[:size]
            repeat_seen.update(old_ids.tolist())
            report["unique_decisions"] = dict(
                fresh=(group + 1) * size, repeat=len(repeat_seen)
            )
            report["banks"].append(
                dict(
                    group=group,
                    path=str(path),
                    sha=sha,
                    first_seed=c["first_seed"],
                    last_seed=c["last_seed"],
                    used=size,
                    available=len(available),
                )
            )
            for width, depth in specs:
                modes = ("fresh",) if group == 0 else ("repeat", "fresh")
                for mode in modes:
                    n, opt = learners[width, mode]
                    ids, bank = (
                        (fresh_ids, current) if mode == "fresh" else (old_ids, bank0)
                    )
                    for j in range(0, size, batch):
                        idx = ids[j : j + batch]
                        opt.zero_grad(set_to_none=True)
                        for k in range(0, len(idx), micro):
                            loss, _, _ = scores(n, bank, idx[k : k + micro])
                            (loss.sum() / len(idx)).backward()
                        opt.step()
                    if group == 0:
                        clone = copy.deepcopy(n)
                        opt2 = torch.optim.AdamW(
                            clone.parameters(), lr=0.001, weight_decay=0.0001
                        )
                        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
                        learners[width, "repeat"] = clone, opt2
                    row = dict(
                        group=group + 1,
                        updates=(group + 1) * steps,
                        validation=summary(evalset(n, bank0, val)),
                    )
                    report["arms"][f"{width}-{mode}"]["curve"].append(row)
                    if group == 0:
                        report["arms"][f"{width}-repeat"]["curve"].append(dict(row))
                    print(json.dumps(dict(width=width, mode=mode, **row)), flush=True)
                    (output / "progress.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False)
                    )
            if group:
                del current, legacy_new
                gc.collect()
    report["resources"] = mon.summary
    # Final, fixed-budget weights; no early stopping on the independent bank.
    for (width, mode), (n, opt) in learners.items():
        torch.save(
            dict(
                width=width,
                depth=getattr(n, "depth", None),
                coupled=isinstance(n, ActionTokenPolicy),
                policy=n.state_dict(),
                optimizer=opt.state_dict(),
                parent_lower_digest=state_digest(mature.state_dict()),
                teacher_digest=teacher,
                updates=steps * len(report["banks"]),
            ),
            test_folder / f"{width}-{mode}.pt",
        )
    holdbase, hold, gh, held_sha, ch = load(holdout)
    register(ch)
    hids = np.arange(len(gh))
    out = {"mature": evalset(mature, holdbase, hids)}
    for (width, mode), (n, _) in learners.items():
        out[f"{width}-{mode}"] = evalset(n, hold, hids)
    if token_reference:
        _, _, token_hold, token_games, token_sha, _, _ = prepare_bank(
            holdout, parent, device, teacher_digest=teacher
        )
        if token_sha != held_sha or not np.array_equal(token_games, gh):
            raise ValueError("independent test rows changed")
        for width, depth in [(32, 1), (64, 2), (128, 3)]:
            ck = torch.load(
                token_reference / f"{width}-fresh.pt",
                weights_only=True,
                map_location=device,
            )
            if ck["teacher_digest"] != teacher or ck["updates"] != steps * len(
                report["banks"]
            ):
                raise ValueError("wrong fresh reference model")
            n = ActionTokenPolicy.from_mature(mature, True, width, depth)
            n.load_state_dict(ck["policy"])
            out[f"{width}-fresh"] = evalset(n, token_hold, hids)
    if pilot_models:
        for path in sorted(pilot_models.glob("*.pt")):
            ck = torch.load(path, weights_only=True, map_location=device)
            if ck["parent_lower_digest"] != state_digest(mature.state_dict()):
                raise ValueError("wrong pilot parent")
            n = ActionTokenPolicy.from_mature(
                mature, ck["coupled"], ck["width"], ck["depth"]
            )
            n.load_state_dict(ck["policy"])
            out["pilot-" + path.stem] = evalset(n, hold, hids)
    report["independent_test"] = dict(
        first_seed=ch["first_seed"],
        last_seed=ch["last_seed"],
        bank_sha=held_sha,
        games_with_labels=len(np.unique(gh)),
        scores={k: summary(v) for k, v in out.items()},
    )
    ug, inverse = np.unique(gh, return_inverse=True)
    counts = np.bincount(inverse)
    picks = np.random.default_rng(seed + 49).integers(len(ug), size=(10000, len(ug)))
    report["paired_test"] = {}

    def compare(a, b):
        sums = np.bincount(
            inverse, weights=out[a][:, 0].astype(np.float64) - out[b][:, 0]
        )
        boot = sums[picks].sum(1) / counts[picks].sum(1)
        report["paired_test"][a + " minus " + b] = dict(
            delta=float(sums.sum() / counts.sum()),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        )

    for name in out:
        if name != "mature":
            compare(name, "mature")
    for width, _ in specs:
        compare(f"{width}-fresh", f"{width}-repeat")
    for a, b in [(64, 32), (128, 32), (128, 64)]:
        if f"{a}-fresh" in out and f"{b}-fresh" in out:
            compare(f"{a}-fresh", f"{b}-fresh")
    if compact_controls:
        compare("sketch-fresh", "compact-fresh")
        if token_reference:
            for width in (32, 64, 128):
                for c in ("compact", "sketch"):
                    compare(f"{width}-fresh", f"{c}-fresh")
    if "pilot-pair32" in out:
        for a, b in [
            ("pilot-pair32", "pilot-add32"),
            ("pilot-pair64", "pilot-pair32"),
            ("pilot-pair128", "pilot-pair64"),
        ]:
            if a in out and b in out:
                compare(a, b)
    assert state_digest(model.policy.state_dict()) == teacher
    assert opt_digests == [
        state_digest(model.policy.optimizer.state_dict()),
        state_digest(model.policy.upper_optimizer.state_dict()),
    ]
    report.update(seconds=time.monotonic() - start, complete=True)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "first", "holdout", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--others", type=Path, nargs="+", required=True)
    p.add_argument("--pilot-models", type=Path)
    p.add_argument(
        "--compact-controls",
        action="store_true",
        help="Test the 183 scorer and public sketch at the same data/update budget",
    )
    p.add_argument(
        "--token-reference",
        type=Path,
        help="Evaluate streaming token weights on the common independent bank alongside compact controls",
    )
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--micro", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--continue-arm",
        choices=("32", "64", "128", "sketch"),
        help="Continue a 105-step scorer to a validation plateau",
    )
    p.add_argument("--resume-lower", type=Path)
    p.add_argument("--validation-bank", type=Path)
    p.add_argument("--check-every", type=int, default=60)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.0003)
    p.add_argument("--max-extra", type=int, default=3600)
    p.add_argument(
        "--matched-teacher",
        choices=("uct", "paired"),
        help="Fit aligned counterfactual targets using a game-wise split of one evaluation bank",
    )
    p.add_argument("--labels-root", type=Path)
    args = p.parse_args()
    if args.matched_teacher and (
        not args.continue_arm
        or args.validation_bank != args.holdout
        or not args.labels_root
        or args.max_extra < 360
        or args.check_every != 60
        or args.patience < 2
    ):
        p.error(
            "matched teacher requires a continuation arm, common evaluation bank, aligned labels and a 360-update comparison"
        )
    if args.labels_root and not args.matched_teacher:
        p.error("--labels-root needs --matched-teacher")
    if not 0 < args.steps <= 64 or not 0 < args.micro <= args.batch:
        p.error("invalid update budget")
    if (args.compact_controls and args.pilot_models) or (
        args.token_reference and not args.compact_controls
    ):
        p.error(
            "compact controls accept streaming token references; token arms accept fixed-bank pilots"
        )
    if args.continue_arm:
        if (
            not args.resume_lower
            or not args.validation_bank
            or args.compact_controls
            or args.pilot_models
            or args.token_reference
        ):
            p.error(
                "continuation requires its checkpoint and a separate validation cohort"
            )
        if (
            min(args.check_every, args.patience, args.max_extra) <= 0
            or not np.isfinite(args.min_gain)
            or args.min_gain <= 0
        ):
            p.error("invalid plateau or update budget")
        from tools.analysis.lower_convergence import continue_fit

        continue_fit(args)
    else:
        base_args = vars(args).copy()
        for key in (
            "continue_arm",
            "matched_teacher",
            "labels_root",
            "resume_lower",
            "validation_bank",
            "check_every",
            "patience",
            "min_gain",
            "max_extra",
        ):
            base_args.pop(key)
        experiment(**base_args)
