"""Continue a lower-only data experiment until held-out KL reaches a plateau."""

from dataclasses import dataclass
import gc
import json
import time

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_probe import state_digest
from rl.policies.entity_action_playout import ActionTokenPolicy
from rl.policies.public_playout import PublicPlayoutPolicy
from rl.training.resource_monitor import ResourceMonitor
from tools.analysis.lower_bank import install_paired_labels, prepare_bank


@dataclass
class Plateau:
    """A small improvement can accumulate across checks before patience resets."""

    reference: float
    min_gain: float
    patience: int
    stale: int = 0
    drops: int = 0

    def observe(self, kl):
        if not np.isfinite(kl):
            raise ValueError("validation KL must be finite")
        if self.reference - kl >= self.min_gain:
            self.reference, self.stale = kl, 0
            return "continue"
        self.stale += 1
        if self.stale < self.patience:
            return "continue"
        if self.drops == 2:
            return "converged"
        self.drops += 1
        self.stale = 0
        return "reduce"


def batches(sets, batch, seed):
    """Shuffle each complete cohort without replacement per pass; interleave batches."""
    rng = np.random.default_rng(seed)
    while True:
        chunks = []
        for bank_number, indices in enumerate(sets):
            perm = rng.permutation(indices)
            chunks.extend(
                (bank_number, perm[j : j + batch]) for j in range(0, len(perm), batch)
            )
        for position in rng.permutation(len(chunks)):
            yield chunks[position]


def write_report(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def continue_fit(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clock = time.monotonic
    started = clock()
    model = AccumulatingMaskablePPO.load(
        args.parent / "model.zip", device=device, amp_mode="off"
    )
    teacher = state_digest(model.policy.state_dict())
    parent = model.policy.small_playout
    parent_digest = state_digest(parent.state_dict())
    fixed_adam = [
        state_digest(o.state_dict())
        for o in (model.policy.optimizer, model.policy.upper_optimizer)
    ]
    saved = torch.load(args.resume_lower, weights_only=True, map_location=device)
    if (
        saved["teacher_digest"] != teacher
        or saved["parent_lower_digest"] != parent_digest
        or saved["updates"] != 105
    ):
        raise ValueError("expected the matching 105-update fresh-data checkpoint")
    sketch = args.continue_arm == "sketch"
    if sketch:
        if saved["width"] != "sketch":
            raise ValueError("expected the linear public scorer")
        net = PublicPlayoutPolicy.from_mature(parent)
    else:
        width = int(args.continue_arm)
        if (
            saved["width"] != width
            or not saved["coupled"]
            or saved["depth"] != {32: 1, 64: 2, 128: 3}[width]
        ):
            raise ValueError("token architecture mismatch")
        net = ActionTokenPolicy.from_mature(parent, True, width, saved["depth"])
    net.load_state_dict(saved["policy"])
    opt = torch.optim.AdamW(net.parameters())
    opt.load_state_dict(saved["optimizer"])
    base_lr = opt.param_groups[0]["lr"]
    initial_updates = int(saved["updates"])
    paths = [args.first, *args.others, args.validation_bank, args.holdout]
    matched = args.matched_teacher is not None
    used_seeds = set()
    for path in paths[:-1] if matched else paths:
        c = json.loads((path / "collection.json").read_text())["cohort"]
        ids = set(range(c["first_seed"], c["last_seed"] + 1))
        if (
            len(ids) != 480
            or c["completed_games"] != 480
            or c["censored_games"]
            or ids & used_seeds
        ):
            raise ValueError("require disjoint, complete 480-game banks")
        used_seeds |= ids

    override_digests = {}

    def load(path):
        payload, _, bank, games, digest, _, _meta = prepare_bank(
            path, args.parent, "cpu", teacher_digest=teacher, sketch=sketch
        )
        del payload
        if args.matched_teacher == "paired":
            override_digests[path.name] = install_paired_labels(
                bank, games, args.labels_root / path.name, _meta["samples_sha256"]
            )
        tensors = (
            [bank.context]
            if sketch
            else [
                getattr(bank, k)
                for k in ("features", "symbols", "extras", "extra_symbols")
            ]
        )
        size = sum(x.numel() * x.element_size() for x in tensors)
        if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] > size + 8 * (
            1024**3
        ):
            if sketch:
                bank.context = bank.context.to(device)
            else:
                bank.cache_features(device)
        return bank, games, digest

    def ce(n, bank, idx):
        inputs, y, teacher_action = bank.batch(idx, device)
        logits = n.forward_packed(*inputs)
        lp = logits.log_softmax(-1).masked_fill(~inputs[-1], 0.0)
        return -(y * lp).sum(-1), y, teacher_action, logits

    @torch.no_grad()
    def predict(n, bank, ids):
        rows = []
        for j in range(0, len(ids), args.micro):
            loss, y, action, logits = ce(n, bank, ids[j : j + args.micro])
            kl = loss + (y * y.clamp_min(1e-30).log()).sum(-1)
            rows.append(
                torch.stack((kl, logits.argmax(-1) == action), dim=1).cpu().numpy()
            )
        return np.concatenate(rows)

    valid, vg, vsig = load(args.validation_bank)
    validation_ids = np.flatnonzero(vg % 5 == 0) if matched else np.arange(len(vg))
    vi = np.random.default_rng(args.seed + 43).permutation(validation_ids)[
        : min(65536, len(vg))
    ]
    banks = []
    sets = []
    digests = []
    for i, path in enumerate(paths[:-2]):
        bank, g, digest = load(path)
        ids = (
            np.flatnonzero(g % 5 != 0) if i == 0 and not matched else np.arange(len(g))
        )
        banks.append(bank)
        sets.append(ids)
        digests.append(digest)
        print(json.dumps(dict(loaded=str(path), samples=len(ids))), flush=True)
    del bank
    train_probes = [
        np.random.default_rng(args.seed + 8000 + i).choice(ids, 1024, replace=False)
        for i, ids in enumerate(sets)
    ]
    stream = batches(sets, args.batch, args.seed + 105)

    def measured(n):
        val = predict(n, valid, vi).mean(0, dtype=np.float64)
        train = np.concatenate(
            [predict(n, b, probe) for b, probe in zip(banks, train_probes)]
        ).mean(0, dtype=np.float64)
        return dict(
            validation_kl=float(val[0]),
            validation_top1=float(val[1]),
            train_kl=float(train[0]),
        )

    def save(path, lineage):
        tmp = path.with_suffix(".tmp")
        torch.save(
            dict(
                policy=net.state_dict(),
                optimizer=opt.state_dict(),
                lineage=lineage,
                teacher_digest=teacher,
                parent_lower_digest=parent_digest,
                width=saved["width"],
                depth=saved["depth"],
                coupled=saved["coupled"],
            ),
            tmp,
        )
        tmp.replace(path)

    start_metric = measured(net)
    best = start_metric["validation_kl"]
    best_at = initial_updates
    lineage = initial_updates
    schedule = Plateau(best, args.min_gain, args.patience)
    save(output / "best.pt", lineage)
    progress = dict(
        arm=args.continue_arm,
        teacher=teacher,
        seed=args.seed,
        batch=args.batch,
        micro=args.micro,
        validation_digest=vsig,
        label_source=args.matched_teacher or "original",
        label_digests=override_digests,
        validation_decisions=len(vi),
        banks=[str(x) for x in paths[:-2]],
        bank_digests=digests,
        train_sizes=list(map(len, sets)),
        min_gain=args.min_gain,
        patience=args.patience,
        check_every=args.check_every,
        max_extra=args.max_extra,
        curve=[dict(updates=initial_updates, lr=base_lr, **start_metric)],
    )
    shown = 0
    with ResourceMonitor(
        interval=1.0, gpu=device.type == "cuda", torch_device=device
    ) as monitor:
        for extra in range(1, args.max_extra + 1):
            i, ids = next(stream)
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(ids), args.micro):
                loss, _, _, _ = ce(net, banks[i], ids[j : j + args.micro])
                (loss.sum() / len(ids)).backward()
            opt.step()
            lineage += 1
            shown += len(ids)
            if extra % args.check_every != 0 and extra != args.max_extra:
                continue
            now = initial_updates + extra
            if matched and extra == 360:
                save(output / "equal-budget.pt", lineage)
            current = measured(net)
            value = current["validation_kl"]
            lr = opt.param_groups[0]["lr"]
            if value < best:
                best = value
                best_at = now
                save(output / "best.pt", lineage)
            status = schedule.observe(value)
            row = dict(
                updates=now,
                lineage=lineage,
                lr=lr,
                **current,
                best_kl=best,
                best_at=best_at,
                action=status,
                used_examples=shown,
                elapsed=clock() - started,
            )
            progress["curve"].append(row)
            write_report(output / "progress.json", progress)
            print(json.dumps(row), flush=True)
            if status == "converged":
                break
            if status == "reduce":
                selected = torch.load(
                    output / "best.pt", weights_only=True, map_location=device
                )
                net.load_state_dict(selected["policy"])
                opt.load_state_dict(selected["optimizer"])
                lineage = selected["lineage"]
                for group in opt.param_groups:
                    group["lr"] = base_lr * (0.3**schedule.drops)
                schedule.reference = best
        save(output / "last.pt", lineage)
    progress.update(
        stop=status if status == "converged" else "budget",
        resources=monitor.summary,
        attempted_updates=now,
        selected_at=best_at,
        prepared_and_fit_seconds=clock() - started,
    )
    banks.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected = torch.load(output / "best.pt", weights_only=True, map_location=device)
    net.load_state_dict(selected["policy"])
    progress["full_validation_kl"] = float(
        predict(net, valid, validation_ids)[:, 0].mean(dtype=np.float64)
    )
    valid = None
    test, tg, tsig = load(args.holdout)
    ids = np.flatnonzero(tg % 5 != 0) if matched else np.arange(len(tg))
    if matched and set(tg[ids]).intersection(vg[vi]):
        raise ValueError("validation games overlap final test games")
    final = predict(net, test, ids)
    if matched:
        survey = np.random.default_rng(20261105731).choice(ids, 384, replace=False)
        with torch.no_grad():
            packed = []
            for j in range(0, len(survey), args.micro):
                subset = survey[j : j + args.micro]
                inputs, _, _ = test.batch(subset, device)
                logp = net.forward_packed(*inputs).log_softmax(-1)
                packed.append(logp[inputs[-1]].cpu().numpy())
        np.savez_compressed(
            output / "survey-logp.npz",
            ids=survey,
            lengths=test.base.lengths[survey],
            logp=np.concatenate(packed),
        )
        compare = torch.load(
            output / "equal-budget.pt", weights_only=True, map_location=device
        )
        net.load_state_dict(compare["policy"])
        fixed_scores = predict(net, test, ids)
    net.load_state_dict(saved["policy"])
    initial = predict(net, test, ids)
    groups, inverse = np.unique(tg[ids], return_inverse=True)
    count = np.bincount(inverse)
    stats = {"counts": count.tolist(), "games": groups.tolist()}
    mature_scores = predict(parent, test.base, ids)
    for name, rows in [
        ("mature", mature_scores),
        ("step105", initial),
        ("selected", final),
    ]:
        stats[name] = dict(
            kl=float(rows[:, 0].mean(dtype=np.float64)),
            top1=float(rows[:, 1].mean(dtype=np.float64)),
            kl_sum=np.bincount(inverse, weights=rows[:, 0]).tolist(),
            top1_count=np.bincount(inverse, weights=rows[:, 1]).astype(int).tolist(),
        )
    if matched:
        y = test.base.targets
        ent = np.add.reduceat(-(y * np.log(np.maximum(y, 1e-30))), test.base.starts)[
            ids
        ]
        stats["target_entropy"] = float(ent.mean(dtype=np.float64))
        stats["selected_ce"] = stats["selected"]["kl"] + stats["target_entropy"]
        stats["equal_budget"] = dict(
            extra_updates=360,
            kl=float(fixed_scores[:, 0].mean(dtype=np.float64)),
            top1=float(fixed_scores[:, 1].mean(dtype=np.float64)),
            kl_sum=np.bincount(inverse, weights=fixed_scores[:, 0]).tolist(),
            top1_count=np.bincount(inverse, weights=fixed_scores[:, 1])
            .astype(int)
            .tolist(),
        )
    diff = final[:, 0].astype(np.float64) - initial[:, 0]
    sums = np.bincount(inverse, weights=diff)
    picks = np.random.default_rng(args.seed + 49).integers(
        len(groups), size=(10000, len(groups))
    )
    top_change = final[:, 1].astype(np.float64) - initial[:, 1]
    top_sums = np.bincount(inverse, weights=top_change)
    progress["test"] = dict(
        top1_delta=float(top_change.mean()),
        top1_ci95=np.quantile(
            top_sums[picks].sum(1) / count[picks].sum(1), [0.025, 0.975]
        ).tolist(),
        digest=tsig,
        samples=len(ids),
        delta=float(diff.mean()),
        ci95=np.quantile(
            sums[picks].sum(1) / count[picks].sum(1), [0.025, 0.975]
        ).tolist(),
        **stats,
    )
    if teacher != state_digest(model.policy.state_dict()) or fixed_adam != [
        state_digest(o.state_dict())
        for o in (model.policy.optimizer, model.policy.upper_optimizer)
    ]:
        raise ValueError("teacher or upper optimizer changed")
    progress.update(complete=True, seconds=clock() - started)
    write_report(output / "result.json", progress)
