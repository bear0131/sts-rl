"""Compare frozen compact-policy distillation targets on paired independent action values."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_probe import state_digest
from rl.policies.small_playout import PreparedPlayoutSamples


def read_banks(paths, label_paths, teacher):
    records, targets, game, hp, boss, reference, provenance = [], [], [], [], [], [], []
    for path, labels in zip(paths, label_paths, strict=True):
        meta = json.loads((path / "collection.json").read_text())
        verified = json.loads((labels / "verified.json").read_text())
        with (path / "samples.pt").open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if (
            digest != verified["bank"]
            or digest != meta["samples_sha256"]
            or teacher != meta["teacher_digest"]
        ):
            raise ValueError("label bank or frozen teacher differs")
        samples = torch.load(path / "samples.pt", weights_only=False)["samples"]
        with np.load(labels / "labels.npz") as data:
            ids = data["ids"]
            subset = [samples[i] for i in ids]
            if (
                len(np.unique(ids)) != len(ids)
                or (ids < 0).any()
                or not np.array_equal(
                    data["sizes"], [len(x["action_bits"]) for x in subset]
                )
                or not np.array_equal(data["game"], [x["env"] for x in subset])
            ):
                raise ValueError("decision identity or legal action alignment differs")
            game.append(data["game"] + meta["cohort"]["first_seed"])
            hp.append(data["hp"])
            boss.append(data["boss"])
            targets.append(
                [data["old_target"], data["return_target"], data["coverage"]]
            )
            reference.append(data["ref_q"])
        with (labels / "labels.npz").open("rb") as f:
            provenance.append([digest, hashlib.file_digest(f, "sha256").hexdigest()])
        records.extend(subset)
        del samples
    bank = PreparedPlayoutSamples(records, 0.2)
    overlays = {
        name: np.concatenate([p[i] for p in targets])
        for i, name in enumerate(("recommended", "returns"))
    }
    for y in overlays.values():
        if (
            y.shape != bank.targets.shape
            or not np.isfinite(y).all()
            or (y < 0).any()
            or not np.allclose(np.add.reduceat(y, bank.starts), 1.0, atol=2e-6, rtol=0)
        ):
            raise ValueError("invalid action distribution")
    bank.coverage = np.concatenate([p[2] for p in targets]).astype(bool)
    return (
        bank,
        overlays,
        np.concatenate(game),
        np.concatenate(hp),
        np.concatenate(boss),
        np.concatenate(reference),
        provenance,
    )


@torch.no_grad()
def assess(net, bank, labels, reference, ids, device):
    totals = []
    for start in range(0, len(ids), 512):
        part = ids[start : start + 512]
        inputs, _, _ = bank.batch(part, device)
        logp = net.forward_packed(*inputs).log_softmax(-1)
        mask = inputs[-1].cpu().numpy()
        flat = logp[inputs[-1]].cpu().numpy().astype(np.float64)
        lengths = bank.lengths[part]
        ends = np.cumsum(lengths)
        begins = ends - lengths
        source = np.repeat(bank.starts[part] - begins, lengths) + np.arange(ends[-1])
        q = reference[source]
        probabilities = np.exp(flat)
        expected = np.add.reduceat(q * probabilities, begins)
        covered = bank.coverage[source]
        supervised_flat = flat - np.repeat(
            np.log(np.add.reduceat(probabilities * covered, begins)), lengths
        )
        choices = logp.argmax(-1).cpu().numpy()
        greedy = q[begins + choices]
        assert np.array_equal(lengths, mask.sum(1))
        scores = [expected, greedy]
        for y in labels.values():
            p = y[source].astype(np.float64)
            scores.append(
                np.add.reduceat(
                    p * (np.log(np.maximum(p, 1e-300)) - supervised_flat), begins
                )
            )
        totals.append(np.stack(scores, axis=1))
    return np.concatenate(totals)


def rescale_target(bank, target, power=1.0, log_prior=None):
    covered = target > 0
    logits = np.full(len(target), -np.inf)
    logits[covered] = power * np.log(target[covered].astype(np.float64))
    if log_prior is not None:
        logits += log_prior
    maxlog = np.maximum.reduceat(logits, bank.starts)
    values = np.exp(logits - np.repeat(maxlog, bank.lengths))
    return (
        values / np.repeat(np.add.reduceat(values, bank.starts), bank.lengths)
    ).astype(np.float32)


@torch.no_grad()
def anchored_comparison(net, bank, y, device, sharpness=None):
    logprior = []
    ids = np.arange(len(bank.lengths))
    for start in range(0, len(ids), 512):
        inputs, _, _ = bank.batch(ids[start : start + 512], device)
        lp = net.forward_packed(*inputs).log_softmax(-1)
        logprior.append(lp[inputs[-1]].cpu().numpy().astype(np.float64))
    anchored = rescale_target(bank, y, log_prior=np.concatenate(logprior))

    def mean_entropy(t):
        pos = t[t > 0].astype(np.float64)
        return float(-(pos * np.log(pos)).sum() / len(bank.lengths))

    if sharpness is None:
        desired = mean_entropy(anchored)
        lo, hi = 1.0, 64.0
        for _ in range(24):
            mid = (lo + hi) / 2
            if mean_entropy(rescale_target(bank, y, mid)) > desired:
                lo = mid
            else:
                hi = mid
        sharpness = (lo + hi) / 2
    sharp = rescale_target(bank, y, sharpness)
    return (
        {"sharpened": sharp, "anchored": anchored},
        sharpness,
        dict(
            returns=mean_entropy(y),
            sharpened=mean_entropy(sharp),
            anchored=mean_entropy(anchored),
        ),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    parent = AccumulatingMaskablePPO.load(
        args.checkpoint / "model.zip", device="cpu", amp_mode="off"
    )
    teacher = state_digest(parent.policy.state_dict())
    model = parent.policy.small_playout
    train, labels, train_games, _, _, _, train_sha = read_banks(
        args.banks[:-1], args.labels[:-1], teacher
    )
    test, both, games, hp, boss, ref, test_sha = read_banks(
        args.banks[-1:], args.labels[-1:], teacher
    )
    offset = json.loads((args.banks[-1] / "collection.json").read_text())["cohort"][
        "first_seed"
    ]
    validation = np.flatnonzero((games - offset) % 5 == 0)
    holdout = np.flatnonzero((games - offset) % 5 != 0)
    if (
        np.intersect1d(train_games, games).size
        or not validation.size
        or not holdout.size
    ):
        raise ValueError("independent training, validation and test games required")
    result = dict(
        teacher=teacher,
        train_roots=len(train_games),
        validation_roots=len(validation),
        test_roots=len(holdout),
        train_banks=train_sha,
        test_bank=test_sha,
        metrics=["expected_q", "greedy_q", "old_target_kl", "return_target_kl"],
        fits={},
    )
    predictions = {}
    init = copy.deepcopy(model).to(device)
    if args.anchored_comparison:
        labels, power, train_entropy = anchored_comparison(
            init, train, labels["returns"], device
        )
        both, _, test_entropy = anchored_comparison(
            init, test, both["returns"], device, power
        )
        result.update(
            sharpness=power,
            train_entropy=train_entropy,
            test_entropy=test_entropy,
            metrics=[
                "expected_q",
                "greedy_q",
                "sharpened_target_kl",
                "anchored_target_kl",
            ],
        )
    predictions["initial"] = assess(init, test, both, ref, holdout, device)
    for seed in args.seeds:
        for name, y in labels.items():
            tag = f"{name}-{seed}"
            net = copy.deepcopy(init)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0001)
            rng = np.random.default_rng(seed)
            train.targets = y
            best_value = -np.inf
            curve = []
            for epoch in range(1, args.epochs + 1):
                order = rng.permutation(len(train_games))
                net.train()
                for start in range(0, len(order), args.batch):
                    batch = order[start : start + args.batch]
                    opt.zero_grad(set_to_none=True)
                    for j in range(0, len(batch), args.micro):
                        inputs, target, _ = train.batch(
                            batch[j : j + args.micro], device
                        )
                        part = batch[j : j + args.micro]
                        lengths = train.lengths[part]
                        ends = lengths.cumsum()
                        source = np.repeat(
                            train.starts[part] - (ends - lengths), lengths
                        ) + np.arange(ends[-1])
                        observed = torch.zeros_like(inputs[-1])
                        observed[inputs[-1]] = torch.as_tensor(
                            train.coverage[source], device=device
                        )
                        logits = net.forward_packed(*inputs).masked_fill(
                            ~observed, -torch.inf
                        )
                        logp = logits.log_softmax(-1).masked_fill(~observed, 0.0)
                        (-(target * logp).sum() / len(batch)).backward()
                    opt.step()
                net.eval()
                val = assess(net, test, both, ref, validation, device).mean(0)
                curve.append(val.tolist())
                if val[0] > best_value:
                    best_value = val[0]
                    best_epoch = epoch
                    torch.save(net.state_dict(), args.output / (tag + ".pt"))
            net.load_state_dict(
                torch.load(
                    args.output / (tag + ".pt"), map_location=device, weights_only=True
                )
            )
            pred = assess(net, test, both, ref, holdout, device)
            predictions[tag] = pred
            result["fits"][tag] = dict(
                epoch=best_epoch, val_curve=curve, test_mean=pred.mean(0).tolist()
            )
            print(tag, best_epoch, pred.mean(0), flush=True)
    np.savez_compressed(
        args.output / "paired-test.npz",
        games=games[holdout],
        boss=boss[holdout],
        hp=hp[holdout],
        **predictions,
    )
    result["spec"] = dict(
        epochs=args.epochs,
        batch=args.batch,
        micro=args.micro,
        lr=args.lr,
        seeds=args.seeds,
    )
    (args.output / "result.json").write_text(
        json.dumps(result, allow_nan=False, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--banks", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261105, 20261117])
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--micro", type=int, default=512)
    p.add_argument("--anchored-comparison", action="store_true")
    args = p.parse_args()
    if (
        len(args.banks) < 2
        or len(args.labels) != len(args.banks)
        or args.epochs < 1
        or not 0 < args.micro <= args.batch
        or not np.isfinite(args.lr)
        or args.lr <= 0
    ):
        p.error("need aligned independent banks and finite training budget")
    run(args)
