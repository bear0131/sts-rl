"""Round-183 fixed-bank capacity × training-duration experiment for token action heads."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_probe import state_digest
from rl.policies.entity_action_playout import ActionTokenPolicy
from tools.analysis.lower_bank import prepare_bank
from rl.training.resource_monitor import ResourceMonitor


def run(
    bank,
    source,
    output,
    seed,
    device,
    short_epochs,
    long_epochs,
    micro,
    batch,
    learning_rate,
    arms=None,
):
    output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clock = time.perf_counter
    start = clock()
    payload, base, data, games, bank_sha, seen_symbols, collection = prepare_bank(
        bank, source, device
    )
    tr = np.flatnonzero(games % 5 != 0)
    va = np.flatnonzero(games % 10 == 0)
    te = np.flatnonzero(games % 10 == 5)
    model = AccumulatingMaskablePPO.load(
        source / "model.zip", device=device, amp_mode="off"
    )
    mature = model.policy.small_playout
    fixed = lambda: [
        state_digest(
            {
                k: v
                for k, v in model.policy.state_dict().items()
                if not k.startswith("small_playout.")
            }
        ),
        state_digest(model.policy.optimizer.state_dict()),
        state_digest(model.policy.upper_optimizer.state_dict()),
    ]
    original = fixed()

    def losses(net, idx):
        inputs, y, teacher = (
            data if isinstance(net, ActionTokenPolicy) else base
        ).batch(idx, device)
        logits = net.forward_packed(*inputs)
        logp = logits.log_softmax(-1).masked_fill(~inputs[-1], 0.0)
        return (
            -(y * logp).sum(-1),
            -(y * y.clamp_min(1e-30).log()).sum(-1),
            logits.argmax(-1) == teacher,
        )

    @torch.no_grad()
    def evaluate(net, idx):
        return np.concatenate(
            [
                torch.stack(losses(net, idx[j : j + micro]), 1).cpu().numpy()
                for j in range(0, len(idx), micro)
            ]
        )

    def metric(arr):
        ce, entropy, top = arr.mean(0, dtype=np.float64)
        return dict(ce=ce, kl=ce - entropy, top1=top, entropy=entropy, samples=len(arr))

    old_val = evaluate(mature, va)
    old_test = evaluate(mature, te)
    report = dict(
        source_sha=payload["source_model_sha256"],
        samples_sha=collection["samples_sha256"],
        public_sha=bank_sha,
        seen_symbol_ids=seen_symbols,
        seed=seed,
        device=device,
        effective_batch=batch,
        micro=micro,
        lr=learning_rate,
        training_games=len(np.unique(games[tr])),
        validation_games=len(np.unique(games[va])),
        test_games=len(np.unique(games[te])),
        baseline=dict(
            train=metric(evaluate(mature, tr)),
            validation=metric(old_val),
            test=metric(old_test),
        ),
        arms=[],
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "baseline": report["baseline"],
                "seconds": clock() - start,
            }
        ),
        flush=True,
    )
    chosen = (metric(old_val)["kl"], None, None)
    for name, width, depth, coupled, epochs in [
        ("add32", 32, 1, False, short_epochs),
        ("pair32", 32, 1, True, long_epochs),
        ("pair64", 64, 2, True, short_epochs),
        ("pair128", 128, 3, True, short_epochs),
    ]:
        if arms is not None and name not in arms:
            continue
        arm_start = clock()
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        net = ActionTokenPolicy.from_mature(mature, coupled, width, depth)
        opt = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.0001)
        check = evaluate(net, va)
        if not np.allclose(check, old_val, atol=2e-6, rtol=0):
            raise ValueError("initial migration changed mature predictions")
        params = sum(p.numel() for p in net.parameters())
        embeddings = sum(
            mod.weight.numel()
            for mod in net.modules()
            if isinstance(mod, torch.nn.Embedding)
        )
        arm = dict(
            name=name,
            width=width,
            depth=depth,
            coupled=coupled,
            epochs=epochs,
            params=params,
            embedding_params=embeddings,
            nonembedding_params=params - embeddings,
            curve=[],
        )
        report["arms"].append(arm)
        best = (float("inf"), None)
        with ResourceMonitor(
            interval=1.0, gpu=device.startswith("cuda"), torch_device=model.device
        ) as monitor:
            for ep in range(1, epochs + 1):
                order = rng.permutation(tr)
                for k in range(0, len(order), batch):
                    group = order[k : k + batch]
                    opt.zero_grad(set_to_none=True)
                    for j in range(0, len(group), micro):
                        loss, _, _ = losses(net, group[j : j + micro])
                        (loss.sum() / len(group)).backward()
                    opt.step()
                if ep not in {
                    1,
                    2,
                    4,
                    8,
                    12,
                    16,
                    24,
                    32,
                    48,
                    64,
                    short_epochs,
                    long_epochs,
                }:
                    continue
                val = evaluate(net, va)
                row = dict(
                    epoch=ep,
                    updates=ep * int(np.ceil(len(tr) / batch)),
                    train=metric(evaluate(net, tr)),
                    validation=metric(val),
                    seconds=clock() - arm_start,
                )
                arm["curve"].append(row)
                if row["validation"]["kl"] < best[0]:
                    best = (row["validation"]["kl"], ep)
                    torch.save(
                        dict(
                            policy=net.state_dict(),
                            optimizer=opt.state_dict(),
                            parent_lower_digest=state_digest(mature.state_dict()),
                            source_sha256=payload["source_model_sha256"],
                            width=width,
                            depth=depth,
                            coupled=coupled,
                            epoch=ep,
                        ),
                        output / (name + ".pt"),
                    )
                if best[0] < chosen[0]:
                    chosen = (best[0], name, best[1])
                (output / "progress.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False)
                )
                print(json.dumps(dict(arm=name, **row)), flush=True)
        arm["resources"] = monitor.summary
        arm["selected_epoch"] = best[1]
        arm["seconds"] = clock() - arm_start
    # Held-out test is scored only after all model/epoch choices are fixed.
    for arm in report["arms"]:
        ck = torch.load(
            output / (arm["name"] + ".pt"), weights_only=True, map_location=device
        )
        net = ActionTokenPolicy.from_mature(
            mature, ck["coupled"], ck["width"], ck["depth"]
        )
        net.load_state_dict(ck["policy"])
        pred = evaluate(net, te)
        arm["selected_test"] = metric(pred)
        delta = pred[:, 0].astype(np.float64) - old_test[:, 0]
        ug = np.unique(games[te])
        sums = np.array([delta[games[te] == g].sum() for g in ug])
        counts = np.array([(games[te] == g).sum() for g in ug])
        pick = np.random.default_rng(seed + 1).integers(len(ug), size=(10000, len(ug)))
        arm["test_delta_ci95"] = np.quantile(
            sums[pick].sum(1) / counts[pick].sum(1), [0.025, 0.975]
        ).tolist()
        np.save(output / (arm["name"] + "-test.npy"), pred)
    assert fixed() == original
    report.update(selected=chosen, seconds=clock() - start, nonlower_unchanged=True)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("bank", "source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--short-epochs", type=int, default=16)
    p.add_argument("--long-epochs", type=int, default=32)
    p.add_argument("--micro", type=int, default=256)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument(
        "--arms", nargs="+", choices=("add32", "pair32", "pair64", "pair128")
    )
    args = p.parse_args()
    if not (
        1 <= args.short_epochs <= args.long_epochs <= 64
        and 1 <= args.micro <= args.batch
    ):
        p.error("invalid fit budget")
    run(**vars(args))
