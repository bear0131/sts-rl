"""Estimate conditional KL noise from independently repeated search targets on identical roots."""

import argparse
import json
from pathlib import Path
import numpy as np


def entropy(p):
    log = np.zeros_like(p, dtype=np.float64)
    np.log(p, out=log, where=p > 0)
    return -(p * log).sum(axis=-1)


def cross_entropy(y, q):
    positive = y > 0
    if np.any(positive & (q <= 0)):
        return float("inf")
    return float(-np.sum(y[positive] * np.log(q[positive])))


def conditional_noise(targets):
    """All risks use KL(teacher || predictor); unknown actions retain their true zero mass."""
    y = np.asarray(targets, np.float64)
    if (
        y.ndim != 2
        or y.shape[0] < 4
        or y.shape[0] % 2
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or not np.allclose(y.sum(1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("need an even set of independent normalized targets")
    n = len(y)
    h = entropy(y)
    mean_h = float(h.mean())
    mu = y.mean(0)
    hm = float(entropy(mu))
    # Entropy of the empirical mixture is downward biased by concavity.
    # Holding a draw out makes its evaluation independent of its fitted predictor.
    loo = (y.sum(0) - y) / (n - 1)
    loo = np.maximum(loo, 0)
    loo_risk = np.mean([cross_entropy(a, b) for a, b in zip(y, loo)]) - mean_h
    first, second = np.split(y, 2)
    half = (
        cross_entropy(second.mean(0), first.mean(0))
        + cross_entropy(first.mean(0), second.mean(0))
    ) / 2 - mean_h
    support = mu > 0
    if np.all(y[:, support] > 0):
        # Including the zero diagonal gives the cross-product of two empirical draws.
        pair = (
            (float(-mu[support] @ np.log(y[:, support]).mean(0)) - mean_h) * n / (n - 1)
        )
    else:
        pair = float("inf")
    corrected = n * hm - (n - 1) * float(entropy(loo).mean()) - mean_h
    out = dict(
        target_h=mean_h,
        mixture_h=hm,
        plug=hm - mean_h,
        jackknife=corrected,
        loo=loo_risk,
        split_half=half,
        pairwise=pair,
    )
    for size in (16, 32, 64):
        if n % size == 0:
            out[f"plug_{size}"] = (
                float(entropy(y.reshape(-1, size, y.shape[1]).mean(1)).mean()) - mean_h
            )
    return out


def run(replicates, aligned, output, predictions=(), seed=20261105):
    with np.load(replicates) as data:
        ids = data["ids"]
        length = data["lens"]
        visits = data["visits"]
        recommend = data["picks"]
    with np.load(aligned) as lab:
        games = lab["games"][ids]
        if not np.array_equal(lab["lengths"][ids], length):
            raise ValueError("survey candidate lengths differ")
    if (
        len(set(ids.tolist())) != len(ids)
        or visits.shape != (2, recommend.shape[-1], int(length.sum()))
        or recommend.shape[:2] != (len(ids), 2)
    ):
        raise ValueError("misaligned independent search roots")
    networks = {}
    for tag, file in predictions:
        with np.load(file) as pred:
            if not np.array_equal(pred["ids"], ids):
                # Models sample the same set in a different order; align by decision identity.
                allids = pred["ids"]
                starts = np.r_[0, pred["lengths"].cumsum()]
                if len(set(allids.tolist())) != len(ids) or set(allids.tolist()) != set(
                    ids.tolist()
                ):
                    raise ValueError("prediction roots differ")
                loc = {int(i): k for k, i in enumerate(allids)}
                logp = np.concatenate(
                    [
                        pred["logp"][starts[loc[int(i)]] : starts[loc[int(i)] + 1]]
                        for i in ids
                    ]
                )
                if not np.array_equal(
                    [pred["lengths"][loc[int(i)]] for i in ids], length
                ):
                    raise ValueError("prediction action lengths differ")
            else:
                if not np.array_equal(pred["lengths"], length):
                    raise ValueError("prediction action lengths differ")
                logp = pred["logp"]
        networks[tag] = logp.astype(np.float64)
    levels = ["uct", "paired"]
    rows = {k: [] for k in levels}
    start = 0
    for i, L in enumerate(length):
        L = int(L)
        sl = slice(start, start + L)
        start += L
        for j, label in enumerate(levels):
            counts = visits[j, :, sl].astype(np.float64)
            if np.any(counts.sum(1) == 0):
                raise ValueError("empty teacher root")
            picks = recommend[i, j]
            if np.any((picks < 0) | (picks >= L)):
                raise ValueError("invalid recommendation")
            y = 0.2 * (counts / counts.sum(1, keepdims=True))
            y[np.arange(len(y)), picks] += 0.8
            row = conditional_noise(y)
            for tag, logp in networks.items():
                pred = logp[sl]
                if not np.isfinite(pred).all() or abs(np.exp(pred).sum() - 1) > 2e-5:
                    raise ValueError("invalid fitted distribution")
                row[tag + "_risk"] = float(-y.mean(0) @ pred) - row["target_h"]
                row[tag + "_excess"] = row[tag + "_risk"] - row["jackknife"]
                row[tag + "_top1"] = float(np.mean(picks == pred.argmax()))
            rows[label].append(row)
    unique, clusters = np.unique(games, return_inverse=True)
    n = np.bincount(clusters)
    selections = np.random.default_rng(seed).integers(
        len(unique), size=(10000, len(unique))
    )
    den = n[selections].sum(1)

    def summarize(v):
        v = np.asarray(v, np.float64)
        if not np.isfinite(v).all():
            return dict(
                mean=None, ci95=None, nonfinite_roots=int(np.sum(~np.isfinite(v)))
            )
        sums = np.bincount(clusters, weights=v)
        return dict(
            mean=float(v.mean()),
            ci95=np.quantile(sums[selections].sum(1) / den, [0.025, 0.975]).tolist(),
        )

    out = dict(
        roots=len(ids),
        games=len(unique),
        repeats=int(recommend.shape[-1]),
        visits_fraction=0.2,
        seed=seed,
        teachers={},
        paired_minus_uct={},
    )
    for label in levels:
        out["teachers"][label] = {
            k: summarize([r[k] for r in rows[label]]) for k in rows[label][0]
        }
    for k in rows["uct"][0]:
        a = np.array([r[k] for r in rows["paired"]])
        b = np.array([r[k] for r in rows["uct"]])
        if np.isfinite(a).all() and np.isfinite(b).all():
            out["paired_minus_uct"][k] = summarize(a - b)
    output.write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicates", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--prediction", nargs=2, action="append", default=[], metavar=("NAME", "PATH")
    )
    a = p.parse_args()
    run(a.replicates, a.aligned, a.output, a.prediction)
