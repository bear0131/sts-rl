"""Load frozen game banks with aligned public entities and verified teacher identity."""

import hashlib
import json

import numpy as np
import torch

from rl.policies.entity_action_playout import ActionTokenBank
from rl.policies.public_playout import public_sketch
from rl.policies.small_playout import PreparedPlayoutSamples


class SketchBank:
    def __init__(self, shards, base, device):
        self.base = base
        self.context = torch.as_tensor(
            np.concatenate([public_sketch(t, t["lengths"]) for t in shards]),
            device=device,
        )

    def batch(self, ids, device):
        args, y, teacher = self.base.batch(ids, device)
        return (*args[:-1], self.context[ids], args[-1]), y, teacher


def prepare_bank(bank, source, device, teacher_digest=None, *, sketch=False):
    collection = json.loads((bank / "collection.json").read_text())
    with (bank / "samples.pt").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != collection["samples_sha256"]:
            raise ValueError("sample digest mismatch")
    payload = torch.load(bank / "samples.pt", weights_only=False)
    with (source / "model.zip").open("rb") as f:
        matching_zip = (
            hashlib.file_digest(f, "sha256").hexdigest()
            == payload["source_model_sha256"]
        )
    canonical = collection.get("teacher_digest") or collection.get(
        "state_digests", {}
    ).get("policy")
    if (
        canonical is not None
        and teacher_digest is not None
        and canonical != teacher_digest
    ) or (not matching_zip and (teacher_digest is None or canonical != teacher_digest)):
        raise ValueError("wrong frozen teacher for this bank")
    rows = payload["samples"]
    base = PreparedPlayoutSamples(rows, 0.2)
    games = np.asarray([r["env"] for r in rows])
    if (games[1:] < games[:-1]).any():
        raise ValueError("bank must preserve per-game decision order")
    digest = hashlib.sha256()

    def shards():
        for g in np.unique(games):
            file = bank / "public" / f"{g:03d}.npz"
            with file.open("rb") as f:
                digest.update(hashlib.file_digest(f, "sha256").digest())
            with np.load(file) as sh:
                t = {k: sh[k] for k in sh.files}
                if not np.array_equal(t["alens"], base.lengths[games == g]):
                    raise ValueError("public/action mismatch")
                yield t

    if sketch:
        data = SketchBank(shards(), base, device)
        return payload, base, data, games, digest.hexdigest(), None, collection
    data = ActionTokenBank(shards(), base)
    seen_symbols = (
        len(set(np.unique(data.symbols)) | set(np.unique(data.extra_symbols))) - 1
    )
    data.cache_features(device)
    return payload, base, data, games, digest.hexdigest(), seen_symbols, collection
