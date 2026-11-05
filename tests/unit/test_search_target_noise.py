import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tools.analysis.lower_bank import install_paired_labels
from tools.analysis.search_target_noise import conditional_noise, entropy


def test_conditional_mean_is_the_best_fixed_prediction():
    targets = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    pred = np.array([0.35, 0.65])

    def kl(a, b):
        return (a * np.log(a / b)).sum()

    noise = conditional_noise(targets)
    risk = np.mean([kl(y, pred) for y in targets])
    assert risk == pytest.approx(noise["plug"] + kl(targets.mean(0), pred))
    self_kl = np.mean(
        [
            kl(a, b)
            for i, a in enumerate(targets)
            for j, b in enumerate(targets)
            if i != j
        ]
    )
    assert noise["pairwise"] == pytest.approx(self_kl)
    assert noise["plug"] < noise["loo"] and noise["plug"] < noise["pairwise"]
    fixed = np.repeat([[0.75, 0.25]], 8, axis=0)
    assert conditional_noise(fixed)["jackknife"] == pytest.approx(0, abs=1e-12)


def test_exclusive_worldwise_actions_preserve_infinite_pairwise_kl():
    targets = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    noise = conditional_noise(targets)
    assert np.isinf(noise["pairwise"]) and np.isinf(noise["split_half"])
    assert noise["plug"] == pytest.approx(np.log(2))
    assert noise["target_h"] == 0.0 and entropy(np.array([[0.0, 1.0]])).item() == 0.0


def test_paired_targets_preserve_candidate_order_and_mix(tmp_path):
    base = SimpleNamespace(
        lengths=np.array([2, 3]),
        starts=np.array([0, 2]),
        targets=np.zeros(5),
        teacher=None,
    )
    bank = SimpleNamespace(base=base)
    games = np.array([0, 4])
    path = tmp_path / "labels.npz"
    np.savez(
        path, visits=[1, 3, 2, 2, 4], teacher=[0, 2], lengths=base.lengths, games=games
    )
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "meta.json").write_text(
        json.dumps(dict(bank_sha="checked", labels_sha=h))
    )
    assert install_paired_labels(bank, games, tmp_path, "checked") == h
    np.testing.assert_allclose(base.targets, [0.85, 0.15, 0.05, 0.05, 0.9], atol=1e-7)
    np.testing.assert_array_equal(base.teacher, [0, 2])
    with pytest.raises(ValueError, match="alignment"):
        install_paired_labels(bank, games[::-1], tmp_path, "checked")
