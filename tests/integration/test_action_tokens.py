import hashlib
import json

import numpy as np
import pytest
import torch
import env.core
import slaythespire as sts
from rl.policies.entity_action_playout import ActionTokenBank, ActionTokenPolicy
from rl.policies.small_playout import PreparedPlayoutSamples, SmallPlayoutPolicy
from env.search.token_context import TOKEN_COUNT
from tools.analysis.lower_bank import prepare_bank

pytestmark = pytest.mark.native


def bank_for(env):
    d = sts.small_playout_public_features(env.thick.gc, env.thick.bc)
    rows = {
        k: np.concatenate([d["entities"][k], d["inventory"][k]]) for k in d["entities"]
    }
    shard = dict(
        rows,
        lengths=np.array([len(rows["ids"])]),
        alens=np.array([len(d["action_bits"])]),
        sources=d["source_rows"],
        targets=d["target_slots"],
    )
    d.update(
        visits=np.ones(len(d["action_bits"])), recommended_action=d["action_bits"][0]
    )
    legacy = PreparedPlayoutSamples([d], 0.2)
    return ActionTokenBank([shard], legacy), d


def test_coupling_can_reverse_the_ranking_between_two_targets(combat_factory):
    env = combat_factory(
        ["BASH", "ANGER", "STRIKE_RED"] * 3, encounter="Three Darklings"
    )
    b = env.thick.bc
    b.apply_custom_state(
        {
            "hand": ["BASH", "ANGER"],
            "monsters": [
                {"hp": 19},
                {"hp": 23},
                {"hp": 0, "is_dead": True, "gone": True},
            ],
        }
    )
    bank, d = bank_for(env)
    args, y, _ = bank.batch(np.array([0]), "cpu")
    _, _, _, _, _, _, sources, targets, _ = args
    pairs = {(int(s[0]), int(t)): i for i, (s, t) in enumerate(zip(sources, targets))}
    indices = [pairs[(s, t)] for s, t in [(18, 28), (18, 29), (19, 28), (19, 29)]]
    torch.manual_seed(721)
    parent = SmallPlayoutPolicy(sts.small_playout_key_count())
    net = ActionTokenPolicy.from_mature(parent)
    np.testing.assert_allclose(
        net.forward_packed(*args).detach(),
        parent.forward_packed(*bank.base.batch(np.array([0]), "cpu")[0]).detach(),
        atol=0,
        rtol=0,
    )
    with torch.no_grad():
        for m in (net.left, net.right, net.source_score, net.target_score):
            m.weight.normal_(std=0.3)

    def cross(coupled):
        net.coupled = coupled
        a, b, c, d = net.forward_packed(*args)[0, indices]
        return float((a - b - c + d).detach())

    assert abs(cross(False)) < 5e-6
    assert abs(cross(True)) > 1e-3
    loss = -(net.forward_packed(*args).log_softmax(-1) * y).sum()
    loss.backward()
    for part in (net.symbol, net.left, net.right, net.source_score, net.target_score):
        assert part.weight.grad.abs().sum() > 0

    # Construct an actual preference reversal on the four legal card–enemy pairs.
    with torch.no_grad():
        h = net.representations(args[4], args[5])[0].double()
        pref = torch.tensor([1.0, -1.0], dtype=h.dtype)
        for m in (net.left, net.right, net.source_score, net.target_score):
            m.weight.zero_()
        net.left.weight[0].copy_(torch.linalg.pinv(h[[18, 19]]) @ pref * 16)
        net.right.weight[0].copy_(torch.linalg.pinv(h[[28, 29]]) @ pref * 16)
        ab0, ab1, cd0, cd1 = net.forward_packed(*args)[0, indices]
        assert ab0 > ab1 and cd1 > cd0


def test_selected_pile_candidates_each_get_an_output_token_without_a_slot_cap(
    combat_factory,
):
    env = combat_factory(["HEADBUTT"] * 5)
    b = env.thick.bc
    b.apply_custom_state(
        {"hand": ["HEADBUTT"], "discard": [("SEARING_BLOW", i) for i in range(100)]}
    )
    b.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    bank, d = bank_for(env)
    args, _, _ = bank.batch(np.array([0]), "cpu")
    assert args[4].shape[1] == TOKEN_COUNT + 100
    np.testing.assert_array_equal(
        args[6][:, 0], np.arange(TOKEN_COUNT, TOKEN_COUNT + 100)
    )
    assert (args[6][:, 1:] == -1).all() and (args[7] == -1).all()
    assert torch.unique(args[4][0, TOKEN_COUNT:], dim=0).shape[0] == 100


def test_cross_rank_banks_check_teacher_and_keep_identical_supervision(
    tmp_path, combat_factory
):
    env = combat_factory(["BASH", "STRIKE_RED", "DEFEND_RED"] * 3)
    _, d = bank_for(env)
    bundle, bank = tmp_path / "parent", tmp_path / "bank"
    bundle.mkdir()
    (bundle / "model.zip").write_bytes(b"rank-zero fixture")
    (bank / "public").mkdir(parents=True)
    rows = {
        k: np.concatenate([d["entities"][k], d["inventory"][k]]) for k in d["entities"]
    }
    np.savez(
        bank / "public" / "000.npz",
        **rows,
        lengths=[len(rows["ids"])],
        alens=[len(d["keys"])],
        sources=d["source_rows"],
        targets=d["target_slots"],
    )
    sample = {
        k: d[k]
        for k in (
            "state",
            "actions",
            "keys",
            "action_bits",
            "visits",
            "recommended_action",
        )
    }
    torch.save(
        dict(samples=[dict(sample, env=0)], source_model_sha256="different-rank-zip"),
        bank / "samples.pt",
    )
    (bank / "collection.json").write_text(
        json.dumps(
            dict(
                samples_sha256=hashlib.sha256(
                    (bank / "samples.pt").read_bytes()
                ).hexdigest(),
                teacher_digest="same-policy",
            )
        )
    )
    with pytest.raises(ValueError, match="wrong frozen teacher"):
        prepare_bank(bank, bundle, "cpu", teacher_digest="wrong-policy")
    results = []
    for sketch in (False, True):
        _, base, features, _, _, _, _ = prepare_bank(
            bank, bundle, "cpu", teacher_digest="same-policy", sketch=sketch
        )
        idx = np.array([0])
        _, y, action = features.batch(idx, "cpu")
        _, old_y, old_action = base.batch(idx, "cpu")
        torch.testing.assert_close(y, old_y)
        torch.testing.assert_close(action, old_action)
        results.append(y)
    torch.testing.assert_close(*results)
