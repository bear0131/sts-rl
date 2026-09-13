from __future__ import annotations

from collections import Counter

from rl.rules.training.distributed_train_rules import _historical_wandb_rows
from rl.rules.rule_data import CardBalancedBatchSampler, action_card_group


def _card_sample(card_id):
    action = {"type": "END"}
    if card_id is not None:
        action = {"type": "CARD", "card": {"id": card_id, "up": 0}}
    return {"action": action}


def test_card_balanced_sampler_is_deterministic_and_length_local():
    lengths = list(range(1, 65))
    samples = [_card_sample("Strike_R") for _length in lengths]
    first = list(CardBalancedBatchSampler(
        samples, lengths, batch_size=4, num_batches=20,
        bucket_multiplier=2, seed=7
    ))
    second = list(CardBalancedBatchSampler(
        samples, lengths, batch_size=4, num_batches=20,
        bucket_multiplier=2, seed=7
    ))
    assert first == second
    assert all(len(batch) == 4 for batch in first)
    assert all(
        max(lengths[index] for index in batch)
        - min(lengths[index] for index in batch) <= 7
        for batch in first
    )


def test_card_balanced_sampler_can_resume_without_repeating_batches():
    lengths = list(range(1, 65))
    samples = [
        _card_sample("Strike_R" if index < 60 else "Dropkick")
        for index in range(len(lengths))
    ]
    uninterrupted = CardBalancedBatchSampler(
        samples, lengths, batch_size=4, num_batches=20,
        bucket_multiplier=2, seed=7
    )
    expected = list(uninterrupted)
    resumed = CardBalancedBatchSampler(
        samples, lengths, batch_size=4, num_batches=10,
        bucket_multiplier=2, seed=7
    )
    resumed.skip_batches(10)
    assert list(resumed) == expected[10:]


def test_card_balanced_sampler_equalizes_extreme_raw_imbalance():
    samples = (
        [_card_sample("Strike_R") for _ in range(1000)]
        + [_card_sample("Dropkick") for _ in range(3)]
        + [_card_sample("Bash")]
        + [_card_sample(None) for _ in range(20)]
    )
    lengths = [10 + index % 17 for index in range(len(samples))]
    batches = list(CardBalancedBatchSampler(
        samples, lengths, batch_size=8, num_batches=41,
        bucket_multiplier=4, seed=11,
    ))
    groups = []
    for batch in batches:
        batch_groups = {action_card_group(samples[index]) for index in batch}
        assert len(batch_groups) == 1
        groups.append(next(iter(batch_groups)))
    counts = Counter(groups)
    assert set(counts) == {"Strike_R", "Dropkick", "Bash", "<NON_CARD_ACTION>"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_historical_wandb_rows_use_global_optimizer_updates(tmp_path):
    log = tmp_path / "rank0.log"
    log.write_text(
        "[rule-dist rank=0] epoch=1 update=50/775 loss=1.25 "
        "sync=18.4s grad_mb=56.2\n"
        "[rule-dist rank=0] epoch=1 val_loss=0.75 effects=80.000% "
        "next=85.500% epoch_sec=470.0 sync_sec=290.0 elapsed=470.0\n"
        "[rule-dist rank=0] epoch=2 update=50/775 loss=0.70 "
        "sync=19.0s grad_mb=56.2\n",
        encoding="utf-8",
    )

    rows = _historical_wandb_rows(log, max_epoch=1, steps_per_epoch=775)

    assert [step for step, _payload in rows] == [50, 775]
    assert rows[0][1]["train/loss"] == 1.25
    assert rows[1][1]["validation/effect_token_accuracy"] == 0.8
    assert rows[1][1]["validation/next_state_token_accuracy"] == 0.855
