"""Response-file migration preserves the original experiment argv semantics."""
import json
from pathlib import Path

import pytest

from rl.training.distributed_sb3 import build_parser

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["nosl_paper_replay_fresh10_20260913",
                                 "nosl_paper_noetl_noreplay_e288_20260913",
                                 "nosl_fixed_fpu_20260909"])
def test_preset_matches_historical_explicit_arguments(name):
    old = json.loads((ROOT / f"tests/fixtures/configs/{name}.json").read_text())
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    parser = build_parser()
    expected = vars(parser.parse_args(base + old))
    actual = vars(parser.parse_args(base + [f"@{ROOT / 'configs/train' / (name + '.args')}"]))
    assert actual == expected


def test_current_replay_experiment_identity_and_explicit_override():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_20260913.args'}",
        "--total-steps", "64",
    ])
    assert args.total_steps == 64
    assert args.nosl_is_az_search_mode == "full_gumbel"
    assert args.nosl_is_az_paper_full_gumbel
    assert args.nosl_is_az_depth == 0
    assert args.nosl_is_az_leaf_batch_size == 1
    assert args.nosl_is_az_replay_capacity == 131072
    assert args.nosl_is_az_replay_max_old_fraction == 0.9
    assert args.nosl_is_az_batch_size == 24576
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_micro_batch_size == 2048
    assert args.n_envs == 24


def test_u5_replay_experiment_only_changes_optimizer_update_count():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    u40 = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_20260913.args'}"]
    ))
    u5 = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u5_20260914.args'}"]
    ))

    differences = {
        key: (u40[key], u5[key])
        for key in u40
        if u40[key] != u5[key]
    }
    assert differences == {"nosl_is_az_updates": (40, 5)}


def test_u40_sixteenfold_width_preset_scales_coupled_geometry():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    previous = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}"]
    ))
    args = parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_16x_20260916.args'}"]
    )

    assert args.n_envs == 384
    assert args.envs_per_worker == 16
    assert args.n_envs // args.envs_per_worker == 24
    assert args.n_steps == 128
    assert args.batch_size == 6144
    assert args.nosl_is_az_batch_size == 393216
    assert args.nosl_is_az_micro_batch_size == 8192
    assert args.nosl_is_az_batch_size // args.nosl_is_az_micro_batch_size == 48
    assert args.nosl_is_az_replay_capacity == 2097152
    assert args.nosl_is_az_inference_batch_size == 256
    assert args.nosl_is_az_inference_max_wait_ms == 0.5
    assert args.nosl_is_az_inference_schedule == "adaptive"
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_max_old_fraction == 0.9
    differences = {
        key: (previous[key], value)
        for key, value in vars(args).items()
        if previous[key] != value
    }
    assert differences == {
        "batch_size": (1536, 6144),
        "envs_per_worker": (4, 16),
        "n_envs": (96, 384),
        "nosl_is_az_batch_size": (98304, 393216),
        "nosl_is_az_inference_batch_size": (None, 256),
        "nosl_is_az_inference_max_wait_ms": (None, 0.5),
        "nosl_is_az_inference_schedule": (None, "adaptive"),
        "nosl_is_az_micro_batch_size": (2048, 8192),
        "nosl_is_az_replay_capacity": (524288, 2097152),
    }


def test_public_scenario_presets_preserve_original_card_order():
    from configs.eval import nosl_scenarios
    expected = json.loads((ROOT / "tests/fixtures/configs/nosl_scenarios.json").read_text())
    actual = {name: getattr(nosl_scenarios, name) for name in expected}
    assert json.loads(json.dumps(actual)) == expected
