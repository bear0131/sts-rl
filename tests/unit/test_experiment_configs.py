"""Response-file migration preserves the original experiment argv semantics."""
import json
from pathlib import Path

import pytest

from rl.training.distributed_sb3 import build_parser

ROOT = Path(__file__).resolve().parents[2]


def test_entity_fresh_initialization_and_distributed_search_share_requested_contract(monkeypatch):
    from rl.tasks import TASKS
    from rl.training.trainer import build_parser as initial_parser
    from rl.search.native import NativeSearchConfig, apply_strict_training_settings

    monkeypatch.chdir(ROOT)
    initial = initial_parser(TASKS['wholerun']).parse_args([
        '--task', 'wholerun', '@configs/train/entity_combat.args', '--steps', '0',
    ])
    runtime = build_parser().parse_args([
        '--rank', '0', '--world-size', '4', '--checkpoint', 'fresh-step0.zip',
        '@configs/train/entity_combat_runtime.args',
    ])
    assert initial.arch == 'entity-combat'
    assert (initial.d_model, initial.n_layers, initial.n_heads) == (256, 4, 8)
    assert initial.init_from is None and initial.resume is None
    assert runtime.resume_state is None and runtime.fork_state is None
    assert not runtime.nosl_is_az_train_shared_embeddings
    assert runtime.card_language
    assert runtime.total_steps == 50_000_000
    for args in (initial, runtime):
        apply_strict_training_settings(args)
        config = NativeSearchConfig.from_model(args)
        assert config.paper_full_gumbel and config.root_search_mode == 2
        assert config.gumbel_rescale_q is False
        assert config.gumbel_value_scale == 1
        assert config.gumbel_interior_min_visits == 0
        assert config.gumbel_interior_value_mode == 0
        assert config.leaf_batch_size == 1 and config.simulations == 128
        assert args.n_envs == 96 and args.n_steps == 128
        assert args.nosl_is_az_replay_max_old_fraction == .9
        assert args.nosl_is_az_updates == 40
        assert args.nosl_is_az_batch_size == 16384
        assert args.nosl_is_az_replay_capacity == 131072


def test_boss_transition_target_cli_is_explicit_and_defaults_to_inherit():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "1", "--checkpoint", "unused.zip"]

    assert parser.parse_args(base).nosl_is_az_boss_transition_target is None
    assert parser.parse_args(
        base + ["--nosl-is-az-boss-transition-target"]
    ).nosl_is_az_boss_transition_target is True
    assert parser.parse_args(
        base + ["--no-nosl-is-az-boss-transition-target"]
    ).nosl_is_az_boss_transition_target is False


def test_standard_target_low_lr_preset_only_changes_requested_boss_target_and_lr():
    parser = build_parser()
    base = [
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
    ]
    boss_target = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_boss_transition_target_mb12288.args'}",
    ]))
    standard_target = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_standard_target_mb12288_lr5e-5.args'}",
    ]))

    assert standard_target["nosl_is_az_boss_transition_target"] is False
    assert standard_target["learning_rate"] == 5e-5
    assert {key: (boss_target[key], standard_target[key]) for key in boss_target
            if boss_target[key] != standard_target[key]} == {
        "learning_rate": (1e-4, 5e-5),
        "nosl_is_az_boss_transition_target": (True, False),
    }


def test_overworld_v3_shared_unfrozen_old80_preset_has_only_planned_deltas():
    parser = build_parser()
    base = [
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
    ]
    standard = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_standard_target_mb12288_lr5e-5.args'}",
    ]))
    shared = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old80_lr5e-5_30m.args'}",
    ]))

    assert shared["nosl_is_az_upper_strategy"] == "rules"
    assert shared["nosl_is_az_boss_transition_target"] is False
    assert shared["learning_rate"] == 5e-5
    assert shared["max_grad_norm"] == 0.5
    assert shared["nosl_is_az_replay_enabled"] is True
    assert shared["nosl_is_az_replay_capacity"] == 524_288
    assert shared["nosl_is_az_replay_max_old_fraction"] == 0.80
    assert shared["nosl_is_az_train_shared_embeddings"] is True
    assert shared["total_steps"] == 30_000_000
    assert {key: (standard[key], shared[key]) for key in standard
            if standard[key] != shared[key]} == {
        "max_grad_norm": (None, 0.5),
        "nosl_is_az_replay_max_old_fraction": (0.90, 0.80),
        "nosl_is_az_train_shared_embeddings": (False, True),
        "total_steps": (10_000_000, 30_000_000),
    }


def test_overworld_v3_fast_preset_only_rebalances_rollout_geometry():
    parser = build_parser()
    base = [
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old80_fast_30m.args'}",
    ]
    fast = vars(parser.parse_args(base))
    current = vars(parser.parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old80_lr5e-5_30m.args'}",
    ]))

    assert fast["n_envs"] == 96
    assert fast["envs_per_worker"] == 16
    assert fast["n_steps"] == 128
    assert fast["n_envs"] * fast["n_steps"] == 12_288
    assert fast["batch_size"] == 1_536
    # Fresh step-0 launches must not require a fork state.  The manager adds
    # --fork-reset-environments itself when this same preset is used for a
    # geometry fork from a schema-2 generation.
    assert fast["fork_reset_environments"] is False
    # Search, replay, learner and objective settings must remain unchanged.
    geometry = {"n_envs", "envs_per_worker", "n_steps", "batch_size",
                "gradient_accumulation_steps", "n_epochs"}
    assert {key: (current[key], fast[key]) for key in current
            if current[key] != fast[key]} == {
        key: (current[key], fast[key]) for key in geometry
        if current[key] != fast[key]
    }
    for key in (
        "nosl_is_az_simulations", "nosl_is_az_depth",
        "nosl_is_az_search_mode", "nosl_is_az_paper_full_gumbel",
        "nosl_is_az_leaf_batch_size", "nosl_is_az_batch_size",
        "nosl_is_az_updates", "nosl_is_az_micro_batch_size",
        "nosl_is_az_replay_enabled", "nosl_is_az_replay_capacity",
        "nosl_is_az_replay_max_old_fraction", "learning_rate",
        "max_grad_norm", "amp_mode", "nosl_is_az_upper_strategy",
        "nosl_is_az_train_shared_embeddings", "total_steps",
    ):
        assert fast[key] == current[key], key


def test_replay_old90_resume_override_only_changes_old_fraction():
    parser = build_parser()
    base = [
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old80_lr5e-5_30m.args'}",
    ]
    previous = vars(parser.parse_args(base))
    resumed = vars(parser.parse_args([
        *base, f"@{ROOT / 'configs/train/nosl_replay_old90_override.args'}"
    ]))

    assert resumed["nosl_is_az_replay_max_old_fraction"] == 0.90
    assert {
        key: (previous[key], resumed[key])
        for key in previous
        if previous[key] != resumed[key]
    } == {"nosl_is_az_replay_max_old_fraction": (0.80, 0.90)}


def test_old90_resume_preset_preserves_u40_geometry_and_old80_deltas():
    parser = build_parser()
    base = [
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
    ]
    old80 = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old80_lr5e-5_30m.args'}",
    ]))
    resumed = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old90_resume.args'}",
    ]))

    assert resumed["n_envs"] == 96
    assert resumed["envs_per_worker"] == 4
    assert resumed["n_steps"] == 128
    assert resumed["batch_size"] == 1536
    assert {
        key: (old80[key], resumed[key])
        for key in old80
        if old80[key] != resumed[key]
    } == {"nosl_is_az_replay_max_old_fraction": (0.80, 0.90)}


@pytest.mark.parametrize(("name", "envs_per_worker"), [
    ("nosl_overworld_v3_shared_unfrozen_old90_epw8_probe.args", 8),
    ("nosl_overworld_v3_shared_unfrozen_old90_epw16_probe.args", 16),
])
def test_old90_worker_grouping_probes_only_change_grouping_and_duration(
    name, envs_per_worker,
):
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    production = vars(parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_overworld_v3_shared_unfrozen_old90_runtime.args'}",
    ]))
    probe = vars(parser.parse_args([*base, f"@{ROOT / 'configs/train' / name}"]))

    assert probe["n_envs"] == 96
    assert probe["n_steps"] == 128
    assert probe["envs_per_worker"] == envs_per_worker
    assert probe["total_steps"] == 20_545_536
    assert {
        key: (production[key], probe[key])
        for key in production
        if production[key] != probe[key]
    } == {
        "envs_per_worker": (4, envs_per_worker),
        "total_steps": (30_000_000, 20_545_536),
    }


def test_ruleupper_override_only_changes_upper_strategy_and_keeps_40_updates():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
            f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}"]
    previous = vars(parser.parse_args(base))
    actual = vars(parser.parse_args(base + [f"@{ROOT / 'configs/train/nosl_ruleupper_v1.args'}"]))
    assert actual["nosl_is_az_updates"] == 40
    assert actual["nosl_is_az_simulations"] == 128
    assert {key: (previous[key], actual[key]) for key in previous
            if previous[key] != actual[key]} == {"nosl_is_az_upper_strategy": (None, "rules")}


def test_staggered_ruleupper_fork_only_changes_budget_upper_and_startup():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
            f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}"]
    previous = vars(parser.parse_args(base))
    actual = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_ruleupper_v1_stagger2048_20m.args'}"]
    ))

    assert actual["nosl_is_az_updates"] == 40
    assert actual["n_steps"] == 128
    assert actual["startup_stagger_steps"] == 2048
    assert {key: (previous[key], actual[key]) for key in previous
            if previous[key] != actual[key]} == {
        "nosl_is_az_upper_strategy": (None, "rules"),
        "startup_stagger_steps": (0, 2048),
        "total_steps": (10_000_000, 20_000_000),
    }


def test_u40_fourfold_preset_carries_validated_rollout_tuning():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
    ])

    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_inference_batch_size == 128
    assert args.nosl_is_az_inference_max_wait_ms == 0
    assert args.nosl_is_az_inference_schedule == "fixed_wait"
    assert args.leaf_graph_batch_multiple == 32
    assert args.leaf_graph_profile_interval == 512
    assert args.rollout_resource_interval == 2


def test_actorbatch_preset_only_rebalances_rollout_geometry_and_marks_fork_reset():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    args = parser.parse_args([
        *base,
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_no_startup_micro24576_30m.args'}",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_actorbatch_384x32.args'}",
    ])

    assert args.n_envs == 384
    assert args.n_steps == 32
    assert args.envs_per_worker == 16
    assert args.n_envs * args.n_steps == 96 * 128
    assert args.fork_reset_environments is True
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 24_576
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_updates == 40


def test_actorbatch_full_preset_is_launcher_ready_without_losing_contract():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_actorbatch_384x32_30m.args'}",
    ])

    assert args.n_envs * args.n_steps == 12_288
    assert args.envs_per_worker == 16
    assert args.total_steps == 30_000_000
    assert args.nosl_is_az_upper_strategy == "rules"
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 24_576
    assert args.nosl_is_az_replay_capacity == 524_288
    assert args.nosl_is_az_updates == 40
    assert args.fork_reset_environments is True


def test_reduced_alignment_preset_matches_large_update_geometry():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_reduced_align_768x16_30m.args'}",
    ])

    assert args.n_envs * args.n_steps == 12_288
    assert args.envs_per_worker == 32
    assert args.learning_rate == 2.5e-4
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 24_576
    assert args.nosl_is_az_batch_size // args.nosl_is_az_micro_batch_size == 4
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_capacity == 524_288
    assert args.nosl_is_az_replay_max_old_fraction == 0.90
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_search_mode == "full_gumbel"
    assert args.nosl_is_az_paper_full_gumbel is True
    assert args.nosl_is_az_upper_strategy == "rules"


def test_reduced_alignment_96x128_preset_preserves_training_contract():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_reduced_align_96x128_30m.args'}",
    ])

    assert args.n_envs == 96
    assert args.envs_per_worker == 4
    assert args.n_steps == 128
    assert args.n_envs * args.n_steps == 12_288
    assert args.learning_rate == 2.5e-4
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 24_576
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_capacity == 524_288
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_search_mode == "full_gumbel"
    assert args.fork_reset_environments is False


def test_original_width_96x128_noclip_preset_only_changes_width_lr_and_clip():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_original_256_align_96x128_30m_noclip.args'}",
    ])

    assert args.n_envs == 96
    assert args.envs_per_worker == 4
    assert args.n_steps == 128
    assert args.n_envs * args.n_steps == 12_288
    assert args.learning_rate == 1e-4
    assert args.max_grad_norm == float("inf")
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 24_576
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_capacity == 524_288
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_search_mode == "full_gumbel"
    assert args.fork_reset_environments is False


def test_original_width_alignment_preset_uses_safe_gpu_microbatch():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_original_256_align_768x16_30m.args'}",
    ])

    assert args.n_envs * args.n_steps == 12_288
    assert args.envs_per_worker == 32
    assert args.learning_rate == 1e-4
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 12_288
    assert args.nosl_is_az_batch_size // args.nosl_is_az_micro_batch_size == 8
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_capacity == 524_288
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_search_mode == "full_gumbel"


def test_original_width_globalbatch_preset_uses_four_rank_root_batch():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_original_256_globalbatch_384x128_30m.args'}",
    ])

    assert args.n_envs == 384
    assert args.envs_per_worker == 16
    assert args.n_steps == 128
    assert args.n_envs * args.n_steps * args.world_size == 196_608
    assert args.learning_rate == 3e-4
    assert args.nosl_is_az_batch_size == 196_608
    assert args.nosl_is_az_updates == 10
    assert args.nosl_is_az_micro_batch_size == 12_288
    assert args.nosl_is_az_replay_enabled is False
    assert args.nosl_is_az_train_shared_embeddings is True
    assert args.total_steps == 30_000_000


def test_occ1k_resume_override_retains_fastest_validated_leaf_schedule():
    args = build_parser().parse_args([
        "--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip",
        f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}",
        f"@{ROOT / 'configs/train/nosl_ruleupper_v1_occupancy1000_30m.args'}",
    ])

    assert args.total_steps == 30_000_000
    assert args.startup_occupancy_max_age == 1000
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_simulations == 128
    assert args.nosl_is_az_batch_size == 98_304
    assert args.nosl_is_az_micro_batch_size == 12_288
    assert args.nosl_is_az_inference_batch_size == 128
    assert args.nosl_is_az_inference_max_wait_ms == 0
    assert args.nosl_is_az_inference_schedule == "fixed_wait"
    assert args.leaf_graph_batch_multiple == 32


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
    assert args.nosl_is_az_inference_batch_size == 128
    assert args.nosl_is_az_inference_max_wait_ms == 0
    assert args.nosl_is_az_inference_schedule == "fixed_wait"
    assert args.nosl_is_az_updates == 40
    assert args.nosl_is_az_replay_max_old_fraction == 1.0
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
        "nosl_is_az_micro_batch_size": (2048, 8192),
        "nosl_is_az_replay_capacity": (524288, 2097152),
        "nosl_is_az_replay_max_old_fraction": (0.9, 1.0),
    }


def test_u10x4_preset_carries_rollout_tuning():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    previous = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args'}"]
    ))
    args = parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u10x4_20260916.args'}"]
    )
    optimized = parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u40_16x_20260916.args'}"]
    )

    assert args.nosl_is_az_updates == 10
    assert args.nosl_is_az_inference_batch_size == 128
    assert args.nosl_is_az_inference_max_wait_ms == 0
    assert args.nosl_is_az_inference_schedule == "fixed_wait"
    assert args.leaf_graph_batch_multiple == 32
    assert args.leaf_graph_profile_interval == 512
    assert args.rollout_resource_interval == 2
    for key in (
        "nosl_is_az_inference_batch_size",
        "nosl_is_az_inference_max_wait_ms",
        "nosl_is_az_inference_schedule",
        "leaf_graph_batch_multiple",
        "leaf_graph_profile_interval",
        "rollout_resource_interval",
    ):
        assert getattr(args, key) == getattr(optimized, key)
    differences = {
        key: (previous[key], value)
        for key, value in vars(args).items()
        if previous[key] != value
    }
    assert differences == {"nosl_is_az_updates": (40, 10)}


def test_u40x4_uniform_replay_only_removes_freshness_ratio():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    u10x4 = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_fresh10_u10x4_20260916.args'}"]
    ))
    args = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_u40x4_uniform_20260916.args'}"]
    ))

    assert args["nosl_is_az_replay_enabled"] is True
    assert args["nosl_is_az_replay_capacity"] == 524288
    assert args["nosl_is_az_updates"] == 40
    assert args["nosl_is_az_replay_max_old_fraction"] == 1.0
    differences = {
        key: (u10x4[key], value)
        for key, value in args.items()
        if u10x4[key] != value
    }
    assert differences == {
        "nosl_is_az_updates": (10, 40),
        "nosl_is_az_replay_max_old_fraction": (0.9, 1.0),
    }


def test_u80x4_uniform_replay_only_quarters_capacity_and_doubles_updates():
    parser = build_parser()
    base = ["--rank", "0", "--world-size", "4", "--checkpoint", "unused.zip"]
    u40x4 = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_u40x4_uniform_20260916.args'}"]
    ))
    args = vars(parser.parse_args(
        base + [f"@{ROOT / 'configs/train/nosl_paper_replay_u80x4_uniform_20260916.args'}"]
    ))

    assert args["nosl_is_az_replay_enabled"] is True
    assert args["nosl_is_az_replay_capacity"] == 131072
    assert args["nosl_is_az_updates"] == 80
    assert args["nosl_is_az_replay_max_old_fraction"] == 1.0
    differences = {
        key: (u40x4[key], value)
        for key, value in args.items()
        if u40x4[key] != value
    }
    assert differences == {
        "nosl_is_az_updates": (40, 80),
        "nosl_is_az_replay_capacity": (524288, 131072),
    }


def test_public_scenario_presets_preserve_original_card_order():
    from configs.eval import nosl_scenarios
    expected = json.loads((ROOT / "tests/fixtures/configs/nosl_scenarios.json").read_text())
    actual = {name: getattr(nosl_scenarios, name) for name in expected}
    assert json.loads(json.dumps(actual)) == expected
