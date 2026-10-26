"""Warm-start logits, new FP32 unbounded value and full native exit routing."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from env.core import native_actions as C
from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig, complete_searches_native, observation_as_tensor

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


@pytest.mark.parametrize('card,eligible', [('SECRET_TECHNIQUE', ['DEFEND_RED', 'SHRUG_IT_OFF']),
                                         ('SECRET_WEAPON', ['STRIKE_RED', 'BASH'])])
@pytest.mark.parametrize('nonce', [1, 42, 42101])
def test_draw_selection_root_preserves_public_handles_and_priors(outer_return_core, card, eligible, nonce):
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    excluded = 'STRIKE_RED' if card == 'SECRET_TECHNIQUE' else 'DEFEND_RED'
    env.thick.bc.apply_custom_state({'hand': [card], 'draw': [excluded]*12 + eligible})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    candidates = env.thick.bc.select_info()['candidates']
    public_index = next(i for i, candidate in enumerate(candidates)
                        if candidate['card'][0] == eligible[1])
    expected = C.AUTOREGRESSIVE_PICK + public_index
    assert env.action_masks()[expected]
    logits = np.full(env.action_space.n, -100., np.float32)
    logits[expected] = 100.
    pool = NativeSearchBatch([env], 1)
    try:
        config = NativeSearchConfig(simulations=4, depth=1, fixed_worldline=True,
                                    outer_return=True, gumbel_scale=0.)
        pool.begin({0: ((logits,), dict(**asdict(config), root_value=0.,
                                      state_nonce=nonce, compact_observation=True))})
        while not pool.done:
            pool.advance(np.zeros((pool.rows, env.action_space.n), np.float32), np.zeros(pool.rows, np.float32))
        result = pool.results()[0]
        assert result['action'] == expected
        assert result['native_action_bits'] == (2 << 29 | 13)
        assert env.thick.bc.to_dict() == before
        env.step(expected)
        assert env.thick.bc.to_dict()['hand'][-1][0] == eligible[1]
    finally:
        pool.close()



def _enter_combat(env):
    for _ in range(64):
        if env.thick.bc is not None:
            return
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    raise AssertionError('no combat within fixture budget')


def test_tensor_encoding_omits_only_unused_snapshot_inventory(outer_return_core, monkeypatch):
    _, _, env = outer_return_core
    seen = set()
    for _ in range(192):
        full = env.thick.obs_canonical()
        assert 'deck_state' in full['global']
        if full['phase'] != 'combat':
            assert {'deck', 'relics', 'keys'} <= full.keys()
        compact = env.thick.encoding_canonical()
        assert 'deck_state' not in compact['global']
        actual = env._obs()
        with monkeypatch.context() as patch:
            patch.setattr(env.thick, 'encoding_canonical', env.thick.obs_canonical)
            expected = env._obs()
        assert actual.keys() == expected.keys()
        for key in actual:
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].tobytes() == expected[key].tobytes(), key
        if full['phase'] == 'combat':
            with pytest.raises(ValueError, match='settled noncombat'):
                env._obs(upper_value_only=True)
        else:
            value_input = env._obs(upper_value_only=True)
            assert value_input.keys() == actual.keys() - {'combat_legal', 'outer_deck_misc'}
            for key in value_input:
                assert value_input[key].dtype == actual[key].dtype
                assert value_input[key].tobytes() == actual[key].tobytes(), key
        seen.add(full['phase'])
        legal = np.flatnonzero(env.action_masks())
        if env.thick.bc is not None:
            env.thick.bc.set_player_strength(999)
            plays = [a for a in legal if C.AUTOREGRESSIVE_PLAY <= a < C.AUTOREGRESSIVE_POTION]
            action = plays[0] if plays else legal[0]
        else:
            action = legal[0]
        _, _, done, truncated, _ = env.step(int(action))
        if done or truncated:
            break
    assert {'event', 'map', 'combat', 'combat_reward'} <= seen


def test_grid_leaf_choices_override_root_context_and_reach_lower(outer_return_core):
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_relics(['SacredBark'])
    env.thick.gc.set_potion_state(['Liquid Memories'])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.apply_custom_state({'discard': [('BASH', 1), ('DEFEND_RED', 0), ('ANGER', 0)]})
    env.thick.bc.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    env._invalidate_backend_cache()
    assert not env._obs()['selected_card_ids'].any()
    batch = NativeSearchBatch([env], 1)
    config = NativeSearchConfig(simulations=4, depth=1, outer_return=True, fixed_worldline=True, gumbel_scale=0.)
    try:
        batch.begin({0: ((np.zeros(198, np.float32),), dict(**asdict(config), root_value=0.,
                         state_nonce=42101, compact_observation=True))})
        observed = batch.observations()
        assert np.count_nonzero(observed['selected_card_ids']) == 1
        tensor = observation_as_tensor(observed, 'cpu')
        core.eval()
        with torch.inference_mode():
            chosen_context = core.combat.context_features(tensor, torch.float32)
            without_choice = dict(tensor, selected_card_ids=torch.zeros_like(tensor['selected_card_ids']))
            assert not torch.equal(chosen_context, core.combat.context_features(without_choice, torch.float32))
        assert not env._obs()['selected_card_ids'].any()
        core.teacher.assert_integrity()
    finally:
        batch.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_outer_native_writer_preserves_selected_rows_context_and_padding(outer_return_core, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('direct pinned graph inputs require CUDA')
    from rl.search.inference import CombatLeafGraphs
    core, _, env = outer_return_core
    _enter_combat(env)
    envs = [env.fork_run() for _ in range(3)]
    for i, item in enumerate(envs):
        item.thick.bc.apply_custom_state({'player': {'hp': 20+i*13, 'gold': 17+i*61}})
        item._invalidate_backend_cache()
    pool = NativeSearchBatch(envs, 2)
    graphs = reference_graphs = None
    if device == 'cuda':
        core.to(device).eval()
        graphs = CombatLeafGraphs(core, device, amp='bf16', batch_multiple=2,
                                  capacity=2, compile_forward=False)
        reference_graphs = CombatLeafGraphs(core, device, amp='bf16', batch_multiple=2,
                                            capacity=2, compile_forward=False)
    config = NativeSearchConfig(simulations=8, depth=1, outer_return=True,
                                fixed_worldline=True, gumbel_scale=0.)
    try:
        pool.begin({i: ((np.zeros(198, np.float32),), dict(**asdict(config), root_value=0.,
                    state_nonce=42101+i, compact_observation=True)) for i in range(3)})
        # Distinct live wallet and HP must be decoded, not copied from GC.
        full = pool.observations()
        assert len(set(full['gold'][:, 0])) == 3
        assert len(set(full['run_hp'][:, 0])) == 3
        # Permanent growth belongs to the selected leaf's original row.
        pool.payload['permanent_misc'] = {2: {0: 41}, 0: {1: 17}}
        full = pool.observations()
        assert full['outer_deck_misc'][2, 0] == np.float32(.41)
        assert full['outer_deck_misc'][0, 1] == np.float32(.17)
        empty = pool.observations([])
        for key, value in full.items():
            assert empty[key].shape == (0, *value.shape[1:])
            assert empty[key].dtype == value.dtype
        with pytest.raises(IndexError):
            pool.observations([0.5])
        target = {k: np.full((5, *v.shape[1:]), 3, dtype=v.dtype) for k, v in full.items()}
        for rows in ([2, 0], [1], [0, 1, 2], [2, 1], np.array([2, 0], dtype=np.int32)):
            padding = {k: v[len(rows):].tobytes() for k, v in target.items()}
            pool.write_observations(target, rows)
            selected = pool.observations(rows)
            for key, value in full.items():
                assert selected[key].tobytes() == value[rows].tobytes(), key
                assert target[key][:len(rows)].tobytes() == value[rows].tobytes(), key
                assert target[key][len(rows):].tobytes() == padding[key], key
            if graphs is not None:
                actual = graphs.forward_native(pool, rows)
                reference = reference_graphs({k: v[rows] for k, v in full.items()}, pool.relic_width(rows))
                for a, b in zip(actual, reference, strict=True):
                    assert a.tobytes() == b.tobytes()
                pending = graphs.forward_native_deferred(pool, rows)
                with pytest.raises(RuntimeError, match='finish pending'):
                    graphs.forward_native(pool, rows)
                if len(rows) == 1:
                    # Clearing during a pending transfer must drain it first;
                    # the caller still owns the completed prediction.
                    graphs.clear()
                for a, b in zip(pending(), reference, strict=True):
                    assert a.tobytes() == b.tobytes()
                assert graphs._pending is None
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
        core.teacher.assert_integrity()
    finally:
        pool.close()
        for graph in (graphs, reference_graphs):
            if graph is not None:
                graph.clear()


def test_warm_actor_scalar_updates_and_frozen_teacher(outer_return_core):
    core, source, env = outer_return_core
    _enter_combat(env)
    obs = observation_as_tensor({k: v[None] for k, v in env._obs().items()}, 'cpu')
    core.eval()
    with torch.no_grad():
        expected = source.combat_actor_critic(obs)[0]
        actual = core.combat_actor_critic(obs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not hasattr(core.combat, 'distribution') and not hasattr(core.combat, 'transition')
    assert core.teacher.upper_tokenizer.card.vectors.data_ptr() != core.combat.card.vectors.data_ptr()
    frozen = core.teacher.assert_integrity()
    core.train()
    optimizer = torch.optim.Adam(core.combat.parameters(), lr=5e-5)
    for target in (2.7, -.4):
        optimizer.zero_grad()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, value = core.combat_actor_critic(obs)
            assert value.dtype == torch.float32
            loss = (value - target).square().mean() + .01 * logits[:, 0].square().mean()
        loss.backward()
        assert all(p.grad is None for p in core.teacher.parameters())
        optimizer.step()
        assert core.teacher.assert_integrity() == frozen
    with torch.no_grad():
        core.combat.scalar[-1].weight.zero_()
        core.combat.scalar[-1].bias.fill_(2.7)
        assert core.combat_actor_critic(obs)[1].item() == pytest.approx(2.7)


def test_native_roots_reuse_collected_context_with_owned_storage(outer_return_core, monkeypatch):
    _, _, env = outer_return_core
    _enter_combat(env)
    envs = [env.fork_run() for _ in range(3)]
    for i, item in enumerate(envs):
        item.thick.bc.apply_custom_state({'player': {'hp': 20+i*13, 'gold': 17+i*61}})
        item._invalidate_backend_cache()
    raw = [item._obs() for item in envs]
    observations = {key: np.stack([row[key] for row in raw]) for key in raw[0]}
    config = NativeSearchConfig(simulations=8, depth=1, outer_return=True,
                                fixed_worldline=True, gumbel_scale=0.)
    # Root subset and order must follow environment indices, not selected rows.
    calls = {i: ((np.zeros(198, np.float32),), dict(**asdict(config), root_value=0.,
                    state_nonce=42101+i, compact_observation=True)) for i in (2, 0)}
    pool = NativeSearchBatch(envs, 2)
    try:
        pool.begin(calls)
        reference = pool.observations()
        pool.clear()
        def unexpected_encode():
            raise AssertionError('collector context must avoid a second full observation')
        for item in envs:
            monkeypatch.setattr(item, '_obs', unexpected_encode)
        # Complete root observations also eliminate redundant GameContext
        # serialization; context-only callers retain the older fallback.
        import env.core.run_env as run_env
        monkeypatch.setattr(run_env, 'normalize_sim_global', unexpected_encode)
        pool.begin(calls, root_observations=observations)
        actual = pool.observations()
        assert actual.keys() == reference.keys()
        for key in reference:
            assert actual[key].tobytes() == reference[key].tobytes(), key
        observations['deck_ids'].fill(0)
        observations['relics'].fill(0)
        assert pool.observations()['deck_ids'].tobytes() == reference['deck_ids'].tobytes()
        invalid = dict(observations, floor=observations['floor'].astype(np.float64))
        with pytest.raises(ValueError, match='root context batch.*floor'):
            pool.begin(calls, root_observations=invalid)
    finally:
        pool.close()


@pytest.mark.parametrize('native_action', [False, True])
def test_combat_label_context_reuses_observation_without_aliasing_or_staleness(
        short_combat_env_type, monkeypatch, native_action):
    from env.core.outer_return import CONTEXT_KEYS
    from env.core import game_actions as ga, actions

    env = short_combat_env_type(global_map_view=True,
        combat_action_contract='autoregressive-v2', outer_return_context=True,
        combat_value_objective='combat-outcome-means-v1')
    env.reset(seed=42101)
    reference = env.fork_run()
    try:
        observation = env._obs()
        expected = {key: observation[key].copy() for key in CONTEXT_KEYS}
        # Mutating collector-owned storage must not corrupt a terminal label.
        for key in CONTEXT_KEYS:
            observation[key].fill(0)

        def transition(item):
            return (item._backend_step_native_combat(4 << 29) if native_action
                    else item._backend_step(ga.combat(actions.END_TURN)))

        reference._nosl_value_capture_pending = True
        expected_result = transition(reference)
        env._nosl_value_capture_pending = True
        with monkeypatch.context() as patch:
            patch.setattr(env, '_obs', lambda **_: pytest.fail('redundant full observation'))
            actual_result = transition(env)
        assert actual_result[:3] == expected_result[:3]
        actual_info, expected_info = actual_result[3], expected_result[3]
        assert actual_info.keys() == expected_info.keys()
        for key in actual_info:
            if key.endswith('_observation'):
                for field in actual_info[key]:
                    assert actual_info[key][field].tobytes() == expected_info[key][field].tobytes(), field
            else:
                assert actual_info[key] == expected_info[key], key
        for key in CONTEXT_KEYS:
            assert env._nosl_context_before[key].tobytes() == expected[key].tobytes(), key
        assert env._nosl_current_context is None, 'transition must invalidate the current-state cache'
        refreshed = env._obs()
        env._capture_nosl_context()
        for key in CONTEXT_KEYS:
            assert env._nosl_context_before[key].tobytes() == refreshed[key].tobytes(), key
        env.reset(seed=42102)
        # reset emits a new observation; it must never retain pre-reset context.
        for key in CONTEXT_KEYS:
            assert env._nosl_current_context[key].tobytes() == env._obs()[key].tobytes(), key
    finally:
        reference.close()
        env.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_exit_uses_teacher_on_settled_reward_page(outer_return_core, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('native outer graph routing requires CUDA')
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    core.to(device).eval()
    model = SimpleNamespace(policy=SimpleNamespace(core=core), device=torch.device(device),
                            action_space=env.action_space)
    if device == 'cuda':
        from rl.search.inference import CombatLeafGraphs
        from rl.search.rollout_inference import RolloutInference
        model._native_leaf_inference = CombatLeafGraphs(core, device, amp='bf16',
                                                       compile_forward=False)
        model._rollout_inference = RolloutInference(model)
    config = NativeSearchConfig(simulations=8, fixed_worldline=True, outer_return=True,
                               gumbel_scale=0., gumbel_value_scale=1., gumbel_rescale_q=False)
    logits = np.zeros(198, np.float32)
    logits[C.AUTOREGRESSIVE_PLAY:C.AUTOREGRESSIVE_POTION] = 20
    calls = {0: ((logits,), dict(**asdict(config), root_value=2.7,
                               state_nonce=42101, compact_observation=True))}
    batch = NativeSearchBatch([env], 2)
    seen = []
    original = batch.combat_exits
    def capture(**kwargs):
        exits = original(**kwargs)
        seen.extend(exits.values())
        return exits
    batch.combat_exits = capture
    try:
        result, stats = complete_searches_native(model, batch, calls)
        assert stats['leaf_evaluations'] > 0 and seen
        assert any(exit.kind == 'upper' and int(obs['phase'][0]) != 1 for exit, obs in seen)
        assert np.isfinite(result[0]['root_value'])
        assert env.thick.bc.to_dict() == before
        core.teacher.assert_integrity()
    finally:
        batch.close()
        if device == 'cuda':
            model._native_leaf_inference.clear()
            model._rollout_inference.clear()


@pytest.mark.parametrize('ending', ['win', 'smoke', 'death'])
@pytest.mark.parametrize('upper_value_only', [False, True])
def test_search_exit_view_matches_complete_branch_and_keeps_live_history(outer_return_core, ending, upper_value_only):
    import copy
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.gc.set_potion_state(['Smoke Bomb'])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env._invalidate_backend_cache()
    history = copy.deepcopy(env._episode_action_journal)
    live = env.thick.bc.to_dict()
    base = env.thick.fork_run(worldline=42101)
    reset_rng = copy.deepcopy(base._np_random.bit_generator.state)
    map_before = base.rmap
    gc_before = base.obs_canonical()
    leaf = env.thick.bc.clone()
    if ending == 'win':
        leaf.set_player_strength(999)
        leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    elif ending == 'smoke':
        leaf.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    else:
        leaf.apply_custom_state({'player': {'hp': 1}})
        for _ in range(16):
            if leaf.is_battle_over:
                break
            leaf.execute(sts.Action(sts.ActionType.END_TURN))
    assert leaf.is_battle_over
    # Independent reference uses the ordinary full-copy API and the same
    # sampled future, retaining its public pending/reset semantics.
    reference = env.fork_run()
    reference.thick, expected_exit = base.settle_combat_leaf(leaf)
    reference._pending = reference._multi = None
    reference._invalidate_backend_cache()
    expected = None if expected_exit.kind == 'terminal' else reference._obs()
    env._nosl_outer_branch = base
    for _ in range(2):
        outcome, observation = env.settle_search_combat_exit(leaf, upper_value_only=upper_value_only)
        assert outcome == expected_exit
        if expected is None:
            assert observation is None
        else:
            omitted = {'combat_legal', 'outer_deck_misc'} if upper_value_only and outcome.kind == 'upper' else set()
            assert observation.keys() == expected.keys() - omitted
            for key in observation:
                assert observation[key].dtype == expected[key].dtype
                assert observation[key].tobytes() == expected[key].tobytes(), key
            if outcome.kind == 'upper':
                wanted = core.teacher.prepare_search_numpy([expected])
                actual = core.teacher.prepare_search_numpy([observation])
                assert len(actual) == len(wanted)
                for (indices, phase, widths, rows), (want_indices, want_phase, want_widths, want_rows) in zip(actual, wanted, strict=True):
                    np.testing.assert_array_equal(indices, want_indices)
                    assert phase == want_phase and widths == want_widths
                    assert rows.keys() == want_rows.keys()
                    for key in rows:
                        assert rows[key].tobytes() == want_rows[key].tobytes(), key
        assert env._episode_action_journal == history
        assert env.thick.bc.to_dict() == live
        assert base._np_random.bit_generator.state == reset_rng
        assert base.rmap is map_before
        assert base.obs_canonical() == gc_before
    assert base.bc.to_dict() == live
    core.teacher.assert_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CPU settlement overlaps a real CUDA graph')
def test_mixed_exit_settlement_overlaps_lower_without_changing_search(outer_return_core, monkeypatch):
    import slaythespire as sts
    from rl.search.inference import CombatLeafGraphs
    from rl.search.rollout_inference import RolloutInference
    from rl.search.native import forward_outer_leaves
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    ordinary = env.fork_run()
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    core.to('cuda').eval()
    graph = CombatLeafGraphs(core, 'cuda', amp='bf16', compile_forward=False)
    model = SimpleNamespace(policy=SimpleNamespace(core=core), device=torch.device('cuda'),
                            action_space=env.action_space, _native_leaf_inference=graph)
    config = NativeSearchConfig(simulations=8, fixed_worldline=True, outer_return=True,
                               gumbel_scale=0., gumbel_value_scale=1., gumbel_rescale_q=False)
    logits = np.zeros(198, np.float32)
    logits[C.AUTOREGRESSIVE_PLAY:C.AUTOREGRESSIVE_POTION] = 20
    calls = {i: ((logits,), dict(**asdict(config), root_value=2.7,
             state_nonce=42101, compact_observation=True)) for i in range(2)}
    pool = NativeSearchBatch([env, ordinary], 2)
    seen = []
    settle = pool.combat_exits
    def capture(**kwargs):
        if 0 < len(pool.combat_exit_rows) < pool.rows:
            seen.append(graph._pending is not None)
        return settle(**kwargs)
    monkeypatch.setattr(pool, 'combat_exits', capture)
    try:
        results = []
        for overlap in (False, True):
            graph.clear()
            model._native_leaf_inference = graph if overlap else SimpleNamespace(forward_native=graph.forward_native)
            model._rollout_inference = RolloutInference(model)
            result, _ = complete_searches_native(model, pool, calls)
            results.append(result)
            model._rollout_inference.clear()
        assert False in seen and True in seen
        for index in results[0]:
            for key in ('visit_policy', 'root_visit_policy', 'root_visits', 'root_value', 'action'):
                np.testing.assert_array_equal(results[0][index][key], results[1][index][key])
        # Failure after launch must not leave a transfer using recyclable input.
        model._native_leaf_inference = graph
        model._rollout_inference = RolloutInference(model)
        pool.begin(calls)
        for _ in range(16):
            if 0 < len(pool.combat_exit_rows) < pool.rows:
                break
            pool.advance(*forward_outer_leaves(model, pool))
        assert 0 < len(pool.combat_exit_rows) < pool.rows
        def fail(**kwargs):
            assert graph._pending is not None
            raise ValueError('settlement failure after GPU launch')
        monkeypatch.setattr(pool, 'combat_exits', fail)
        with pytest.raises(ValueError, match='settlement failure'):
            forward_outer_leaves(model, pool)
        assert graph._pending is None
        core.teacher.assert_integrity()
    finally:
        pool.close()
        graph.clear()
        model._rollout_inference.clear()


def test_material_manifest_guard(tmp_path):
    from rl.policies.outer_return import load_initialization_material
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest SHA256'):
        load_initialization_material(tmp_path, manifest_sha256='0' * 64, source_sha256='1' * 64)


def test_permanent_growth_context_matches_real_settlement(outer_return_core):
    import slaythespire as sts
    from env.core.outer_return import permanent_misc
    _, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck_state([dict(id='RITUAL_DAGGER', upgrades=0, misc=15, bottle='NONE')])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.set_player_strength(999)
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert env.thick.bc.is_battle_over
    assert env.thick.gc.deck_state()[0]['misc'] == 15
    assert permanent_misc(env.thick.gc, env.thick.bc)[0] == pytest.approx(.18)
    branch, _ = env.thick.settle_combat_leaf(env.thick.bc)
    assert branch.gc.deck_state()[0]['misc'] == 18
    assert permanent_misc(branch.gc)[0] == pytest.approx(.18)


@pytest.mark.parametrize('objective', ['combat-outcome-quantiles-v1', 'combat-outcome-means-v1'])
def test_train47_context_branch_receives_leaf_hp_and_permanent_growth(frozen_upper_source, objective):
    import slaythespire as sts
    from env.core.run_env import SlayTheSpireEnv
    from rl.policies.outer_return import OuterReturnCore
    source, space, _, _ = frozen_upper_source
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
        outer_return_context=True, combat_value_objective=objective)
    pool = None
    try:
        env.reset(seed=42101)
        _enter_combat(env)
        env.thick.gc.set_deck_state([dict(id='RITUAL_DAGGER', upgrades=0, misc=15, bottle='NONE')])
        env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
        env.thick.bc.apply_custom_state({'player': {'hp': 11}})
        env._invalidate_backend_cache()
        core = OuterReturnCore(source, space, env.observation_space, source_sha256='1'*64,
                               objective=objective)
        assert not hasattr(core.combat, 'scalar')
        assert hasattr(core.combat, 'distribution') == (objective == 'combat-outcome-quantiles-v1')
        assert hasattr(core.combat, 'means') == (objective == 'combat-outcome-means-v1')
        assert not torch.equal(core.combat.end.weight, source.combat.end.weight)
        pool = NativeSearchBatch([env], 1)
        pool.begin({0: ((np.zeros(env.action_space.n),), dict(state_nonce=42, simulations=4,
            depth=1, compact_observation=True, fixed_worldline=True, boss_win_weight=.9))})
        assert not pool.done
        obs = pool.observations()
        np.testing.assert_array_equal(obs['run_hp'][:, :2], obs['player_scalars'][:, :2])
        assert np.all(obs['outer_deck_misc'][:, 0] == np.float32(.15))
        with torch.no_grad():
            logits, values = core.combat_actor_critic(observation_as_tensor(obs, 'cpu'))
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()
        env.thick.bc.set_player_strength(999)
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        raw = env.thick.bc.to_dict()
        assert raw['permanent_misc'][0] == 18
        encoded = env._native_leaf_observation(raw, combat_only=True)
        assert encoded['outer_deck_misc'][0] == pytest.approx(.18)
    finally:
        if pool is not None:
            pool.close()
        env.close()
