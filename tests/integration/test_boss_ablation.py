import numpy as np
import pytest
import torch
import slaythespire as sts

from env.core import native_actions as C
from rl.policies.outer_return import OuterReturnCore
from rl.algorithms.combat_distribution import MEAN_CONTRACT, boss_ablation_target

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


@pytest.mark.parametrize('mode', [None, 'full-or-zero', 'real-hp'])
def test_fresh_preparation_has_no_training_state(outer_material, tmp_path, monkeypatch, mode):
    import json
    from pathlib import Path
    from rl.training.combat_outer import prepare
    from rl.training.distributed_sb3 import build_parser
    from rl.policies.checkpoint import load_checkpoint
    monkeypatch.setenv('STS_OUTER_MATERIAL_DIR', outer_material['material_directory'])
    profile = json.loads(Path('configs/train/combat_two_means_876_4x5090.json').read_text())
    profile['source']['model_sha256'] = outer_material['source_sha256']
    if mode is not None:
        profile['contract']['boss_ablation'] = mode
    profile['training'].update(batch_size_per_rank=8, micro_batch_size=4,
                               optimizer_updates_per_round=1)
    profile['training']['replay']['capacity_per_rank'] = 64
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(json.dumps(profile))
    output = tmp_path / 'initial.zip'
    args = build_parser().parse_args(['--rank', '0', '--world-size', '1',
        '--prepare-combat-outer', str(profile_path), '--material-directory',
        outer_material['material_directory'], '--material-sha256',
        outer_material['material_sha256'], '--output', str(output), '--device', 'cpu',
        '--native-search-threads', '2'])
    prepare(args)
    model = load_checkpoint(output, device='cpu', inference_only=False)
    assert model.num_timesteps == 0 and model.seed == 9001000
    assert not model.policy.optimizer.state
    assert model._ensure_nosl_is_az_replay().size == 0
    assert model.policy.core.lower_initialization == 'random'
    assert model.policy.core.boss_ablation == mode
    assert all(p.requires_grad for p in model.policy.core.combat.parameters())
    assert all(not p.requires_grad for p in model.policy.core.teacher.parameters())
    assert model.outer_return_contract['boss_win_weight'] == (1. if mode else .9)
    if mode:
        assert model.outer_return_contract['boss_ablation']['hp_target'] == mode
    else:
        assert 'boss_ablation' not in model.outer_return_contract


@pytest.mark.parametrize('mode', ['full-or-zero', 'real-hp'])
def test_boss_pure_win_leaf_and_unchanged_nonboss_and_teacher(frozen_upper_source, mode):
    source, space, env, obs = frozen_upper_source
    from env.core.run_env import SlayTheSpireEnv
    extended = SlayTheSpireEnv(global_map_view=True, combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
                              outer_return_context=True, combat_value_objective=MEAN_CONTRACT)
    try:
        extended.reset(seed=42101)
        core = OuterReturnCore(source, space, extended.observation_space, source_sha256='1'*64,
                               objective=MEAN_CONTRACT, boss_ablation=mode)
        network = core.combat
        with torch.no_grad():
            state = torch.randn(2, network.config.width)
            values, outputs = network._value(state, {'combat_is_boss': torch.tensor([[1.], [0.]])})
        p, hp = outputs['win_logits'].sigmoid(), outputs['hp_mean']
        torch.testing.assert_close(values[0], p[0], rtol=0, atol=0)
        torch.testing.assert_close(values[1], (p*hp)[1], rtol=0, atol=0)
        assert ('hp_all_outcomes' in outputs) == (mode == 'full-or-zero')
        assert source.combat.config.boss_win_weight == .9
        assert core.teacher.assert_integrity()
    finally:
        extended.close()


@pytest.mark.parametrize('boss,expected', [(True, 1.), (False, .05)])
def test_exact_search_terminal_win_and_actual_full_hp_label(short_combat_env_type, boss, expected):
    # Enter via the real run continuation so battle exit has a live callback.
    env = short_combat_env_type(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True,
                               global_map_view=True)
    env.reset(seed=42101)
    env.thick.gc.cur_room = sts.Room.BOSS if boss else sts.Room.MONSTER
    env.thick.bc.apply_custom_state({'hand':['THUNDERCLAP'], 'player':{'hp':10,'energy':3}})
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    action = next(int(a) for a in np.flatnonzero(env.action_masks())
                  if C.AUTOREGRESSIVE_PLAY <= a < C.AUTOREGRESSIVE_POTION)
    logits = np.full(env.action_space.n, -1000.); logits[action] = 1000.
    response = env.nosl_is_az_cpp_begin(logits, state_nonce=42, simulations=1, depth=0,
        root_search_mode=0, paper_full_gumbel=False, boss_win_weight=1.)
    assert response['done'] and response['result']['root_value'] == pytest.approx(expected)
    _, _, _, _, info = env.step(action)
    assert info['nosl_value_terminal_target'] == pytest.approx(.05)
    value = boss_ablation_target('full-or-zero', boss=boss, outcome=info['nosl_value_terminal_outcome'],
        value=info['nosl_value_terminal_target'], terminal_max_hp=info['nosl_value_terminal_max_hp'])
    assert value == (info['nosl_value_terminal_max_hp']/200 if boss else .05)
    env.close()


@pytest.mark.parametrize('mode',['full-or-zero','real-hp'])
def test_boss_cuda_graph_shapes_eviction_release_and_weight_change(frozen_upper_source,mode):
    from env.core.run_env import SlayTheSpireEnv
    from rl.search.inference import CombatLeafGraphs
    if not torch.cuda.is_available():
        pytest.fail('explicit CUDA gate requires a GPU')
    source,space,_,_=frozen_upper_source
    env=SlayTheSpireEnv(global_map_view=True,combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
                       outer_return_context=True,combat_value_objective=MEAN_CONTRACT)
    env.reset(seed=42101)
    for _ in range(64):
        if env.thick.bc is not None:break
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.thick.bc is not None
    core=OuterReturnCore(source,space,env.observation_space,source_sha256='1'*64,
                         objective=MEAN_CONTRACT,boss_ablation=mode).to('cuda').eval()
    graph=CombatLeafGraphs(core,'cuda',amp='bf16',batch_multiple=1,capacity=2,compile_forward=False)
    obs=env._obs()
    try:
        for count in (1,2,3,2,4,1):
            batch={k:np.stack([v]*count) for k,v in obs.items()}
            batch['combat_is_boss'][:]=np.arange(count)[:,None]%2
            if count==4:
                with torch.no_grad():core.combat.means.win.bias.add_(.5)
            fresh=CombatLeafGraphs(core,'cuda',amp='bf16',batch_multiple=1,compile_forward=False)
            try:
                actual=graph(batch,1);expected=fresh(batch,1)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
            finally:fresh.clear()
            del batch
            torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
        assert graph.statistics()['evictions']>0
        core.teacher.assert_integrity()
    finally:
        graph.clear();env.close()
