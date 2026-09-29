"""Complete actions retain source entities beyond legacy checkpoint capacities."""
import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv

pytestmark = pytest.mark.native


@pytest.mark.parametrize('card,zone,filler,choices', [
    ('SECRET_TECHNIQUE','draw','STRIKE_RED',['DEFEND_RED','SHRUG_IT_OFF']),
    ('SECRET_WEAPON','draw','DEFEND_RED',['STRIKE_RED','BASH']),
    ('HEADBUTT','discard','DEFEND_RED',['STRIKE_RED','BASH']),
    ('EXHUME','exhaust','EXHUME',['STRIKE_RED','BASH']),
])
def test_native_selection_preserves_long_pile_indices(combat_factory,card,zone,filler,choices):
    env = combat_factory([card]*5, combat_action_contract='native-v1')
    state = {'player':{'energy':3},'hand':[card],'draw':[],'discard':[],'exhaust':[]}
    state[zone] = [filler]*10+choices
    env.thick.bc.apply_custom_state(state)
    env._invalidate_backend_cache()
    bits = next(b for b in env.native_combat_actions() if b >> 29 == 0)
    _, _, terminal, truncated, info = env.step_native_combat(bits)
    assert not terminal and not truncated
    assert info['combat_action_contract']=='native-v1'
    candidates = env.combat_selection_candidates()
    candidate = next(c for c in candidates if c.native_index == 11)
    assert candidate.zone == zone and candidate.card[0] == choices[1]
    branch = env.fork_combat()
    try:
        assert branch.combat_selection_candidates() == candidates
        _, _, terminal, truncated, _ = branch.step_native_combat(candidate.action_bits)
        assert not terminal and not truncated
        assert branch.thick.select_descriptor() is None
        destination = 'draw' if card == 'HEADBUTT' else 'hand'
        assert any(c[0]==choices[1] for c in branch.thick.bc.to_dict()[destination])
        assert env.combat_selection_candidates() == candidates  # branch isolation
    finally:
        branch.close()


def test_native_mode_does_not_relabel_legacy_masks_or_observations(combat_factory):
    legacy = combat_factory(['SECRET_TECHNIQUE']*5)
    legacy.thick.bc.apply_custom_state({'hand':['SECRET_TECHNIQUE'],
        'draw':['STRIKE_RED']*10+['DEFEND_RED','SHRUG_IT_OFF']})
    legacy._invalidate_backend_cache()
    native = legacy.fork_combat()
    native.combat_action_contract = 'native-v1'
    try:
        assert legacy.action_space.n == native.action_space.n == A.N_ACTIONS == 181
        for key,value in legacy._obs().items():
            np.testing.assert_array_equal(value,native._obs()[key])
        old = legacy.step(A._PLAY)
        new = native.step(A._PLAY)
        if legacy._pending:
            old = legacy.step(int(np.flatnonzero(legacy.action_masks())[0]))
        if native._pending:
            new = native.step(int(np.flatnonzero(native.action_masks())[0]))
        assert not old[3] and not old[4].get('dead_screen')
        assert not new[3]
        np.testing.assert_array_equal(legacy.action_masks(), native.action_masks())
        assert np.count_nonzero(legacy.action_masks()) == 2
        assert len(native.native_combat_actions())==2
        with pytest.raises(RuntimeError,match='native-v1'):
            legacy.step_native_combat(native.native_combat_actions()[0])
    finally:
        native.close()


def test_complete_actions_reject_illegal_and_pending_thin_state(combat_factory):
    env = combat_factory(['STRIKE_RED']*5,combat_action_contract='native-v1')
    before = env.thick.bc.to_dict()
    with pytest.raises(ValueError,match='illegal'):
        env.step_native_combat(2 << 29 | 65535)
    assert env.thick.bc.to_dict()==before
    env.step(A._PLAY)
    assert not env.native_combat_actions()
    with pytest.raises(ValueError,match='unfinished'):
        env.step_native_combat(0)


def test_native_multi_includes_nonempty_subsets(combat_factory):
    env = combat_factory(['PURITY']*5,combat_action_contract='native-v1')
    env.thick.bc.apply_custom_state({'hand':['PURITY','STRIKE_RED','DEFEND_RED'],
                                    'draw':[], 'discard':[], 'exhaust':[]})
    env._invalidate_backend_cache()
    env.step_native_combat(0)
    assert {b & 0xffff for b in env.native_combat_actions()}=={0,1,2,3}
    env.step_native_combat(3 << 29 | 2)
    assert env.thick.select_descriptor() is None
    assert [c[0] for c in env.thick.bc.to_dict()['hand']]==['STRIKE_RED']


def test_generated_selection_uses_generated_region(combat_factory):
    env = combat_factory(['DISCOVERY']*5,combat_action_contract='native-v1')
    env.thick.bc.apply_custom_state({'hand':['DISCOVERY']})
    env._invalidate_backend_cache()
    env.step_native_combat(0)
    candidates=env.combat_selection_candidates()
    assert len(candidates)==3 and all(c.zone=='generated' for c in candidates)
    assert [c.card for c in candidates]==[tuple(c) for c in env.thick.bc.select_info()['generated_cards']]
    env.step_native_combat(candidates[2].action_bits)
    assert env.thick.select_descriptor() is None


def test_contract_option_rejects_unknown_version():
    with pytest.raises(ValueError,match='combat_action_contract'):
        SlayTheSpireEnv(combat_action_contract='native-v0')


def test_71_action_checkpoint_roundtrip_prediction_and_environment(tmp_path):
    from sb3_contrib import MaskablePPO
    from env.core.combat_env import SimCombatEnv
    from rl.policies.model import StSMLPExtractor
    env=SimCombatEnv()
    try:
        observation,_=env.reset(seed=42)
        model=MaskablePPO('MultiInputPolicy',env,n_steps=2,batch_size=2,
            policy_kwargs={'features_extractor_class':StSMLPExtractor,
                           'features_extractor_kwargs':{'features_dim':16},'net_arch':[16]},
            seed=42,device='cpu')
        path=tmp_path/'legacy71.zip'
        model.save(path)
        restored=MaskablePPO.load(path,env=env,device='cpu')
        assert restored.action_space.n==71
        mask=env.action_mask()
        action=int(restored.predict(observation,action_masks=mask,deterministic=True)[0])
        assert mask[action]
        assert int(model.predict(observation,action_masks=mask,deterministic=True)[0])==action
        assert not env.step(action)[4].get('illegal',False)
    finally:
        env.close()
