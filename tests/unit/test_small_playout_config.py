"""The lower replacement must retain the actual TRAIN47 upper runtime contract."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.training.distributed_sb3 import build_parser


@pytest.mark.parametrize('invalid', ['resume-change', 'replay', 'reasoning', 'auxiliary', 'backend'])
def test_lower_only_scope_rejects_incompatible_training(invalid):
    from rl.training.distributed_sb3 import configure_training_scope
    args = build_parser().parse_args(['--rank', '0', '--world-size', '1',
        '--training-scope', 'lower', '--vec', 'native', '--no-nosl-is-az-replay',
        '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning'])
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(joint_upper=True),
        small_playout=object(), upper_auxiliary_config=None), nosl_is_az=True,
        nosl_is_az_inner_only=True, nosl_is_az_combat_strategy='small-playout')
    if invalid == 'resume-change':
        args.resume_state = Path('existing-run')
    elif invalid == 'replay':
        args.nosl_is_az_replay_enabled = True
    elif invalid == 'reasoning':
        args.reasoning_updates = 1
    elif invalid == 'auxiliary':
        model.policy.upper_auxiliary_config = {'loss_coef': .02}
    else:
        args.vec = 'dummy'
    with pytest.raises(ValueError, match='training scope'):
        configure_training_scope(model, args)
    assert not hasattr(model, 'training_scope')


def test_lower_replacement_preserves_upper_except_authorized_memory_chunk():
    # Audited TRAIN47/876 model 693670e4, global step 688914432.
    expected=dict(width=256,layers=4,heads=8,learning_rate=.0001,epochs=10,
                  micro_batch_size=4096,gamma=1.,gae_lambda=1.,clip=.2,
                  entropy_coef=.02,value_coef=.5,max_grad_norm=.5,target_kl=.02)
    # User explicitly allows 2048 after public-context4096 exhausts 5090 VRAM.
    expected['micro_batch_size']=2048
    root=Path(__file__).resolve().parents[2]
    lines=(root/'configs/train/small_playout_fresh.args').read_text().splitlines()
    assert json.loads(lines[lines.index('--upper-config')+1]) == expected
    args=build_parser().parse_args(['--rank','0','--world-size','4','--checkpoint','unused.zip',
        '@'+str(root/'configs/train/small_playout_distributed.args')])
    assert args.upper_micro_batch_size == expected['micro_batch_size']
    assert args.upper_gae_lambda == expected['gae_lambda']
    # Explicit follow-up experiment from COMBAT23 round215; fresh historical
    # recipe remains intact, while continuation requests thirty upper updates.
    assert args.upper_epochs == 30
    assert (args.n_envs,args.n_steps)==(384,512)
    # COMBAT18 saved manifest: 102400 per rank, hence 409600 over four ranks.
    assert args.small_playout_batch_size == 102400
    assert json.loads(lines[lines.index('--small-playout-online')+1])['batch_size']==102400

    assert args.small_playout_micro_batch_size == 8192
