"""The lower replacement must retain the actual TRAIN47 upper runtime contract."""
import json
from pathlib import Path

from rl.training.distributed_sb3 import build_parser


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
    assert (args.n_envs,args.n_steps)==(384,512)
    # COMBAT18 saved manifest: 102400 per rank, hence 409600 over four ranks.
    assert args.small_playout_batch_size == 102400
    assert json.loads(lines[lines.index('--small-playout-online')+1])['batch_size']==102400

    assert args.small_playout_micro_batch_size == 8192
