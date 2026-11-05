"""Public combat context adapter for the mature compact policy.

Only entity facts enter the sketch. Hashing shares weights between collisions;
the opaque execution handles and the simulator's unknown draw order are absent.
"""
import numpy as np
import torch
from torch import nn

from rl.policies.small_playout import SmallPlayoutPolicy, PreparedPlayoutSamples


PUBLIC_CONTRACT = 'sts1-playout-public-entities-v1'
SKETCH_SIZE = 1024


def _hash32(x):
    x = x.astype(np.uint32, copy=False)
    x = (x ^ (x >> 16)) * np.uint32(0x7feb352d)
    x = (x ^ (x >> 15)) * np.uint32(0x846ca68b)
    return x ^ (x >> 16)


def public_sketch(table, lengths):
    """A signed count and a signed continuous value per nonzero property.

    All rows (including master inventory) contribute. Property columns are
    conditioned on kind, identity, zone and actual monster slot. An additional
    kind-level numeric channel shares evidence across uncommon cards/monsters.
    """
    ids, zones, monsters = (np.asarray(table[k], np.int32) for k in ('ids','zones','monsters'))
    numbers = np.asarray(table['numbers'], np.float32)
    lengths = np.asarray(lengths, np.int64)
    if (lengths.ndim != 1 or (lengths <= 0).any() or lengths.sum() != len(ids)
            or ids.shape != zones.shape or ids.shape != monsters.shape
            or numbers.shape != (len(ids), 16) or not np.isfinite(numbers).all()
            or (ids < 0).any() or (ids >= 12*512).any() or (zones < 0).any()
            or (zones >= 15).any() or (monsters < -1).any() or (monsters >= 5).any()):
        raise ValueError('invalid native public combat entity table')
    owners = np.repeat(np.arange(len(lengths)),lengths)
    identity = (ids*15 + zones)*6 + monsters + 1
    kind = ((ids//512*15+zones)*6 + monsters+1) + 12*512*15*6
    result = np.zeros((len(lengths),SKETCH_SIZE),np.float32)
    def add(owner, name, value):
        h = _hash32(name)
        index = h & (SKETCH_SIZE-1)
        np.add.at(result, (owner, index), np.where(h>>31,-value,value))
    add(owners,identity*17,1.)
    rows, cols = np.nonzero(numbers)
    values = numbers[rows,cols]
    add(owners[rows],identity[rows]*17+cols+1,values)
    add(owners[rows],kind[rows]*17+cols+1,values)
    return result / np.sqrt(lengths.astype(np.float32))[:,None]


class PublicPlayoutPolicy(SmallPlayoutPolicy):
    """Mature policy plus a context shift before its first hidden activation."""
    def __init__(self, key_count, context_mode='root'):
        super().__init__(key_count)
        if context_mode not in ('root','state'):
            raise ValueError('unknown public context refresh mode')
        self.context_mode=context_mode
        self.context = nn.Linear(SKETCH_SIZE,64,bias=False)
        nn.init.zeros_(self.context.weight)

    @classmethod
    def from_mature(cls, net):
        out = cls(net.key_count).to(net.fc1.weight.device)
        result = out.load_state_dict(net.state_dict(),strict=False)
        if result.missing_keys != ['context.weight'] or result.unexpected_keys:
            raise ValueError('requires a mature compact v1 policy')
        return out

    def forward(self, state, actions, keys, context, mask):
        z = self.fc2(torch.relu(self.fc1(state) + self.context(context)))
        return self._score_actions(z[:,None],actions,keys).masked_fill(~mask,-torch.inf)

    def forward_packed(self,state,actions,keys,slots,context,mask):
        z = self.fc2(torch.relu(self.fc1(state) + self.context(context)))
        return self._score_packed(z,actions,keys,slots,mask)

    def prepare_samples(self,samples,target_mix):
        return PreparedPublicSamples(samples,target_mix)

    def export(self):
        # Never silently truncate this model to the v1 simulator ABI.
        return np.concatenate((super().export(),self.context.weight.detach().cpu().numpy().ravel()))


class PreparedPublicSamples(PreparedPlayoutSamples):
    def __init__(self,samples,target_mix):
        super().__init__(samples,target_mix)
        self.public=np.stack([row['public_context'] for row in samples])
        if self.public.shape!=(len(samples),SKETCH_SIZE) or not np.isfinite(self.public).all():
            raise ValueError('invalid public training context')

    def batch(self,indices,device):
        inputs,target,teacher=super().batch(indices,device)
        return (*inputs[:-1],torch.as_tensor(self.public[indices],dtype=torch.float32,device=device),inputs[-1]),target,teacher
