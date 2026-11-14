from .run_device import artifact_directory
from pathlib import Path
import sys,inspect,textwrap
P=artifact_directory()

from .flex_ragged import *
from .flex_ragged import BattlePool as BasePool
from . import ragged_runtime
from .graph_model import GraphScorer
source=textwrap.dedent(inspect.getsource(ragged_runtime.RaggedBattlePool.score))
assert source.count('ragged_forward(')==1
namespace=dict(vars(ragged_runtime));exec(source.replace('ragged_forward(','self.graph_scorer('),namespace)
class BattlePool(BasePool):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  scorer=GraphScorer
  if getattr(self.flat,'consumer',None)=='deck-latent-combat-v1':
   from .combo_model import GraphScorer as scorer
  self.graph_scorer=scorer(self.flat)
 def _buffer(self,name,count,tail=(),dtype=torch.float32):
  old=getattr(self,'ragged_buffers',{}).get(name)
  if old is None or len(old)<count:
   if hasattr(self,'graph_scorer'):self.graph_scorer.release()
  return BasePool._buffer(self,name,count,tail,dtype)
 score=namespace['score']
 def close(self):
  if hasattr(self,'graph_scorer') and not self.graph_scorer.closed:self.graph_scorer.close()
  super().close()
