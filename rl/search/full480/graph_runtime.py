from pathlib import Path
import sys,inspect,textwrap
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'deviceflex141'))
from flex_ragged import *
from flex_ragged import BattlePool as BasePool
import ragged_runtime
from graph_model import GraphScorer
source=textwrap.dedent(inspect.getsource(ragged_runtime.RaggedBattlePool.score))
assert source.count('ragged_forward(')==1
namespace=dict(vars(ragged_runtime));exec(source.replace('ragged_forward(','self.graph_scorer('),namespace)
class BattlePool(BasePool):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.graph_scorer=GraphScorer(self.flat)
 def _buffer(self,name,count,tail=(),dtype=torch.float32):
  old=getattr(self,'ragged_buffers',{}).get(name)
  if old is None or len(old)<count:
   if hasattr(self,'graph_scorer'):self.graph_scorer.release()
  return BasePool._buffer(self,name,count,tail,dtype)
 score=namespace['score']
 def close(self):
  if hasattr(self,'graph_scorer') and not self.graph_scorer.closed:self.graph_scorer.close()
  super().close()
