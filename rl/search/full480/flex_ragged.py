from pathlib import Path
import sys
from flex_runtime import *
from flex_runtime import BattlePool as FlexBattlePool
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'deviceragged141'))
from ragged_runtime import RaggedBattlePool
class BattlePool(FlexBattlePool):
 _buffer=RaggedBattlePool._buffer
 score=RaggedBattlePool.score
 def close(self):
  if self.closed:return
  FlexBattlePool.close(self)
  if hasattr(self,'ragged_buffers'):self.ragged_buffers.clear()
