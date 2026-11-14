"""Exclusive root-slot leases; generations prevent stale root reuse."""
from dataclasses import dataclass
import heapq
@dataclass(frozen=True)
class Handle:
 slot:int
 generation:int
 key:object
class RootLedger:
 def __init__(self,capacity):
  if capacity<1:raise ValueError('positive root capacity required')
  self.capacity=capacity;self.free=list(range(capacity));self.generations=[0]*capacity;self.pending={}
 def claim(self,keys):
  keys=list(keys)
  if len(set(keys))!=len(keys) or any(k in self.pending for k in keys):raise ValueError('root key already in flight')
  if len(keys)>len(self.free):raise OverflowError('no free root slots')
  out=[]
  for key in keys:
   slot=heapq.heappop(self.free);self.generations[slot]+=1
   h=Handle(slot,self.generations[slot],key);self.pending[key]=h;out.append(h)
  return out
 def release(self,handle):
  if self.pending.get(handle.key)!=handle:raise ValueError('stale or unknown root handle')
  del self.pending[handle.key];heapq.heappush(self.free,handle.slot)
 def __len__(self):return len(self.pending)
