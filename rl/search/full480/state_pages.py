"""Exclusive reusable pages for logically indexed search jobs."""
from dataclasses import dataclass
import heapq
import numpy as np

@dataclass(frozen=True)
class Lease:
 key: object
 serial: int
 count: int
 pages: tuple

class PageLedger:
 def __init__(self,max_pages,page_size=16):
  if type(max_pages) is not int or max_pages<1 or type(page_size) is not int or page_size<1:raise ValueError('positive page limits required')
  self.max_pages=max_pages;self.page_size=page_size;self.capacity=0;self.free=[];self.owned={};self.serial=0
  self.used_jobs=0;self.used_pages=0;self.peak_jobs=0;self.peak_pages=0
 def grow(self,count):
  if type(count) is not int or not 0<count<=self.max_pages-self.capacity:raise ValueError('invalid page growth')
  self.free.extend(range(self.capacity,self.capacity+count));heapq.heapify(self.free);self.capacity+=count
 def preflight(self,specs):
  specs=list(specs);keys=[k for k,n in specs]
  if len(set(keys))!=len(keys) or any(k in self.owned for k in keys):raise ValueError('page owner already exists')
  if any(type(n) is not int or n<1 for k,n in specs):raise ValueError('positive job counts required')
  return sum((n+self.page_size-1)//self.page_size for k,n in specs)
 def claim_many(self,specs):
  specs=list(specs);required=self.preflight(specs)
  if required>len(self.free):raise OverflowError('insufficient free pages')
  out={}
  for key,n in specs:
   pages=tuple(heapq.heappop(self.free) for _ in range((n+self.page_size-1)//self.page_size))
   self.serial+=1;lease=Lease(key,self.serial,n,pages);self.owned[key]=lease;out[key]=lease
   self.used_jobs+=n;self.used_pages+=len(pages)
  self.peak_jobs=max(self.peak_jobs,self.used_jobs);self.peak_pages=max(self.peak_pages,self.used_pages)
  return out
 def release_many(self,leases):
  leases=list(leases)
  if len({x.key for x in leases})!=len(leases) or any(self.owned.get(x.key)!=x for x in leases):raise ValueError('stale, duplicate, or unknown page lease')
  for x in leases:
   del self.owned[x.key];self.used_jobs-=x.count;self.used_pages-=len(x.pages)
   for page in x.pages:heapq.heappush(self.free,page)
 def statistics(self):
  return dict(page_size=self.page_size,capacity_pages=self.capacity,max_pages=self.max_pages,free_pages=len(self.free),owners=len(self.owned),
   leased_jobs=self.used_jobs,leased_pages=self.used_pages,peak_leased_jobs=self.peak_jobs,peak_leased_pages=self.peak_pages)

class DevicePages:
 """Own nonmoving blocks; allocator accepts the number of physical states."""
 def __init__(self,root_capacity,budget_cap,state_bytes,allocator,page_size=16,growth_pages=256):
  if root_capacity<1 or budget_cap<1 or state_bytes<1 or growth_pages<1:raise ValueError('positive arena limits required')
  self.ledger=PageLedger(root_capacity*((budget_cap+page_size-1)//page_size),page_size)
  self.root_capacity=root_capacity;self.budget_cap=budget_cap
  self.state_bytes=state_bytes;self.allocator=allocator;self.growth_pages=growth_pages
  self.blocks=[];self.page_addresses=np.empty(0,np.uint64);self.closed=False
 def acquire(self,specs):
  if self.closed:raise RuntimeError('arena closed')
  specs=list(specs);required=self.ledger.preflight(specs)
  if len(self.ledger.owned)+len(specs)>self.root_capacity or any(n>self.budget_cap for k,n in specs):raise ValueError('root count or per-root budget exceeds arena contract')
  needed=max(0,required-len(self.ledger.free))
  if needed>self.ledger.max_pages-self.ledger.capacity:raise OverflowError('logical root budgets exceed arena capacity')
  if needed:
   count=min(self.ledger.max_pages-self.ledger.capacity,((needed+self.growth_pages-1)//self.growth_pages)*self.growth_pages)
   block=self.allocator(count*self.ledger.page_size);base=int(block.data_ptr())
   addresses=np.uint64(base)+np.arange(count,dtype=np.uint64)*np.uint64(self.ledger.page_size*self.state_bytes)
   combined=np.concatenate((self.page_addresses,addresses))
   self.blocks.append(block);self.page_addresses=combined;self.ledger.grow(count)
  leases=self.ledger.claim_many(specs);pointers={}
  try:
   for key,lease in leases.items():
    base=self.page_addresses[np.asarray(lease.pages,np.int64)]
    pointers[key]=(base[:,None]+np.arange(self.ledger.page_size,dtype=np.uint64)[None,:]*np.uint64(self.state_bytes)).reshape(-1)[:lease.count].copy()
  except BaseException:
   self.ledger.release_many(leases.values());raise
  return leases,pointers
 def release(self,leases):
  if self.closed:raise RuntimeError('arena closed')
  self.ledger.release_many(leases)
 def statistics(self):
  return dict(self.ledger.statistics(),blocks=len(self.blocks),state_bytes=self.state_bytes,allocated_state_bytes=0 if self.closed else self.ledger.capacity*self.ledger.page_size*self.state_bytes,peak_allocated_state_bytes=self.ledger.capacity*self.ledger.page_size*self.state_bytes,closed=self.closed)
 def close(self):
  if self.closed:return
  if self.ledger.owned:raise RuntimeError('cannot release live battle-state blocks')
  self.blocks.clear();self.page_addresses=np.empty(0,np.uint64);self.closed=True
