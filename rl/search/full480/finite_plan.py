"""Finite shared-world task ledger. No simulator or model imports."""
import numpy as np
MASK=(1<<64)-1
GOLDEN=0x9e3779b97f4a7c15
PAIRED_ANCHOR='paired-pcg64-pre-reward-anchor-v1'

def paired_worlds(nonce,actions,budget):
 """Current consumer's complete worlds and NumPy PCG64 initial state.

 The original finite ledger still owns admission, jobs and completion. These
 five words only replace its former counter-draw seed contract.
 """
 if actions<2 or budget<1:raise ValueError('paired roots require choices and a positive budget')
 seeds=np.random.default_rng(nonce).integers(0,2**63-1,max(1,budget//actions),dtype=np.int64)
 words=np.empty((len(seeds),5),np.uint64)
 for i,seed in enumerate(seeds):
  state=np.random.PCG64(int(seed)^0x504F4C494359).state['state']
  words[i]=[seed,state['state']&MASK,state['state']>>64,state['inc']&MASK,state['inc']>>64]
 return words
def mix(value):
 value=(int(value)+GOLDEN)&MASK
 value=((value^(value>>30))*0xbf58476d1ce4e5b9)&MASK
 value=((value^(value>>27))*0x94d049bb133111eb)&MASK
 return (value^(value>>31))&MASK
def uniform(policy_seed,draw_index):
 return (mix((int(policy_seed)+GOLDEN*int(draw_index))&MASK)>>11)*2.**-53
def build_plan(roots):
 columns={k:[] for k in ['root','local','action_index','action_bits','world_index','world_seed','policy_seed']}
 for root_index,root in enumerate(roots):
  assert root['eligible'],'Nonportable or pending roots require the existing CPU search path'
  n=len(root['action_bits']);assert n>0 and sorted(root['order'])==list(range(n))
  budget=int(root['budget']);assert budget>0
  assert 0<=root['depth_limit']<=200 and np.isfinite(root['temperature']) and root['temperature']>0 and root['max_hp']>0
  for j in range(budget):
   action=root['order'][j%n];world_index=j//n;world_seed=mix((mix(root['nonce'])+world_index)&MASK)
   vals=dict(root=root_index,local=j,action_index=action,action_bits=root['action_bits'][action],world_index=world_index,world_seed=world_seed,policy_seed=mix(world_seed^0x504149524544))
   for k,v in vals.items():columns[k].append(v)
 types={'root':np.int32,'local':np.int32,'action_index':np.int32,'action_bits':np.uint32,'world_index':np.int32,'world_seed':np.uint64,'policy_seed':np.uint64}
 return {k:np.asarray(v,dtype=types[k]) for k,v in columns.items()}
def aggregate(roots,plan,values,outcomes,statistics):
 """Preserve original binding: finite terminal sums / all visits, plus terminalVisits."""
 rows=[]
 assert len(values)==sum(int(r['budget']) for r in roots)==len(plan['root'])
 for i,root in enumerate(roots):
  select=np.flatnonzero(plan['root']==i);idx=plan['action_index'][select];n=len(root['action_bits'])
  assert len(select)==root['budget']
  outcome=np.asarray(outcomes)[select];terminal=outcome!=0;v=np.asarray(values)[select]
  paired=root.get('contract')==PAIRED_ANCHOR
  if paired:
   assert np.isin(outcome,[1,2,3,4]).all() and np.isfinite(v).all()
   terminal=outcome!=4
  else:assert np.array_equal(np.isfinite(v),terminal)
  visits=np.bincount(idx,minlength=n);terminal_visits=np.bincount(idx[terminal],minlength=n)
  sums=np.bincount(idx if paired else idx[terminal],weights=v if paired else v[terminal],minlength=n)
  if paired:assert (visits==len(root['worlds'])).all()
  means=np.divide(sums,visits,out=np.zeros(n),where=visits>0)
  stats=np.asarray(statistics)[select].sum(0)
  rows.append(dict(action_bits=np.asarray(root['action_bits'],np.uint32),visits=visits,terminal_visits=terminal_visits,values=means,
   simulations=len(select),actual_transitions=int(stats[0])+len(select),nonforced_decisions=int(stats[1]),forced_actions=int(stats[2]),
   depth_cuts=int((~terminal).sum()),statistics=stats))
  if paired:
   rows[-1].update(atomic_decisions=int(stats[7]),actual_transitions=int(stats[7])-int(stats[11]),
    selection_prefix_decisions=int(stats[11]),value_statistic=PAIRED_ANCHOR)
 return rows
