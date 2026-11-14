"""Finite shared-world task ledger. No simulator or model imports."""
import numpy as np
MASK=(1<<64)-1
GOLDEN=0x9e3779b97f4a7c15
PAIRED_TERMINAL='paired-pcg64-terminal-hp-boss-win-v1'
WORLD_STATE_VERSION='pcg64-bulk-state-v1'

def paired_worlds(nonce,actions,budget):
 """Paired worlds covering the exact budget, including its partial last world.

 The original finite ledger still owns admission, jobs and completion. These
 five words only replace its former counter-draw seed contract. Initialization
 is versioned separately: it need not reproduce NumPy's per-world SeedSequence.
 """
 if actions<1 or budget<1:raise ValueError('paired roots require a legal action and positive budget')
 rng=np.random.default_rng(nonce);count=(budget+actions-1)//actions
 seeds=rng.integers(0,2**63-1,count,dtype=np.int64)
 words=np.empty((count,5),np.uint64);words[:,0]=seeds
 words[:,1:]=rng.bit_generator.random_raw(count*4).reshape(count,4)
 words[:,3]|=np.uint64(1)  # PCG's 128-bit stream increment must be odd.
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
  paired=root.get('contract')==PAIRED_TERMINAL
  anchored=root.get('value_target')=='pre-reward-anchor'
  assert np.isfinite(v).all() if anchored else np.array_equal(np.isfinite(v),terminal)
  visits=np.bincount(idx,minlength=n);terminal_visits=np.bincount(idx[terminal],minlength=n)
  sums=np.bincount(idx if anchored else idx[terminal],weights=v if anchored else v[terminal],minlength=n)
  means=np.divide(sums,visits,out=np.zeros(n),where=visits>0)
  stats=np.asarray(statistics)[select].sum(0)
  rows.append(dict(action_bits=np.asarray(root['action_bits'],np.uint32),visits=visits,terminal_visits=terminal_visits,values=means,
   simulations=len(select),actual_transitions=int(stats[0])+len(select),nonforced_decisions=int(stats[1]),forced_actions=int(stats[2]),
   depth_cuts=int((~terminal).sum()),statistics=stats))
  if paired:
   rows[-1].update(atomic_decisions=int(stats[7]),actual_transitions=int(stats[7])-int(stats[11]),
    selection_prefix_decisions=int(stats[11]),value_statistic='pre-reward-anchor' if anchored else PAIRED_TERMINAL)
 return rows
