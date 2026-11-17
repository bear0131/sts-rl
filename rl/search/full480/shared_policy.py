"""Reuse deck action scores only for bitwise-equal inputs in the current frame.

The hash proposes one representative; full comparison decides whether it is safe
to omit a row. The table is cleared on every replay, so it never reuses scores
across frames or weight versions. Only ordinary action rows participate. Root
and value-only scoring must retain their own value outputs.

Deck action heads read keys[:, 0], legal actions, selected cards, remaining
selections and the packed token inputs. Entity order and action order are also
compared. Hash collisions can lose sharing opportunities, never change scores.
Temporary action counts are restored even when packing exceeds its capacity.
"""
import triton
import triton.language as tl

@triton.jit
def propose(META,CHUNK,FEAT,IDS,DISPATCH,FLAGS,TABLE,HASH,CAP:tl.constexpr):
 if tl.load(FLAGS+2)==0:
  count=tl.load(DISPATCH)
  d=tl.arange(0,32)
  for sid in range(tl.program_id(0),count,tl.num_programs(0)):
   na=tl.load(META+sid*4);ne=tl.load(META+sid*4+1)
   if (na>1)&(na<=256)&(ne>0)&(ne<=121)&(tl.load(META+sid*4+3)==1):
    pos=d%ne;at=tl.load(CHUNK+sid*16+pos//64)+pos%64
    sample=tl.load(FEAT+at*64+(d*13)%64).to(tl.uint32,bitcast=True)
    slot=tl.load(IDS+at).to(tl.uint32)
    x=(sample ^ (slot*0x9e3779b9))*(1+d.to(tl.uint32)*2)
    x=(x^(x>>16))*0x85ebca6b
    h=tl.sum(x,0).to(tl.uint32)^(na.to(tl.uint32)*0x27d4eb2d)
    h=((h^(h>>16))*0x85ebca6b).to(tl.uint32);key=(h^(h>>13))&(CAP-1)
    tl.store(HASH+sid,key);tl.atomic_min(TABLE+key,sid,sem='relaxed')

@triton.jit
def compare(META,CHUNK,FEAT,IDS,ZONES,MON,KEYS,BITS,LEGAL,SEL,REM,DISPATCH,FLAGS,TABLE,HASH,REPS,SAVED,OWNERS):
 if tl.load(FLAGS+2)==0:
  count=tl.load(DISPATCH);d=tl.arange(0,256);e=tl.arange(0,512)
  for sid in range(tl.program_id(0),count,tl.num_programs(0)):
   na=tl.load(META+sid*4);ne=tl.load(META+sid*4+1)
   tl.store(REPS+sid,-1)
   if (na>1)&(na<=256)&(ne>0)&(ne<=121)&(tl.load(META+sid*4+3)==1):
    other=tl.load(TABLE+tl.load(HASH+sid));mb=META+other*4
    if (other!=sid)&(na==tl.load(mb))&(ne==tl.load(mb+1))&(tl.load(META+sid*4+3)==tl.load(mb+3)):
     a=tl.load(META+sid*4+2);b=tl.load(mb+2)
     differ=(tl.load(BITS+a+d,d<na,0)!=tl.load(BITS+b+d,d<na,0)) | (tl.load(KEYS+(a+d)*10,d<na,0)!=tl.load(KEYS+(b+d)*10,d<na,0))
     differ |= tl.load(LEGAL+sid*198+d,d<198,0)!=tl.load(LEGAL+other*198+d,d<198,0)
     differ |= tl.load(SEL+sid*10+d,d<10,0).to(tl.int32,bitcast=True)!=tl.load(SEL+other*10+d,d<10,0).to(tl.int32,bitcast=True)
     at=tl.load(CHUNK+sid*16+d//64,d<ne,0)+d%64;bt=tl.load(CHUNK+other*16+d//64,d<ne,0)+d%64
     differ |= (tl.load(IDS+at,d<ne,0)!=tl.load(IDS+bt,d<ne,0)) | (tl.load(ZONES+at,d<ne,0)!=tl.load(ZONES+bt,d<ne,0)) | (tl.load(MON+at,d<ne,0)!=tl.load(MON+bt,d<ne,0))
     equal=(tl.sum(differ.to(tl.int32),0)==0)&(tl.load(REM+sid)==tl.load(REM+other))
     block=0
     while equal & (block*512<ne*64):
      j=block*512+e;t=j//64
      x=tl.load(CHUNK+sid*16+t//64,t<ne,0)+t%64;y=tl.load(CHUNK+other*16+t//64,t<ne,0)+t%64
      u=tl.load(FEAT+x*64+e%64,t<ne,0).to(tl.uint32,bitcast=True);v=tl.load(FEAT+y*64+e%64,t<ne,0).to(tl.uint32,bitcast=True)
      equal=tl.sum((u!=v).to(tl.int32),0)==0;block+=1
     if equal:
      tl.store(REPS+sid,other);tl.store(SAVED+sid,na)
      tl.store(OWNERS+a+d,sid,d<na)
      tl.store(META+sid*4,1)

@triton.jit
def scatter(META,SCORE,DISPATCH,SUMMARY,REPS,SAVED,TOTAL):
 n=tl.load(DISPATCH);d=tl.arange(0,256);sumrows=0
 for sid in range(tl.program_id(0),n,tl.num_programs(0)):
  k=tl.load(REPS+sid)
  if k>=0:
   na=tl.load(SAVED+sid);tl.store(META+sid*4,na)
   if tl.load(SUMMARY+2)==0:
    a=tl.load(META+sid*4+2);b=tl.load(META+k*4+2)
    tl.store(SCORE+a+d,tl.load(SCORE+b+d,d<na,0.),d<na);sumrows+=1
 if sumrows:
  tl.atomic_add(SUMMARY+9,sumrows,sem='relaxed');tl.atomic_add(TOTAL,sumrows,sem='relaxed')

def prepare(p,w,n):
 cap=w['shared_table'].numel();w['shared_table'].fill_(2147483647);w['shared_rep'].fill_(-1)
 propose[(n,)](p.meta,p.chunks,p.numbers,p.ids,w['dispatch'],p.allocator,w['shared_table'],w['shared_hash'],cap,num_warps=1)
 compare[(n,)](p.meta,p.chunks,p.numbers,p.ids,p.zones,p.monsters,p.keys,p.bits,w['legal'],w['selected'],w['remaining'],w['dispatch'],p.allocator,w['shared_table'],w['shared_hash'],w['shared_rep'],w['shared_saved'],w['owners'],num_warps=4)
def finish(p,w,n):
 scatter[(n,)](p.meta,p.scores,w['dispatch'],w['summary'],w['shared_rep'],w['shared_saved'],w['shared_total'],num_warps=4)

