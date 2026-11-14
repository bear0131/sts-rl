from pathlib import Path
import ctypes,hashlib,json
from graph_runtime import torch,ptr,lib,check
P=Path(__file__).resolve().parent
class FrameModules:
 def __init__(self,layout):
  self.handles=[];self.functions={};self.closed=False;self.identity={}
  for kind,receipt,name,functions in [
   ('observer','observer-compile-0-result.json','observer-0.cubin',['frame_observe','frame_observe_special','normal_prepare','normal_features','frame_layout','observer_layout','packed_layout']),
   ('builder','builder-build.json','token_builder.cubin',['pack_short','pack_medium','pack_long','packed_finalize'])]:
   r=json.loads((P/receipt).read_text());assert r['completed'] and r['exit']==0
   binary=P/name;digest=hashlib.sha256(binary.read_bytes()).hexdigest();assert digest==r['cubin_sha256']
   h=ptr();check(lib.cuModuleLoad(ctypes.byref(h),str(binary).encode()),'load frame '+kind);self.handles.append(h)
   self.identity[kind]=dict(sha256=digest)
   for name in functions:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),h,name.encode()),name);self.functions[name]=f
  lib.cuFuncSetAttribute.argtypes=[ptr,ctypes.c_int,ctypes.c_int]
  for name,cap in [('pack_short',32),('pack_medium',64),('pack_long',161)]:
   check(lib.cuFuncSetAttribute(self.functions[name],8,cap*128*4),name+' shared limit')
  out=torch.empty(2,device='cuda',dtype=torch.int64);self.launch('observer_layout',[out],[],1,threads=1)
  assert out.cpu().tolist()==layout
  mode=torch.empty(3,device='cuda',dtype=torch.int64);self.launch('packed_layout',[mode],[],1,threads=1)
  assert mode.cpu().tolist()==[1,16864,8]
  self.identity['observer']['chain_addressing_version']=1
 def launch(self,name,tensors,ints,grid,threads=32,shared=0,after=None):
  assert not self.closed
  params=[ptr(x.data_ptr()) for x in tensors]+[ctypes.c_int(x) for x in ints]
  if after is not None:params.append(ptr(after.data_ptr()))
  args=(ptr*len(params))(*(ctypes.cast(ctypes.byref(x),ptr) for x in params))
  check(lib.cuLaunchKernel(self.functions[name],grid,1,1,threads,1,1,shared,ptr(torch.cuda.current_stream().cuda_stream),args,None),name)
 def close(self):
  if self.closed:return
  torch.cuda.synchronize()
  for h in reversed(self.handles):check(lib.cuModuleUnload(h),'unload frame module')
  self.handles.clear();self.closed=True
