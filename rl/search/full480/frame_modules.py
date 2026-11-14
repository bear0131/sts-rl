from .run_device import artifact_directory
from pathlib import Path
import ctypes,hashlib,json
from .graph_runtime import torch,ptr,lib,check
P=artifact_directory()

def build_deck_artifacts(inputs,*,snapshot,canonical_header,nvrtc,nvcc,directory=None,kinds=('observer','rule','builder')):
 """Original NVRTC observer and nvcc builder, with explicit consumer inputs.

 The simulator stays in the qualified external snapshot. No generated game
 source, checkpoint or binary is copied into the repository.
 """
 import subprocess,time
 from .deck_inputs import portable_compact_header
 output=Path(directory or P/'deck');output.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).resolve().parent;snapshot=Path(snapshot)
 original=snapshot/'devicecompact141';header=snapshot/'deviceflex141/device_std.h'
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert digest(header)=='b4295d78a2df81e696177d5f97f2d5a1b07de6aa09c5d0570831a6335b727e51'
 assert (original/'generated/include/combat/BattleContext.h').is_file()
 (output/'deck_schema.h').write_text(inputs.header())
 (output/'compact-observation-device.h').write_text(portable_compact_header(canonical_header))
 compiled={}
 for kind,filename in [('observer','observer_entry.cu'),('rule','rules_entry.cu')]:
  if kind not in kinds:continue
  generated=snapshot/('deviceflex141/generated' if kind=='rule' else 'devicecompact141/generated')
  if kind=='rule':
   manifest=json.loads((snapshot/'deviceflex141/outlined-source-manifest.json').read_text())
   assert len(manifest['selected_functions'])==185 and manifest['factory_count']==99
   for name,row in manifest['overrides'].items():assert digest(generated/name)==row['sha256'],('original outlined method differs',name)
   # Retain the qualified outlined simulator. Add only the native queue's
   # data-only opening continuation and its unchanged Actions consumers.
   simulator=Path(canonical_header).resolve().parents[1]
   continuation=output/'continuation-simulator'
   continuation_sources={}
   for relative,marker,anchor in [
    ('include/combat/Actions.h','PORTABLE OPENING ACTION','    class CardQueueItem;'),
    ('include/combat/ActionQueue.h','PENDING ACTION ACCESS','        [[nodiscard]] int getCapacity() const;'),
    ('src/combat/Actions.cpp','PORTABLE OPENING ACTION','using namespace sts;')]:
    native=simulator/relative;text=native.read_text();begin='// BEGIN '+marker;end='// END '+marker
    section=text[text.index(begin):text.index(end)+len(end)]
    section=section.replace('throw std::logic_error("unknown opening continuation");',
                            'gpu_fail(__LINE__); return {};')
    original_text=(generated/relative).read_text();assert original_text.count(anchor)==1,(relative,anchor)
    target=continuation/relative;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(original_text.replace(anchor,anchor+'\n'+section))
    continuation_sources[relative]=dict(native_sha256=digest(native),generated_sha256=digest(target))
   relative='src/combat/BattleContext.cpp'
   original_text=(generated/relative).read_text()
   old='addToBot(Actions::OnAfterCardUsed());'
   replacement='addToBot(OpeningAction::create(OpeningAction::AfterCardUsed));'
   assert original_text.count(old)==1 and replacement in (simulator/relative).read_text()
   target=continuation/relative;target.parent.mkdir(parents=True,exist_ok=True)
   target.write_text(original_text.replace(old,replacement))
   continuation_sources[relative]=dict(native_sha256=digest(simulator/relative),generated_sha256=digest(target))
  entry=(source/filename).read_text()
  old='"/root/autodl-tmp/token-speed-r7-20261105/nine-token-speed/online480/waves/devicebudgetarena141/device_std.h"'
  assert entry.count(old)==1
  entry=entry.replace(old,'"'+str(header)+'"')
  options=['--std=c++17','--gpu-architecture=sm_120','--device-as-default-execution-space','--fmad=false',
   '--generate-line-info','--display-error-number','--diag-suppress=177','-DSTS_FULL480_DECK=1',
   '-I'+str(generated),'-I'+str(generated/'include'),'-I'+str(original),'-I'+str(source),'-I'+str(output)]
  if kind=='rule':options[0:0]=['-I'+str(continuation),'-I'+str(continuation/'include')]
  compiler=ctypes.CDLL(str(nvrtc));program=ptr()
  major=ctypes.c_int();minor=ctypes.c_int();assert compiler.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))==0
  assert major.value==13,'full480 generated rules require the archived CUDA 13 compiler'
  for name,args in [('nvrtcCreateProgram',[ctypes.POINTER(ptr),ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ptr,ptr]),
   ('nvrtcCompileProgram',[ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]),
   ('nvrtcGetProgramLogSize',[ptr,ctypes.POINTER(ctypes.c_size_t)]),('nvrtcGetProgramLog',[ptr,ptr]),
   ('nvrtcGetCUBINSize',[ptr,ctypes.POINTER(ctypes.c_size_t)]),('nvrtcGetCUBIN',[ptr,ptr]),
   ('nvrtcDestroyProgram',[ctypes.POINTER(ptr)])]:getattr(compiler,name).argtypes=args
  assert compiler.nvrtcCreateProgram(ctypes.byref(program),entry.encode(),str(source/filename).encode(),0,None,None)==0
  sources={name:digest(source/name) for name in ((filename,'deck_inputs.cuh') if kind=='observer' else (filename,'native/include/state_codec.h'))}
  dependencies={str(p.relative_to(snapshot)):digest(p) for p in original.rglob('*') if p.suffix in ('.h','.cuh','.cpp')}
  dependencies[str(header.relative_to(snapshot))]=digest(header)
  if kind=='rule':dependencies.update({str(p.relative_to(snapshot)):digest(p) for p in generated.rglob('*') if p.is_file()})
  record=dict(completed=False,options=options,input_sha256=hashlib.sha256(entry.encode()).hexdigest(),
   compiler=dict(version=[major.value,minor.value],sha256=digest(nvrtc)),
   package_sources=sources,dependencies=dependencies,canonical_header_sha256=digest(canonical_header),
   schema_sha256=digest(output/'deck_schema.h'),codec_sha256=digest(output/'compact-observation-device.h'))
  if kind=='rule':record.update(state_schema=2,continuation_sources=continuation_sources)
  path=output/(kind+'-compile-0-result.json');path.write_text(json.dumps(record,indent=2))
  start=time.monotonic()
  try:
   opts=(ctypes.c_char_p*len(options))(*(s.encode() for s in options))
   rc=compiler.nvrtcCompileProgram(program,len(options),opts)
   size=ctypes.c_size_t();compiler.nvrtcGetProgramLogSize(program,ctypes.byref(size))
   buffer=ctypes.create_string_buffer(size.value);compiler.nvrtcGetProgramLog(program,buffer)
   (output/(kind+'-compile-0.log')).write_bytes(buffer.value)
   record.update(completed=True,exit=rc,seconds=time.monotonic()-start)
   if rc==0:
    compiler.nvrtcGetCUBINSize(program,ctypes.byref(size));buffer=ctypes.create_string_buffer(size.value)
    compiler.nvrtcGetCUBIN(program,buffer);(output/(kind+'-0.cubin')).write_bytes(bytes(buffer))
    record.update(cubin_bytes=size.value,cubin_sha256=digest(output/(kind+'-0.cubin')))
   path.write_text(json.dumps(record,indent=2))
   if rc:raise RuntimeError('full480 '+kind+' compilation failed: '+str(output/(kind+'-compile-0.log')))
  finally:compiler.nvrtcDestroyProgram(ctypes.byref(program))
  compiled[kind]=record
 if 'builder' not in kinds:return compiled
 args=[str(nvcc),'-std=c++17','-arch=sm_120','-cubin','--fmad=false','-Xptxas=-v','-DSTS_FULL480_DECK=1',
  str(source/'token_builder.cu'),'-o',str(output/'token_builder.cubin')]
 start=time.monotonic();result=subprocess.run(args,capture_output=True,text=True,timeout=120)
 (output/'builder-build.log').write_text(result.stdout+result.stderr)
 builder=dict(completed=True,exit=result.returncode,seconds=time.monotonic()-start,options=args,
  package_sources={'token_builder.cu':digest(source/'token_builder.cu')})
 if result.returncode==0:builder['cubin_sha256']=digest(output/'token_builder.cubin')
 (output/'builder-build.json').write_text(json.dumps(builder,indent=2))
 if result.returncode:raise RuntimeError('full480 builder compilation failed: '+str(output/'builder-build.log'))
 return dict(compiled,builder=builder)

class FrameModules:
 def __init__(self,layout,*,deck=False,schema=None):
  self.handles=[];self.functions={};self.closed=False;self.identity={}
  try:self.load(layout,deck,schema)
  except BaseException:
   self.close();raise
 def load(self,layout,deck,schema):
  for kind,receipt,name,functions in [
   ('observer','observer-compile-0-result.json','observer-0.cubin',['frame_observe','frame_observe_special','normal_prepare','normal_features','frame_layout','observer_layout','packed_layout']),
   ('builder','builder-build.json','token_builder.cubin',['pack_short','pack_medium','pack_long','packed_finalize'])]:
   directory=P/'deck' if deck else P
   r=json.loads((directory/receipt).read_text());assert r['completed'] and r['exit']==0
   if deck:
    if kind=='observer':
     assert schema is not None and hashlib.sha256(schema.encode()).hexdigest()==r['schema_sha256'],'full480 input schema does not match compiled observer'
    for filename,digest in r['package_sources'].items():
     assert hashlib.sha256((Path(__file__).resolve().parent/filename).read_bytes()).hexdigest()==digest,('stale full480 artifact',filename)
    if kind=='observer':functions=functions+['deck_anchor_inputs','deck_context_misc','deck_fallback_features']
   binary=directory/name;digest=hashlib.sha256(binary.read_bytes()).hexdigest();assert digest==r['cubin_sha256']
   h=ptr();check(lib.cuModuleLoad(ctypes.byref(h),str(binary).encode()),'load frame '+kind);self.handles.append(h)
   self.identity[kind]=dict(sha256=digest)
   for name in functions:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),h,name.encode()),name);self.functions[name]=f
  lib.cuFuncSetAttribute.argtypes=[ptr,ctypes.c_int,ctypes.c_int]
  for name,cap in [('pack_short',32),('pack_medium',64),('pack_long',161)]:
   check(lib.cuFuncSetAttribute(self.functions[name],8,cap*128*4),name+' shared limit')
  out=torch.empty(2,device='cuda',dtype=torch.int64);self.launch('observer_layout',[out],[],1,threads=1)
  actual=out.cpu().tolist();assert actual==layout,(actual,layout)
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
