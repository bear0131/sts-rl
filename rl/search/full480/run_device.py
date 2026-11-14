from pathlib import Path
import ctypes,json,time,subprocess,os,hashlib
import torch
P=Path(__file__).resolve().parent
assert json.loads((P/'compile-result.json').read_text())['exit']==0
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
torch.cuda.init()
_context_anchor=torch.zeros(1,device="cuda")
lib=ctypes.CDLL('libcuda.so.1');ptr=ctypes.c_void_p;u32=ctypes.c_uint
def check(code,where):
    if code:raise RuntimeError((where,code))
lib.cuModuleLoad.argtypes=[ctypes.POINTER(ptr),ctypes.c_char_p]
lib.cuModuleGetFunction.argtypes=[ctypes.POINTER(ptr),ptr,ctypes.c_char_p]
lib.cuLaunchKernel.argtypes=[ptr,u32,u32,u32,u32,u32,u32,u32,ptr,ctypes.POINTER(ptr),ptr]
lib.cuCtxSynchronize.argtypes=[]
lib.cuCtxSetLimit.argtypes=[ctypes.c_int,ctypes.c_size_t]
check(lib.cuCtxSetLimit(0,65536),'stack limit')
check(lib.cuCtxSetLimit(2,512*1024*1024),'device heap limit')
module=ptr();check(lib.cuModuleLoad(ctypes.byref(module),str(P/'device_battle.cubin').encode()),'load')
result=dict(device=torch.cuda.get_device_name(),cuda=torch.version.cuda,
            cubin_sha256=hashlib.sha256((P/'device_battle.cubin').read_bytes()).hexdigest(),
            scope='device rule execution smoke only, not a complete combat or throughput qualification')
for name in ('device_layout','device_rule_probe'):
    fn=ptr();check(lib.cuModuleGetFunction(ctypes.byref(fn),module,name.encode()),'getfunction')
    out=torch.zeros(2,dtype=torch.int64,device='cuda')
    addr=ptr(out.data_ptr())
    if name=='device_rule_probe':
        arena=torch.zeros(result['device_layout']['values'][0],dtype=torch.uint8,device='cuda');state=ptr(arena.data_ptr());args=(ptr*2)(ctypes.cast(ctypes.byref(state),ptr),ctypes.cast(ctypes.byref(addr),ptr))
    else:args=(ptr*1)(ctypes.cast(ctypes.byref(addr),ptr))
    torch.cuda.synchronize();start=time.monotonic()
    check(lib.cuLaunchKernel(fn,1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'launch')
    check(lib.cuCtxSynchronize(),'sync')
    values=out.cpu().tolist();result[name]=dict(values=values,seconds=time.monotonic()-start)
    print(name,values,flush=True)
    (P/'device-smoke.json').write_text(json.dumps(result,indent=2))
assert result['device_rule_probe']['values']==[3,2],result
result['completed']=True
result['torch_allocated']=torch.cuda.memory_allocated();result['torch_reserved']=torch.cuda.memory_reserved()
(P/'device-smoke.json').write_text(json.dumps(result,indent=2))
