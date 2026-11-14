"""Original CUDA driver setup, explicitly owned by the sampling worker."""
from pathlib import Path
import ctypes, hashlib, importlib.util, os, sys
import torch

def artifact_directory():
    return Path(os.environ.get('STS_FULL480_ARTIFACT_DIR', Path(__file__).resolve().parent))

P=artifact_directory()
ptr=ctypes.c_void_p
u32=ctypes.c_uint
module=None
_context_anchor=None

class _Driver:
    handle=None
    def __getattr__(self,name):
        if self.handle is None:
            self.handle=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
        return getattr(self.handle,name)

lib=_Driver()

def check(code,where):
    if code:raise RuntimeError((where,code))

def initialize():
    global _context_anchor
    if _context_anchor is not None:return
    torch.cuda.init()
    _context_anchor=torch.zeros(1,device='cuda')
    lib.cuModuleLoad.argtypes=[ctypes.POINTER(ptr),ctypes.c_char_p]
    lib.cuModuleGetFunction.argtypes=[ctypes.POINTER(ptr),ptr,ctypes.c_char_p]
    lib.cuLaunchKernel.argtypes=[ptr,u32,u32,u32,u32,u32,u32,u32,ptr,ctypes.POINTER(ptr),ptr]
    lib.cuModuleUnload.argtypes=[ptr]
    lib.cuCtxSynchronize.argtypes=[]
    lib.cuCtxSetLimit.argtypes=[ctypes.c_int,ctypes.c_size_t]
    check(lib.cuCtxSetLimit(0,65536),'stack limit')
    check(lib.cuCtxSetLimit(2,512*1024*1024),'device heap limit')

def cpu_bridge():
    """Use the exact original state ABI, never a same-named ambient binary."""
    path=P/'devicerootsearch141/gpu_search_bridge.cpython-311-x86_64-linux-gnu.so'
    digest='36da97eb79bff5c2e1fe42d8bc65e01fc87971c8dcf487cdd609421977396c76'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('full480 original state bridge identity mismatch')
    name='gpu_search_bridge'
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve()!=path.resolve():
            raise RuntimeError('another state bridge is already loaded')
        return existing
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded
    spec.loader.exec_module(loaded)
    return loaded

def load_net():
    raise ValueError('full480 requires the explicitly supplied frozen checkpoint consumer')

def Fused(net,precision):
    raise ValueError('full480 requires explicit consumer weight tables')
