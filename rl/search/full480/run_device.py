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
    """Original bridge with the explicit pending-continuation state schema."""
    path=P/'cpu-build/gpu_search_bridge.cpython-311-x86_64-linux-gnu.so'
    if not path.is_file():raise RuntimeError('build the full480 state bridge before collection')
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
    if loaded.state_schema!=2:raise RuntimeError('full480 pending-continuation state schema required')
    return loaded

def cpu_fallback():
    """Load the explicitly built adaptation of the original WaveDiag source."""
    import slaythespire as sts
    path=P/'cpu-build/_full480_cpu.cpython-311-x86_64-linux-gnu.so'
    if not path.is_file():raise RuntimeError('build the STS_BUILD_FULL480 target before collection')
    name='_full480_cpu'
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path.resolve():raise RuntimeError('another full480 CPU build is loaded')
        return module
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if module.contract!='paired-pcg64-terminal-hp-boss-win-v1':raise RuntimeError('wrong full480 CPU contract')
    if bool(module.rule_trace)!=hasattr(sts.BattleContext,'set_player_strength'):raise RuntimeError('CPU simulator trace ABI differs')
    return module

def load_net():
    raise ValueError('full480 requires the explicitly supplied frozen checkpoint consumer')

def Fused(net,precision):
    raise ValueError('full480 requires explicit consumer weight tables')
