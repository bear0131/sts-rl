from pathlib import Path
import importlib.util,sys,hashlib
P=Path(__file__).resolve().parent
ACTUAL_LAYOUT_MODULES={}
for name in ['flex_stream','flex_adapter','flex_runtime','flex_ragged']:
 path=P/(name+'.py')
 if name in sys.modules:
  assert Path(sys.modules[name].__file__).resolve()==path,(name,sys.modules[name].__file__,str(path))
 else:
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 ACTUAL_LAYOUT_MODULES[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
