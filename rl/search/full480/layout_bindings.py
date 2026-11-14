"""Keep every original layout implementation bound to this package."""
from pathlib import Path
import hashlib
from . import flex_stream, flex_adapter, flex_runtime, flex_ragged
ACTUAL_LAYOUT_MODULES={}
for module in (flex_stream,flex_adapter,flex_runtime,flex_ragged):
    path=Path(module.__file__).resolve()
    assert path.parent==Path(__file__).resolve().parent
    ACTUAL_LAYOUT_MODULES[module.__name__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
