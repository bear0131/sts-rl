"""Learning runtime on env/. New imports use the owning subpackage.

Legacy checkpoint and CLI module names resolve lazily through _compat/.
That directory contains aliases only; canonical modules never import it.
"""
from pathlib import Path as _Path

__path__.append(str(_Path(__file__).with_name("_compat")))
