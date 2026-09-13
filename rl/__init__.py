"""Learning runtime on env/. New imports use the owning subpackage.

Legacy checkpoint library module names resolve lazily through _compat/.
That directory contains aliases only; canonical modules never import it.
"""
from pathlib import Path as _Path

# A deployment can still contain retired flat files. Historical checkpoint
# names must resolve to canonical aliases before any stale implementation.
__path__.insert(0, str(_Path(__file__).with_name("_compat")))
