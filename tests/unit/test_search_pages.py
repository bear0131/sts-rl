import numpy as np
import pytest

from env.search.root_ledger import RootLedger
from env.search.state_pages import DevicePages


def test_root_generation_and_pages_reject_stale_owners_and_keep_live_siblings():
    class Block:
        def __init__(self, address): self.address=address
        def data_ptr(self): return self.address
    allocations=[]
    def allocate(count):
        address=0x100000+len(allocations)*0x100000
        allocations.append((address,count))
        return Block(address)
    roots=RootLedger(2)
    pages=DevicePages(2,64,128,allocate,page_size=16,growth_pages=1)
    a,b=roots.claim(['first','sibling'])
    leases,pointers=pages.acquire([((a.slot,a.generation),17),((b.slot,b.generation),64)])
    sibling=pointers[(b.slot,b.generation)].copy()
    pages.release([leases[(a.slot,a.generation)]])
    roots.release(a)
    c=roots.claim(['replacement'])[0]
    assert c.slot==a.slot and c.generation>a.generation
    new,ptr=pages.acquire([((c.slot,c.generation),32)])
    assert not np.intersect1d(sibling,ptr[(c.slot,c.generation)]).size
    with pytest.raises(ValueError,match='stale'): roots.release(a)
    with pytest.raises(ValueError,match='stale'): pages.release([leases[(a.slot,a.generation)]])
    with pytest.raises(RuntimeError,match='live'): pages.close()
    pages.release([leases[(b.slot,b.generation)],new[(c.slot,c.generation)]])
    roots.release(b); roots.release(c); pages.close()
    assert pages.statistics()['owners']==0 and len(roots)==0


def test_pending_first_choice_without_portable_origin_uses_native_fallback(monkeypatch):
    from types import SimpleNamespace
    import env.search.resident_transport as transport
    closed=[]
    branch=SimpleNamespace(restore_training_state=lambda *a,**k: None,
                           close=lambda: closed.append(True))
    env=SimpleNamespace(fork_run=lambda:branch,export_training_state=lambda:{'journal':[]})
    exporter=object.__new__(transport.ResidentRootExporter)
    exporter.envs=[env]; exporter.bridge=object(); exporter.cached=[None]
    monkeypatch.setattr(transport,'root_payload',lambda *a:dict(snapshot={'portable':False}))
    result=exporter.export([0])
    assert result[0]['native_fallback']=='pending-without-portable-origin'
    assert result[0]['snapshot']['portable'] is False
    assert result[0]['replay']==[] and closed==[True] and exporter.cached==[None]
