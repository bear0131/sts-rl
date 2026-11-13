import gc

import numpy as np
import pytest
import torch

from env.search.paired_rng import paired_uniforms

pytestmark = [pytest.mark.cuda, pytest.mark.timeout(90)]


@pytest.mark.parametrize('depth', [1,16,200,257])
def test_paired_pcg64_all_draws_bitwise_match_numpy_at_sparse_slots(depth):
    boundaries=np.array([0,1,2**32-1,2**32,2**63-1,2**63,2**64-1,
        0x504F4C494359,0x504F4C494359^1],np.uint64)
    seeds=np.r_[boundaries,np.random.default_rng(20261111).integers(0,2**64-1,513,dtype=np.uint64),boundaries]
    slots=np.arange(len(seeds))*2+1
    out=torch.full((len(seeds)*2+3,depth),-1.,device='cuda',dtype=torch.float64)
    paired_uniforms(torch.as_tensor(seeds.view(np.int64),device='cuda'),
                    torch.as_tensor(slots,device='cuda'),out)
    expected=np.stack([np.random.default_rng(int(s)^0x504F4C494359).random(depth) for s in seeds])
    actual=out.cpu().numpy()
    np.testing.assert_array_equal(actual[slots].view(np.uint64),expected.view(np.uint64))
    assert (np.delete(actual,slots,axis=0)==-1.).all()
    # Admissions vary at every real rollout step. Dynamic row counts must use
    # the same stream and may not touch slots outside the current admission.
    for count in (1,17,32,128):
        out.fill_(-1.)
        paired_uniforms(torch.as_tensor(seeds[:count].view(np.int64),device='cuda'),
                        torch.as_tensor(slots[:count],device='cuda'),out)
        actual=out.cpu().numpy()
        np.testing.assert_array_equal(actual[slots[:count]].view(np.uint64),expected[:count].view(np.uint64))
        assert (np.delete(actual,slots[:count],axis=0)==-1.).all()


def test_paired_pcg64_graph_replay_uses_new_seeds_after_allocator_release():
    seeds=torch.tensor([7,19,42],device='cuda',dtype=torch.int64)
    slots=torch.tensor([2,0,4],device='cuda',dtype=torch.int64)
    out=torch.full((6,200),-1.,device='cuda',dtype=torch.float64)
    paired_uniforms(seeds,slots,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):paired_uniforms(seeds,slots,out)
        seeds.copy_(torch.tensor([29,65536,20261111],device='cuda'))
        gc.collect();torch.cuda.empty_cache();graph.replay()
        expected=np.stack([np.random.default_rng(s^0x504F4C494359).random(200) for s in (29,65536,20261111)])
        np.testing.assert_array_equal(out[slots].cpu().numpy().view(np.uint64),expected.view(np.uint64))
    finally:graph.reset()
