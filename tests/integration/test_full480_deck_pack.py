"""Deck token packing across queue boundaries and CUDA graph replays."""
import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.native, pytest.mark.training,
              pytest.mark.timeout(120)]


def test_deck_pack_preserves_tokens_queries_and_padding():
    from rl.search.full480.combo_runtime import BattleArena
    from rl.search.full480.frame_modules import FrameModules
    from rl.search.full480.run_device import artifact_directory

    assert torch.cuda.is_available(), 'requires CUDA and explicit full480 artifacts'
    lengths = [32, 33, 64, 65, 121]
    rows, capacity, blocks = len(lengths), 512, 37
    shapes = dict(worlds=(rows,), ids=(rows*128,), zones=(rows*128,),
        monsters=(rows*128,), numbers=(rows*128, 64), meta=(rows, 4),
        chunks=(rows, 16), inv_ids=(1,), inv_zones=(1,), inv_monsters=(1,),
        inv_numbers=(1, 16), inv_counts=(1,), inv_slots=(1,),
        source_rows=(rows*2, 10), target_slots=(rows*2,),
        canonical_sources=(rows*2, 10), canonical_targets=(rows*2,),
        query_map=(rows, 161), features=(capacity, 128), symbols=(capacity,),
        slots=(capacity,), present=(capacity,), queries=(capacity,),
        blocks1=(blocks, 2), blocks2=(blocks, 2), owners=(rows*2,),
        starts=(rows, 4), lens=(rows, 4), scores=(rows*2,), allocator=(5,),
        dispatch=(1,), totals=(5,), queues=(2*rows,), queue_counts=(2,))
    floats = {'numbers', 'inv_numbers', 'features', 'scores'}
    longs = {'symbols', 'slots', 'queries', 'starts', 'lens'}
    data = {key: torch.zeros(shape, device='cuda', dtype=(
        torch.float32 if key in floats else torch.int64 if key in longs else
        torch.bool if key == 'present' else torch.int32)) for key, shape in shapes.items()}
    physical_rows, logical_rows, query_rows, present_rows = [], [], [], []
    for row, count in enumerate(lengths):
        logical = torch.cat((torch.arange(count-1, device='cuda'),
                             torch.tensor([120], device='cuda')))
        base = (rows-1-row)*128
        data['chunks'][row, :2] = torch.tensor([base+64, base], device='cuda')
        index = torch.arange(count, device='cuda')
        physical = data['chunks'][row, index//64].long() + index%64
        query = logical.remainder(3) == 0
        present = (logical.remainder(4) != 0) | ~query | (logical == 0)
        data['ids'][physical] = logical.int()
        data['zones'][physical] = present.int()
        data['monsters'][physical] = query.int()
        data['numbers'][physical] = (torch.arange(count*64, device='cuda').reshape(count, 64)-1000)/32
        data['numbers'][physical[0], 0] = -0.0
        data['meta'][row] = torch.tensor([2, count, row*2, 1], device='cuda')
        physical_rows.append(physical); logical_rows.append(logical)
        query_rows.append(query); present_rows.append(present)
    data['dispatch'].fill_(rows)
    arena = BattleArena(1, deck=True)
    modules = graph = None
    try:
        modules = FrameModules(arena.modules.identity['rules']['abi_layout'], deck=True,
            schema=(artifact_directory()/'deck/deck_schema.h').read_text())

        def pack():
            data['allocator'].zero_(); data['totals'].zero_(); data['queue_counts'].zero_()
            for name, cap in [('pack_short', 32), ('pack_medium', 64), ('pack_long', 161)]:
                modules.launch(name, list(data.values()),
                    [1, 1, 161, rows, capacity, capacity, blocks, blocks],
                    rows, threads=256, shared=cap*128*4)

        pack()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack()
        for iteration in range(3):
            if iteration:
                data['numbers'].add_(.5)
                temporary = torch.empty(64*1024*1024, device='cuda', dtype=torch.uint8)
                temporary.fill_(17)
                del temporary
                torch.cuda.empty_cache()
            data['features'].fill_(float('nan'))
            graph.replay()
            assert data['allocator'][2].item() == 0
            assert data['queue_counts'].tolist() == [4, 2]
            assert data['totals'][0].item() == sum(lengths)
            for row, count in enumerate(lengths):
                start, query_start = map(int, data['starts'][row, :2].tolist())
                logical, query = logical_rows[row], query_rows[row]
                assert data['lens'][row, :2].tolist() == [count, int(query.sum())]
                actual = data['features'][start:start+count]
                expected = data['numbers'][physical_rows[row]]
                torch.testing.assert_close(actual[:, :64].view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
                assert (actual[:, 64:].view(torch.int32) == 0).all()
                torch.testing.assert_close(data['present'][start:start+count], present_rows[row])
                expected_queries = torch.arange(count, device='cuda')[query] + start
                torch.testing.assert_close(data['queries'][query_start:query_start+len(expected_queries)], expected_queries)
                mapping = torch.full((161,), -1, device='cuda', dtype=torch.int32)
                mapping[logical[query]] = torch.arange(int(query.sum()), device='cuda', dtype=torch.int32)
                torch.testing.assert_close(data['query_map'][row], mapping)
    finally:
        if graph is not None:
            torch.cuda.synchronize(); graph.reset()
        if modules is not None:
            modules.close()
        arena.close()
