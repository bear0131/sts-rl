"""FP16 inference layout keeps padding semantics and the BF16 root path."""
import pytest
import torch

from rl.policies.entity_combat import EntityEncoderLayer

pytestmark = [pytest.mark.native, pytest.mark.timeout(240),
              pytest.mark.usefixtures('isolated_torch_compile_cache'),
              pytest.mark.skipif(not torch.cuda.is_available(), reason='requires explicit CUDA validation')]


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('width,heads', [(32, 4), (256, 8)])
def test_fp16_attention_dynamic_shapes_masks_and_mutable_weights(compiled, width, heads):
    torch.manual_seed(1729)
    kwargs = dict(d_model=width, nhead=heads, dim_feedforward=4*width, dropout=0.,
                  activation='gelu', batch_first=True, norm_first=True, device='cuda')
    reference = torch.nn.TransformerEncoderLayer(**kwargs).eval()
    actual = EntityEncoderLayer(**kwargs).eval()
    actual.load_state_dict(reference.state_dict())
    run = torch.compile(actual, fullgraph=True, dynamic=True) if compiled else actual
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        with torch.inference_mode():
            for count, length in ((2, 37), (4, 54), (3, 49)):
                data = torch.randn(count, length, width, device='cuda')
                padding = torch.rand(count, length, device='cuda') > .5
                padding[:, 0] = False
                expected = reference(data, src_key_padding_mask=padding)
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    observed = run(data, src_key_padding_mask=padding)
                    poisoned = data.masked_fill(padding[..., None], 20.)
                    masked = run(poisoned, src_key_padding_mask=padding)
                # The FP32 layer is the oracle, not a second version of the new layout.
                torch.testing.assert_close(observed, expected, atol=.003, rtol=.005)
                # Masked keys cannot influence the visible tokens, even with large inputs.
                torch.testing.assert_close(observed[~padding], masked[~padding], atol=0, rtol=0)
            actual.self_attn.out_proj.bias.add_(.125)
            reference.self_attn.out_proj.bias.add_(.125)
            expected = reference(data, src_key_padding_mask=padding)
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                changed = run(data, src_key_padding_mask=padding)
            assert not torch.equal(changed, observed)
            torch.testing.assert_close(changed, expected, atol=.003, rtol=.005)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_bf16_entity_root_retains_original_attention_outputs():
    torch.manual_seed(23)
    kwargs = dict(d_model=32, nhead=4, dim_feedforward=128, dropout=0.,
                  activation='gelu', batch_first=True, norm_first=True, device='cuda')
    reference = torch.nn.TransformerEncoderLayer(**kwargs).eval()
    actual = EntityEncoderLayer(**kwargs).eval()
    actual.load_state_dict(reference.state_dict())
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for count, length in ((2, 37), (4, 54), (3, 49)):
            data = torch.randn(count, length, 32, device='cuda')
            padding = torch.rand(count, length, device='cuda') > .5
            padding[:, 0] = False
            torch.testing.assert_close(actual(data, src_key_padding_mask=padding),
                                       reference(data, src_key_padding_mask=padding), atol=0, rtol=0)
