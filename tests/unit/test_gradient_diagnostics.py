"""Transport diagnostics retain every nonfinite gradient, including after reduction."""
import pytest
import torch

from rl.training.distributed_sb3 import count_nonfinite_gradients


def test_cpu_gradient_scan_handles_nan_infinity_subnormal_and_empty():
    values = torch.tensor([0., -0., 1., -1., torch.finfo(torch.float32).max,
                           torch.finfo(torch.float32).tiny / 2,
                           float('nan'), float('inf'), -float('inf')])
    assert count_nonfinite_gradients(values) == 3
    assert count_nonfinite_gradients(values[:6]) == 0
    assert count_nonfinite_gradients(values[:0]) == 0
    assert count_nonfinite_gradients(values[::2]) == 2
    torch.testing.assert_close(values[-3:], torch.tensor([float('nan'), float('inf'),
                                                       -float('inf')]), equal_nan=True)


def test_cpu_gradient_scan_rejects_a_different_transport_dtype():
    with pytest.raises(TypeError, match='CPU float32'):
        count_nonfinite_gradients(torch.zeros(3, dtype=torch.float64))
