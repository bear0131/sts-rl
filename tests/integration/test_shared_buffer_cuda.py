"""GPU-resident datasets retain SB3 ordering and independent sample storage."""
import numpy as np
import pytest
import torch

from tests.unit.test_shared_buffer import assert_owned_storage_returns_and_env_major_shuffle


@pytest.mark.training
@pytest.mark.parametrize('allocation_failure', [False, True])
def test_device_gather_and_partial_allocation_fallback(monkeypatch, allocation_failure):
    assert torch.cuda.is_available(), 'explicit CUDA buffer qualification required'
    original = torch.as_tensor
    copies, failed = 0, False

    def transfer(data, *args, **kwargs):
        nonlocal copies, failed
        device = kwargs.get('device')
        if isinstance(data, np.ndarray) and device is not None and torch.device(device).type == 'cuda':
            copies += 1
            if allocation_failure and copies == 3:
                failed = True
                raise torch.OutOfMemoryError('injected partial device dataset allocation')
        return original(data, *args, **kwargs)

    monkeypatch.setattr(torch, 'as_tensor', transfer)
    assert_owned_storage_returns_and_env_major_shuffle(3, 4, True, device='cuda')
    assert copies > 3 and failed == allocation_failure
    torch.cuda.empty_cache()
