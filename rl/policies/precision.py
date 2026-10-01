"""CUDA arithmetic shared by policy inference and learning."""
import torch


def bf16_autocast(*, enabled=True, cache_enabled=True):
    """Keep GEMM partial sums at full precision, including during backward.

    This setting is process-wide and intentionally remains disabled after the
    context exits. Restoring it would make concurrent inference/backward depend
    on thread timing. This is not a cross-shape bitwise equality guarantee.
    """
    if enabled:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled,
                          cache_enabled=cache_enabled)
