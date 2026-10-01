"""CUDA arithmetic shared by policy inference and learning."""
import torch


def bf16_autocast(*, enabled=True, cache_enabled=True):
    """Use BF16 operands without truncating GEMM partial sums back to BF16.

    PyTorch's reduction setting is process-wide. Keep it disabled after the
    context exits so backward and other inference threads use the same policy;
    temporarily restoring it would make arithmetic depend on thread timing.
    This improves accuracy but does not promise cross-shape bitwise equality.
    """
    if enabled:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled,
                          cache_enabled=cache_enabled)
