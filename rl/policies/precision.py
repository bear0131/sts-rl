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


def fp16_autocast(*, cache_enabled=True):
    """Inference FP16 with full-precision GEMM partial sums.

    Like the BF16 policy, the reduction setting stays process-wide. This does
    not select an optimizer precision or require a training gradient scaler.
    """
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    return torch.autocast('cuda', dtype=torch.float16, cache_enabled=cache_enabled)
