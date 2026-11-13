"""Device generation of the existing NumPy SeedSequence/PCG64 double stream.

Only scalar uint64 entropy, pool_size=4, empty spawn_key and float64 uniform
output are supported. This is the paired search's existing RNG, not a new one.
Reference: numpy v2.2.0 random/bit_generator.pyx and src/pcg64/pcg64.h.

Derived from SeedSequence and PCG's MIT-licensed algorithms:
Copyright (c) 2015 Melissa E. O'Neill
Copyright (c) 2019 NumPy Developers
Copyright (c) 2014 Melissa O'Neill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hash(x, h, multiplier: tl.constexpr):
    x = x ^ h
    h = (h * multiplier).to(tl.uint32)
    x = (x * h).to(tl.uint32)
    return x ^ (x >> 16), h


@triton.jit
def _mix(x, y):
    z = (x * 0xca01f9dd - y * 0x4973f715).to(tl.uint32)
    return z ^ (z >> 16)


@triton.jit
def _advance(hi, lo, inc_hi, inc_lo):
    low = lo * 4865540595714422341
    high = tl.umulhi(lo, tl.full((), 4865540595714422341, tl.uint64))
    high += hi * 4865540595714422341 + lo * 2549297995355413924
    summed = low + inc_lo
    return high + inc_hi + (summed < low).to(tl.uint64), summed


@triton.jit
def _uniforms(SEEDS, SLOTS, OUT, N, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)*B + tl.arange(0, B)
    seed = tl.load(SEEDS+row, row < N, 0).to(tl.uint64) ^ 0x504F4C494359
    h = tl.full((), 0x43b0d7e5, tl.uint32)
    p0, h = _hash(seed.to(tl.uint32), h, 0x931e8875)
    p1, h = _hash((seed >> 32).to(tl.uint32), h, 0x931e8875)
    p2, h = _hash(tl.full((), 0, tl.uint32), h, 0x931e8875)
    p3, h = _hash(tl.full((), 0, tl.uint32), h, 0x931e8875)
    # Scalar entropy has at most two words; padding with zero uses exactly
    # the same hashmix calls as NumPy's absent-entropy branch.
    for src in tl.static_range(4):
        if src == 0: value = p0
        elif src == 1: value = p1
        elif src == 2: value = p2
        else: value = p3
        for dst in tl.static_range(4):
            if src != dst:
                hashed, h = _hash(value, h, 0x931e8875)
                if dst == 0: p0 = _mix(p0, hashed)
                elif dst == 1: p1 = _mix(p1, hashed)
                elif dst == 2: p2 = _mix(p2, hashed)
                else: p3 = _mix(p3, hashed)
    h = tl.full((), 0x8b51f9dd, tl.uint32)
    a0,h=_hash(p0,h,0x58f38ded);a1,h=_hash(p1,h,0x58f38ded)
    a2,h=_hash(p2,h,0x58f38ded);a3,h=_hash(p3,h,0x58f38ded)
    a4,h=_hash(p0,h,0x58f38ded);a5,h=_hash(p1,h,0x58f38ded)
    a6,h=_hash(p2,h,0x58f38ded);a7,h=_hash(p3,h,0x58f38ded)
    seed_hi=a0.to(tl.uint64)|(a1.to(tl.uint64)<<32)
    seed_lo=a2.to(tl.uint64)|(a3.to(tl.uint64)<<32)
    seq_hi=a4.to(tl.uint64)|(a5.to(tl.uint64)<<32)
    seq_lo=a6.to(tl.uint64)|(a7.to(tl.uint64)<<32)
    inc_hi=(seq_hi<<1)|(seq_lo>>63);inc_lo=(seq_lo<<1)|1
    lo=inc_lo+seed_lo;hi=inc_hi+seed_hi+(lo<inc_lo).to(tl.uint64)
    hi,lo=_advance(hi,lo,inc_hi,inc_lo)
    slot=tl.load(SLOTS+row,row<N,0)
    for depth in range(D):
        hi,lo=_advance(hi,lo,inc_hi,inc_lo)
        raw64=hi^lo;rotation=hi>>58
        raw64=(raw64>>rotation)|(raw64<<((64-rotation)&63))
        uniform=(raw64>>11).to(tl.float64)*1.1102230246251565e-16
        tl.store(OUT+slot*D+depth,uniform,row<N)


def paired_uniforms(seeds, slots, output):
    """Fill only newly admitted jobs, retaining every original FP64 draw."""
    if (seeds.device != output.device or slots.device != output.device
            or not output.is_cuda or output.dtype != torch.float64
            or seeds.dtype != torch.int64 or slots.dtype != torch.int64
            or len(seeds) != len(slots) or output.ndim != 2
            or not output.is_contiguous()):
        raise ValueError('paired uniforms require CUDA int64 seeds/slots and contiguous FP64 rows')
    if len(seeds):
        _uniforms[(triton.cdiv(len(seeds),128),)](seeds,slots,output,len(seeds),output.shape[1],128,
            num_warps=4,enable_fp_fusion=False)
