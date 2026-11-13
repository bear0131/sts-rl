"""Device scheduling adapted from the accepted step/frame search pipeline.

Stable prefix compaction preserves logical lane order. Classification and
capacity checks are read-only; commit is disabled on every overflow or context
miss, so graph capture and retry never advance a battle or a random cursor.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def classify(OUTCOME, MASK, CATS, MISC, EXPECTED, DECK, OWNERS, FRAME_IDS, FRAME_COUNT, DEPTH,
             KIND, FORCED, NEED, CHANGED, STATUS, OFFSET, N: tl.constexpr,
             MAX_DEPTH: tl.constexpr, MISC_SCALE: tl.constexpr, B: tl.constexpr):
    lane = tl.program_id(0)*B + tl.arange(0, B)
    active = (lane < N) & (lane < tl.load(FRAME_COUNT))
    i = tl.load(FRAME_IDS+lane, active, 0)
    offset = tl.load(OFFSET)
    a, d = tl.arange(0, 256), tl.arange(0, 128)
    outcome = tl.load(OUTCOME+offset+i, active, 0)
    mask = tl.load(MASK+(offset+i[:, None])*198+a[None, :], active[:, None] & (a[None, :] < 198), 0)
    count = tl.sum((mask != 0).to(tl.int32), 1)
    depth = tl.load(DEPTH+i, active, 0)
    cutoff = active & (outcome == 0) & (depth >= MAX_DEPTH)
    combat = active & (outcome == 0) & ~cutoff
    kind = tl.where(~active, 0, tl.where(outcome == 2, 2,
        tl.where((outcome == 1) | (outcome == 3), 1,
        tl.where(cutoff, 4, tl.where(count == 1, 5, 6)))))
    need = (kind == 4) | (kind == 6)
    col = tl.load(CATS+(offset+i[:, None])*226+127+d[None, :], need[:, None] & (d[None, :] < 96), 0)
    pick = tl.load(MASK+(offset+i[:, None])*198+91+d[None, :], need[:, None] & (d[None, :] < 96), 0)
    width = tl.max(tl.where((col > 0) | (pick != 0), d[None, :]+1, 1), 1)
    owner = tl.load(OWNERS+i, active, 0)
    misc = tl.load(MISC+(offset+i[:, None])*96+d[None, :], need[:, None] & (d[None, :] < 96), -1)
    expected = tl.load(EXPECTED+i[:, None]*96+d[None, :], need[:, None] & (d[None, :] < 96), 0.)
    card = tl.load(DECK+owner[:, None]*96+d[None, :], need[:, None] & (d[None, :] < 96), 0)
    normalized = libdevice.div_rn(misc.to(tl.float64),
        tl.full((), MISC_SCALE, tl.float64)).to(tl.float32)
    changed = tl.sum(((misc >= 0) & (card != 0) & (normalized != expected)).to(tl.int32), 1) > 0
    tl.store(CHANGED+i, changed.to(tl.int32), active)
    tl.store(KIND+i, kind, active)
    tl.store(FORCED+i, tl.min(tl.where(mask != 0, a[None, :], 198), 1), active)
    tl.store(NEED+lane, need.to(tl.int32), lane < N)
    tl.atomic_max(STATUS+1, tl.max(tl.where(need, width, 1), 0))
    tl.atomic_or(STATUS+2, tl.max(changed.to(tl.int32), 0))
    tl.atomic_or(STATUS+3, tl.max((combat & (count < 1)).to(tl.int32), 0))
    tl.atomic_add(STATUS+4, tl.sum(combat.to(tl.int32), 0))


@triton.jit
def compact_local(NEED, PREFIX, TOTALS, N: tl.constexpr, B: tl.constexpr):
    b = tl.program_id(0)
    i = b*B + tl.arange(0, B)
    x = tl.load(NEED+i, i < N, 0)
    y = tl.cumsum(x, 0)
    tl.store(PREFIX+i, y, i < N)
    tl.store(TOTALS+b, tl.sum(x, 0))


@triton.jit
def compact_scatter(NEED, PREFIX, TOTALS, IDS, STATUS, N: tl.constexpr,
                    B: tl.constexpr, NB: tl.constexpr, CB: tl.constexpr):
    b = tl.program_id(0)
    i, q = b*B + tl.arange(0, B), tl.arange(0, CB)
    totals = tl.load(TOTALS+q, q < NB, 0)
    offset = tl.sum(tl.where(q < b, totals, 0), 0)
    valid = (i < N) & (tl.load(NEED+i, i < N, 0) != 0)
    dst = offset + tl.load(PREFIX+i, i < N, 0) - 1
    tl.store(IDS+dst, i, valid)
    if b == 0:
        tl.store(STATUS, tl.sum(totals, 0))


@triton.jit
def compact_live_local(LIVE, PREFIX, TOTALS, CONTROL, N: tl.constexpr, B: tl.constexpr):
    if tl.load(CONTROL+1):
        compact_local(LIVE, PREFIX, TOTALS, N, B)


@triton.jit
def compact_live_scatter(LIVE, PREFIX, TOTALS, IDS, COUNT, CONTROL, N: tl.constexpr,
                         B: tl.constexpr, NB: tl.constexpr, CB: tl.constexpr):
    if tl.load(CONTROL+1):
        compact_scatter(LIVE, PREFIX, TOTALS, IDS, COUNT, N, B, NB, CB)


@triton.jit
def frame_prepare(IDS, COUNT, FRAME_IDS, FRAME_COUNT, CONTROL, CAP: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    cursor=tl.load(CONTROL);count=tl.minimum(tl.load(COUNT)-cursor,CAP)
    value=tl.load(IDS+cursor+i,(i<CAP)&(i<count),0)
    tl.store(FRAME_IDS+i,value,i<CAP)
    if tl.program_id(0)==0:
        tl.store(FRAME_COUNT,count);tl.store(CONTROL+1,0)


@triton.jit
def frame_finish(COUNT, FRAME_COUNT, CONTROL, STATUS, ENABLED, F: tl.constexpr, WIDTH: tl.constexpr):
    if (tl.load(ENABLED)!=0)&(tl.load(STATUS)<=F)&(tl.load(STATUS+1)<=WIDTH)&(tl.load(STATUS+2)==0)&(tl.load(STATUS+3)==0)&(tl.load(STATUS+7)==0):
        cursor=tl.load(CONTROL)+tl.load(FRAME_COUNT)
        complete=cursor>=tl.load(COUNT)
        tl.store(CONTROL,tl.where(complete,0,cursor))
        tl.store(CONTROL+1,complete.to(tl.int32))
    tl.store(STATUS+8, tl.load(CONTROL))
    tl.store(STATUS+9, tl.load(COUNT))


@triton.jit
def remaining_live(LIVE, STATUS, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    live = tl.load(LIVE+i, i < N, 0)
    tl.atomic_add(STATUS+4, tl.sum(live, 0))


@triton.jit
def prepare(IDS, FRAME_IDS, ROWS, RANDOM, CURSOR, UNIFORMS, STATUS, F: tl.constexpr,
            MAX_DEPTH: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    count = tl.load(STATUS)
    frame = tl.load(IDS+tl.where(i < count, i, 0), (i < F) & (count > 0), 0)
    row = tl.load(FRAME_IDS+frame, (i < F) & (count > 0), 0)
    cursor = tl.minimum(tl.load(CURSOR+row, i < F, 0), MAX_DEPTH-1)
    u = tl.load(RANDOM+row*MAX_DEPTH+cursor, i < F, .5)
    tl.store(ROWS+i, row.to(tl.int64), i < F)
    tl.store(UNIFORMS+i, u, i < F)


@triton.jit
def sample_fp64(LOGITS, MASK, ROWS, OFFSET, UNIFORMS, SAMPLED, STATUS,
                A: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0);j=tl.arange(0,B)
    if (i<tl.load(STATUS)) & (tl.load(STATUS+7)==0):
        physical=tl.load(ROWS+i)+tl.load(OFFSET)
        legal=tl.load(MASK+physical*A+j,j<A,0)!=0
        x=tl.load(LOGITS+i*A+j,j<A,0.).to(tl.float64)
        x=tl.where(legal,x,-float('inf'))
        weight=libdevice.exp(x-tl.max(x,0))
        probability=weight/tl.sum(weight,0)
        cdf=tl.cumsum(probability,0)
        last=tl.sum(tl.where(j==A-1,cdf,0.),0)
        uniform=tl.load(UNIFORMS+i)
        selected=tl.sum(((j<A)&(cdf/last<=uniform)).to(tl.int32),0)
        tl.store(SAMPLED+i,selected)
    else:
        tl.store(SAMPLED+i,0)


@triton.jit
def validate(VALUES, STATUS, F: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    value = tl.load(VALUES+i, (i < F) & (i < tl.load(STATUS)), 0.)
    bad = (value != value) | (tl.abs(value) == float('inf'))
    tl.atomic_or(STATUS+3, tl.max(bad.to(tl.int32), 0)*2)


@triton.jit
def commit(KIND, FORCED, LIVE, DEPTH, CURSOR, ANCHORS, FINAL, SELECTED,
           STATUS, ENABLED, COUNTERS, OFFSET, FRAME_IDS, FRAME_COUNT, N: tl.constexpr,
           F: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr):
    if (tl.load(ENABLED) != 0) & (tl.load(STATUS) <= F) & (tl.load(STATUS+1) <= WIDTH) & (tl.load(STATUS+2) == 0) & (tl.load(STATUS+3) == 0) & (tl.load(STATUS+7) == 0):
        lane = tl.program_id(0)*B + tl.arange(0, B)
        valid = (lane < N) & (lane < tl.load(FRAME_COUNT))
        i = tl.load(FRAME_IDS+lane, valid, 0)
        offset = tl.load(OFFSET)
        kind = tl.load(KIND+i, valid, 0)
        live = (kind == 5) | (kind == 6)
        # Rows waiting outside this physical frame retain their state. A zero
        # kind is either already finished or has not been observed this tick.
        tl.store(LIVE+i, live.to(tl.int32), valid)
        depth = tl.load(DEPTH+i, valid, 0)
        tl.store(DEPTH+i, depth + live.to(tl.int32), valid)
        cursor = tl.load(CURSOR+i, valid, 0)
        tl.store(CURSOR+i, cursor+(kind == 6).to(tl.int32), valid)
        tl.store(ANCHORS+i, 1, valid & (kind == 1))
        action = tl.load(FORCED+i, valid, 0)
        tl.store(SELECTED+offset+i, action, valid & (kind == 5))
        for slot in tl.static_range(5):
            target = 2 if slot == 0 else 1 if slot == 1 else 4 if slot == 2 else 5 if slot == 3 else 6
            tl.atomic_add(COUNTERS+slot, tl.sum((kind == target).to(tl.int32), 0))


@triton.jit
def commit_scores(ROWS, SAMPLED, VALUES, KIND, FINAL, SELECTED, STATUS,
                  ENABLED, OFFSET, F: tl.constexpr,
                  WIDTH: tl.constexpr, B: tl.constexpr):
    if (tl.load(ENABLED) != 0) & (tl.load(STATUS) <= F) & (tl.load(STATUS+1) <= WIDTH) & (tl.load(STATUS+2) == 0) & (tl.load(STATUS+3) == 0) & (tl.load(STATUS+7) == 0):
        i = tl.program_id(0)*B + tl.arange(0, B)
        offset = tl.load(OFFSET)
        valid = (i < F) & (i < tl.load(STATUS))
        row = tl.load(ROWS+i, valid, 0)
        kind = tl.load(KIND+row, valid, 0)
        action = tl.load(SAMPLED+i, valid, 0)
        value = tl.load(VALUES+i, valid, 0.)
        tl.store(SELECTED+offset+row, action, valid & (kind == 6))
        tl.store(FINAL+row, value.to(tl.float64), valid & (kind == 4))
