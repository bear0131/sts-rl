"""One owned, bounded fallback input arena and complete encode/score/sample graph."""
from pathlib import Path
import sys, collections
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'devicefallback141'))
from packed_fallback import torch, np, triton, tl, Fused, hash_table
sys.path.insert(0, str(P.parent / 'devicecombo141'))
from .combo_model import GraphScorer


@triton.jit
def clear_features(features, header, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(header) * tl.load(header + 1) * 128
    tl.store(features + i, 0., i < n)


@triton.jit
def scatter_frame(nums, key, dest, extra, table, result, header, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); row = p // 16; j = p % 16
    e = tl.load(header + 4)
    d = tl.load(dest + row, row < e, -1); live = d >= 0
    k = tl.load(key + row, live, 0); ex = tl.load(extra + row, live, -1)
    val = tl.load(nums + p, live, 0.)
    c0 = tl.load(table + 33*k, (j == 0) & live, 0).to(tl.int32)
    z = tl.where(c0 >= 128, -1., 1.)
    tl.atomic_add(result + 128*d + (c0 & 127), z, mask=(j == 0) & live, sem="relaxed")
    tl.atomic_add(result + 128*ex + (c0 & 127), z, mask=(j == 0) & live & (ex >= 0), sem="relaxed")
    used = live & (val != 0.)
    h = tl.load(table + 33*k + j + 1, used, 0).to(tl.int32)
    g = tl.load(table + 33*k + j + 17, used, 0).to(tl.int32)
    v = tl.where(h >= 128, -val, val); u = tl.where(g >= 128, -val, val)
    tl.atomic_add(result + 128*d + (h & 127), v, mask=used, sem="relaxed")
    tl.atomic_add(result + 128*d + (g & 127), u, mask=used, sem="relaxed")
    tl.atomic_add(result + 128*ex + (h & 127), v, mask=used & (ex >= 0), sem="relaxed")
    tl.atomic_add(result + 128*ex + (g & 127), u, mask=used & (ex >= 0), sem="relaxed")


@triton.jit
def normalize(features, counts, header, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(header) * tl.load(header + 1)
    c = tl.load(counts + i // 128, i // 128 < n, 0).to(tl.float32)
    x = tl.load(features + i, i // 128 < n, 0.)
    x = tl.where(i % 128 == 127, tl.log(1. + c), x * tl.rsqrt(tl.maximum(c, 1.)))
    tl.store(features + i, x, i // 128 < n)


@triton.jit
def metadata(header, query_in, lengths, starts, lens, counts, blocks1, blocks2,
             queries, meta, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(header); t = tl.load(header + 1); q = tl.load(header + 2)
    n = tl.load(header + 3); k1 = tl.cdiv(t, 16); k2 = tl.cdiv(q, 16)
    dim = i % 4; row = i // 4
    extent = tl.where(dim == 0, t, tl.where(dim == 1, q, tl.where(dim == 2, k1, k2)))
    tl.store(starts + i, row.to(tl.int64) * extent, row < b)
    tl.store(lens + i, extent.to(tl.int64), row < b)
    length = tl.load(lengths + row, row < b, 0)
    tl.store(meta + i, tl.where(dim == 0, length, 0), row < b)
    query = tl.load(query_in + i, i < b*q, 0)
    tl.store(queries + i, (i // q).to(tl.int64) * t + query, i < b*q)
    r = i // 2
    tl.store(blocks1 + i, tl.where(i % 2 == 0, r // k1, (r % k1)*16), r < b*k1)
    tl.store(blocks2 + i, tl.where(i % 2 == 0, r // k2, (r % k2)*16), r < b*k2)
    total = tl.where(i == 0, b*t, tl.where(i == 1, b*q, tl.where(i == 2, b*k1,
            tl.where(i == 3, b*k2, tl.where(i == 4, b, n)))))
    tl.store(counts + i, total, i < 6)


@triton.jit
def select_frame(logits, offsets, lengths, draws, temperature, selected, header):
    row = tl.program_id(0)
    if row < tl.load(header):
        start = tl.load(offsets + row); n = tl.load(lengths + row)
        temp = tl.load(temperature); j = tl.arange(0, 32)
        mx = tl.full((), -float('inf'), tl.float64); valid = tl.full((), True, tl.int1)
        for k in range(0, tl.cdiv(n, 32)):
            x = tl.load(logits + start + k*32 + j, k*32+j < n, -float('inf')).to(tl.float64)
            valid = valid & (tl.sum(((k*32+j<n) & ((x!=x)|(tl.abs(x)==float('inf')))).to(tl.int32),0)==0)
            mx = tl.maximum(mx, tl.max(x, 0))
        z = tl.full((), 0., tl.float64)
        for k in range(0, tl.cdiv(n,32)):
            x = tl.load(logits+start+k*32+j, k*32+j<n, -float('inf')).to(tl.float64)
            z += tl.sum(tl.exp((x-mx)/temp), 0)
        u = tl.load(draws+row); previous = tl.full((),0.,tl.float64); choice = tl.full((),0,tl.int32)
        for k in range(0,tl.cdiv(n,32)):
            x = tl.load(logits+start+k*32+j,k*32+j<n,-float('inf')).to(tl.float64)
            prob = tl.exp((x-mx)/temp)/z
            choice += tl.sum(((previous+tl.cumsum(prob,0)<=u)&(k*32+j<n)).to(tl.int32),0)
            previous += tl.sum(prob,0)
        tl.store(selected+row,tl.where(valid,tl.minimum(choice,n-1),-1))


class FallbackFrame:
    def __init__(self, net, *, precision="ieee"):
        self.net = net
        self.flat = Fused(net, precision)
        if precision == 'ieee':
            from fallback_ieee_model import GraphScorer as Model
        elif precision == 'tf32x3':
            Model = GraphScorer
        else:
            raise ValueError(precision)
        self.model = Model(self.flat)
        self.table = hash_table()
        self.graph = None; self.capacity = (0,)*5
        self.host = self.device = None; self.dev = []; self.views = []
        self.buffers = {}; self.counts = collections.Counter(); self.closed = False
        self.net_signature = self.signature(tuple(self.net.parameters()))
        self.bound_signature = None

    @staticmethod
    def signature(values):
        result = []
        for t in values:
            try: v = t._version
            except RuntimeError: v = None
            result.append((t.data_ptr(), v))
        return tuple(result)

    def release(self):
        if self.graph is not None:
            torch.cuda.synchronize(); self.graph.reset(); self.graph = None
            self.counts['evictions'] += 1
        self.model.release()
        self.host = self.device = None; self.dev = []; self.views = []; self.buffers = {}
        self.capacity = (0,)*5; self.bound_signature = None
        self.header_host = self.temp_host = self.header = self.temperature = self.picks_host = None

    def close(self):
        if not self.closed:
            self.release(); self.model.close(); self.closed = True

    def statistics(self):
        return dict(self.counts, cached=int(self.graph is not None), closed=self.closed,
                    capacity=list(self.capacity), input_bytes=0 if self.host is None else self.host.numel(),
                    model=self.model.statistics())

    def allocate(self, capacity):
        self.release(); self.capacity = capacity
        b,t,q,n,e = capacity
        shapes = [(b,32),(n,24),(n,10),(n,),(t,),(t,),(t,),(t,),(q,),
                  (n,10),(n,),(e,16),(e,),(e,),(e,),(b,),(b,),(b,)]
        dtypes = [np.float32,np.float32,np.int32,np.int32,np.int32,np.int32,np.int32,np.bool_,
                  np.int32,np.int32,np.int32,np.float32,np.int32,np.int32,np.int32,np.int32,np.int32,np.float64]
        offsets=[]; end=64
        for shape,dtype in zip(shapes,dtypes):
            end=(end+7)//8*8; offsets.append(end)
            end += int(np.prod(shape))*np.dtype(dtype).itemsize
        self.host=torch.empty(end,dtype=torch.uint8,pin_memory=True)
        self.device=torch.empty_like(self.host,device='cuda')
        host=self.host.numpy()
        for shape,dtype,offset in zip(shapes,dtypes,offsets):
            view=np.ndarray(shape,dtype=dtype,buffer=host,offset=offset)
            self.views.append(view)
            torch_dtype=torch.from_numpy(np.empty(0,dtype=dtype)).dtype
            size=view.size*view.itemsize
            self.dev.append(self.device[offset:offset+size].view(torch_dtype).reshape(shape))
        self.header_host=np.ndarray((8,),np.int32,buffer=host,offset=0)
        self.temp_host=np.ndarray((1,),np.float64,buffer=host,offset=32)
        self.header=self.device[:32].view(torch.int32)
        self.temperature=self.device[32:40].view(torch.float64)
        def arr(shape,dtype=torch.float32):return torch.empty(shape,device='cuda',dtype=dtype)
        w=self.buffers=dict(features=arr((t,128)),queries=arr((q,),torch.int64),
            blocks1=arr((triton.cdiv(t,16)+b,2),torch.int32),blocks2=arr((triton.cdiv(q,16)+b,2),torch.int32),
            meta=arr((b,4),torch.int32),mapping=torch.arange(161,device='cuda',dtype=torch.int32).expand(b,161).contiguous(),
            logits=arr((n,)),picks=arr((b,),torch.int32))
        self.picks_host=torch.empty(b,dtype=torch.int32,pin_memory=True)
        d=self.dev
        bound=(w['features'],d[5],d[6],d[7],w['queries'],w['blocks1'],w['blocks2'],
               *d[:4],d[9],d[10],w['meta'],w['mapping'],w['logits'])
        self.model.allocate(tuple(x.data_ptr() for x in bound),bound,t,q,b)
        self.bound_signature=self.signature((self.flat.lookup,self.flat.head_weight,*self.net.parameters()))
        self.counts['allocations']+=1

    def execute(self):
        self.device.copy_(self.host,non_blocking=True)
        b,t,q,n,e=self.capacity; d=self.dev; w=self.buffers; m=self.model.buffers
        clear_features[(triton.cdiv(t*128,512),)](w['features'],self.header,512,num_warps=4)
        scatter_frame[(triton.cdiv(e*16,512),)](d[11],d[12],d[13],d[14],self.table,w['features'],self.header,512,num_warps=4)
        normalize[(triton.cdiv(t*128,512),)](w['features'],d[4],self.header,512,num_warps=4)
        extent=max(b*4,q,len(w['blocks1'])*2,len(w['blocks2'])*2,6)
        metadata[(triton.cdiv(extent,256),)](self.header,d[8],d[16],m['starts'],m['lens'],m['counts'],
                    w['blocks1'],w['blocks2'],w['queries'],w['meta'],256,num_warps=4)
        self.model.execute()
        select_frame[(b,)](w['logits'],d[15],d[16],d[17],self.temperature,w['picks'],self.header,num_warps=1)
        self.picks_host.copy_(w['picks'],non_blocking=True)

    @torch.inference_mode()
    def score(self,fields,t,temperatures):
        assert not self.closed and len(fields)==18
        b,actual_t=fields[4].shape; q=fields[8].shape[1]; n=len(fields[1]); e=len(fields[11])
        assert b>0 and actual_t==t and 1<=q<=t and n>0 and e>0
        need=(b,b*t,b*q,n,e)
        net_sig=self.signature(tuple(self.net.parameters()))
        if net_sig!=self.net_signature:
            self.release()
            self.flat.lookup=self.net.symbol_lift(self.net.symbol.weight).detach().contiguous()
            self.flat.head_weight=torch.cat((self.net.source_score.weight,self.net.left.weight,
                self.net.target_score.weight,self.net.right.weight),0).detach().contiguous()
            self.net_signature=net_sig
        sig=self.signature((self.flat.lookup,self.flat.head_weight,*self.net.parameters()))
        if self.host is None or sig!=self.bound_signature or any(a>c for a,c in zip(need,self.capacity)):
            self.allocate(tuple(triton.next_power_of_2(max(1,a,c)) for a,c in zip(need,self.capacity)))
        for src,dst in zip(fields,self.views,strict=True):
            assert src.dtype==dst.dtype
            np.copyto(dst.reshape(-1)[:src.size],src.reshape(-1))
        self.header_host[:]=[b,t,q,n,e,0,0,0]
        self.temp_host[0]=float(temperatures) if not torch.is_tensor(temperatures) else float(temperatures.item())
        assert self.temp_host[0]>0
        if self.graph is None:
            self.execute();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):self.execute()
            except BaseException:
                graph.reset();raise
            self.graph=graph;self.counts['captures']+=1
        self.graph.replay();torch.cuda.current_stream().synchronize()
        picks=self.picks_host[:b].numpy().copy()
        assert (picks>=0).all() and (picks<fields[16]).all(), 'nonfinite logits or illegal sample'
        self.counts['calls']+=1;self.counts['actual_rows']+=b;self.counts['input_bytes_copied']+=self.host.numel()
        return picks,self.buffers['logits'][:n]
