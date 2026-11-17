#include "activated_linear_lt.cpp"
struct FrozenPlan {Params p; cublasLtMatmulAlgo_t algo{};};
extern "C" int open_plan(int m,int n,int k,const void* bias,const void* algo,size_t ws,void** result,int activation){
 auto* q=new FrozenPlan;int rc=q->p.init(m,n,k,bias,ws,activation);
 if(rc){delete q;return rc;}std::memcpy(&q->algo,algo,sizeof(q->algo));
 cublasLtMatmulHeuristicResult_t info{};
 auto& p=q->p;rc=cublasLtMatmulAlgoCheck(p.lt,p.op,p.a,p.b,p.c,p.c,&q->algo,&info);
 if(rc||info.state!=CUBLAS_STATUS_SUCCESS||info.workspaceSize>ws){delete q;return rc?rc:15;}
 *result=q;return 0;
}
extern "C" int apply_plan(void* owned,const void*x,const void*w,void*y,void* scratch,size_t size,void* stream){
 auto* q=static_cast<FrozenPlan*>(owned);auto& p=q->p;float alpha=1.f,beta=0.f;
 return cublasLtMatmul(p.lt,p.op,&alpha,w,p.a,x,p.b,&beta,y,p.c,y,p.c,&q->algo,scratch,size,reinterpret_cast<cudaStream_t>(stream));
}
extern "C" void close_plan(void* owned){delete static_cast<FrozenPlan*>(owned);}
