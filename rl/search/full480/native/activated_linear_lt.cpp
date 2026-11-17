#include <cublasLt.h>
#include <cstdint>
#include <cstring>
#include <array>
#define TRY(expr) do { auto status=(expr); if(status!=CUBLAS_STATUS_SUCCESS) return int(status); } while(0)
struct Params {
 cublasLtHandle_t lt{}; cublasLtMatmulDesc_t op{};
 cublasLtMatrixLayout_t a{},b{},c{}; cublasLtMatmulPreference_t pref{};
 ~Params(){if(pref)cublasLtMatmulPreferenceDestroy(pref); if(a)cublasLtMatrixLayoutDestroy(a); if(b)cublasLtMatrixLayoutDestroy(b); if(c)cublasLtMatrixLayoutDestroy(c); if(op)cublasLtMatmulDescDestroy(op); if(lt)cublasLtDestroy(lt);}
 int init(int m,int n,int k,const void* bias,size_t workspace,int activation=0){
  TRY(cublasLtCreate(&lt)); TRY(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;TRY(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  if(bias){cublasLtEpilogue_t ep=activation==1?CUBLASLT_EPILOGUE_GELU_BIAS:(activation==2?CUBLASLT_EPILOGUE_RELU_BIAS:CUBLASLT_EPILOGUE_BIAS);TRY(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));TRY(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));}
  TRY(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,k,n,k));TRY(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,k,m,k));TRY(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,n,m,n));
  TRY(cublasLtMatmulPreferenceCreate(&pref));TRY(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));return 0;
 }
};
extern "C" size_t plan_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
extern "C" int get_plans(int m,int n,int k,const void* bias,void* plans,int* num,size_t ws,int activation){
 Params p;int status=p.init(m,n,k,bias,ws,activation);if(status)return status;
 std::array<cublasLtMatmulHeuristicResult_t,32> hs{};int count=0;TRY(cublasLtMatmulAlgoGetHeuristic(p.lt,p.op,p.a,p.b,p.c,p.c,p.pref,hs.size(),hs.data(),&count));
 auto* out=static_cast<cublasLtMatmulAlgo_t*>(plans);*num=0;for(int i=0;i<count;++i)if(hs[i].state==CUBLAS_STATUS_SUCCESS && hs[i].workspaceSize<=ws)out[(*num)++]=hs[i].algo;return 0;
}
extern "C" int execute_plan(int m,int n,int k,const void*x,const void*w,const void*bias,void*y,void* workspace,size_t size,const void* plan,void* stream){
 Params p;int status=p.init(m,n,k,bias,size);if(status)return status;
 const auto* algo=static_cast<const cublasLtMatmulAlgo_t*>(plan);
 cublasLtMatmulHeuristicResult_t test{};TRY(cublasLtMatmulAlgoCheck(p.lt,p.op,p.a,p.b,p.c,p.c,algo,&test));
 if(test.state!=CUBLAS_STATUS_SUCCESS || test.workspaceSize>size)return 15;
 const float alpha=1,beta=0;
 TRY(cublasLtMatmul(p.lt,p.op,&alpha,w,p.a,x,p.b,&beta,y,p.c,y,p.c,algo,workspace,size,reinterpret_cast<cudaStream_t>(stream)));return 0;
}
