#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/scan.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(128) gemv_q80_kernel(signed char* __restrict__ Qs, half_t* __restrict__ Scales, bfloat16_t* __restrict__ X, float* __restrict__ Y) {
  float acc[1];
  float xf[1];
  float prod[16];
  float red_clear[16];
  float red[1];
  extern __shared__ __align__(1024) float workspace[];
  acc[0] = 0.000000e+00f;
  xf[0] = ((float)X[((int)threadIdx.x)]);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float d = ((float)Scales[(((((int)blockIdx.x) * 64) + (i * 4)) + (((int)threadIdx.x) >> 5))]);
    float wq = ((float)Qs[(((((int)blockIdx.x) * 2048) + (i * 128)) + ((int)threadIdx.x))]);
    prod[i] = ((d * wq) * xf[0]);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    red_clear[i_1] = 0.000000e+00f;
    red_clear[i_1] = (red_clear[i_1] + prod[i_1]);
    __syncthreads();
    red_clear[i_1] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(red_clear[i_1], (&(workspace[0])));
    if ((((int)threadIdx.x) & 15) == i_1) {
      red[0] = red_clear[i_1];
    }
  }
  acc[0] = (acc[0] + red[0]);
  if ((((int)threadIdx.x) >> 4) == 0) {
    Y[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) & 15))] = acc[0];
  }
}

