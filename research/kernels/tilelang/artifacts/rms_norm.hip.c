#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/scan.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(256) rms_norm_kernel(float* __restrict__ W, float* __restrict__ X, float* __restrict__ Y) {
  float sq[20];
  float tot[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    float4 __1;
      float4 v_ = *(float4*)(X + ((i * 1024) + (((int)threadIdx.x) * 4)));
      __1.x = (v_.x*v_.x);
      __1.y = (v_.y*v_.y);
      __1.z = (v_.z*v_.z);
      __1.w = (v_.w*v_.w);
    *(float4*)(sq + (i * 4)) = __1;
  }
  tot[0] = 0.000000e+00f;
  #pragma unroll
  for (int rv = 0; rv < 20; ++rv) {
    tot[0] = (tot[0] + sq[(((rv % 5) * 4) + (rv / 5))]);
  }
  tot[0] = tl::AllReduce<tl::SumOp, 256, 1, 0>::run(tot[0], (&(workspace[0])));
  float scale = (1.000000e+00f / sqrtf(((tot[0] / 5.120000e+03f) + 1.000000e-06f)));
  #pragma unroll
  for (int i_1 = 0; i_1 < 5; ++i_1) {
    float4 __2;
      float4 __3;
        float4 v__1 = *(float4*)(W + ((i_1 * 1024) + (((int)threadIdx.x) * 4)));
        float4 v__2 = *(float4*)(X + ((i_1 * 1024) + (((int)threadIdx.x) * 4)));
        __3.x = (v__1.x*v__2.x);
        __3.y = (v__1.y*v__2.y);
        __3.z = (v__1.z*v__2.z);
        __3.w = (v__1.w*v__2.w);
      float4 v__3 = make_float4(scale, scale, scale, scale);
      __2.x = (__3.x*v__3.x);
      __2.y = (__3.y*v__3.y);
      __2.z = (__3.z*v__3.z);
      __2.w = (__3.w*v__3.w);
    *(float4*)(Y + ((i_1 * 1024) + (((int)threadIdx.x) * 4))) = __2;
  }
}

