#include "core/backend.h"
#include "ops/quant.h"

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ static __forceinline__ unsigned char cvt_to_fp8(float v, int e5m2) {
    return e5m2 ? (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E5M2)
                : (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ static __forceinline__ float cvt_from_fp8(unsigned char v, int e5m2) {
    __half_raw h = e5m2 ? __nv_cvt_fp8_to_halfraw(v, __NV_E5M2) : __nv_cvt_fp8_to_halfraw(v, __NV_E4M3);
    return __half2float(h);
}

// One block reduces the whole tensor: max is order-independent, so this stays
// deterministic regardless of how the grid-stride loop is scheduled.
static __global__ void amax_kernel(const float *x, float *out, long long n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float m = 0.0f;
    for (long long i = tid; i < n; i += blockDim.x) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0];
}

static __global__ void quant_scale_kernel(const float *amax, float *out, float max_value, float min_scale) {
    float s = amax[0] / max_value;
    out[0] = (s >= min_scale) ? s : min_scale;
}

static __global__ void quantize_kernel(const float *x, unsigned char *out, float inv, int rows, int cols, int transpose,
                                       int e5m2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols;
    int c = i - r * cols;
    unsigned char q = cvt_to_fp8(x[i] * inv, e5m2);
    if (transpose) {
        out[(long long)c * rows + r] = q;
    } else {
        out[i] = q;
    }
}

static __global__ void dequantize_kernel(const unsigned char *x, float *out, float scale, int rows, int cols,
                                         int transpose, int e5m2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols;
    int c = i - r * cols;
    if (transpose) {
        out[i] = cvt_from_fp8(x[(long long)c * rows + r], e5m2) * scale;
    } else {
        out[i] = cvt_from_fp8(x[i], e5m2) * scale;
    }
}

static __global__ void roundtrip_kernel(const float *x, float *out, float scale, int n, int e5m2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.0f / scale;
    out[i] = cvt_from_fp8(cvt_to_fp8(x[i] * inv, e5m2), e5m2) * scale;
}

extern "C" void blt_amax_cuda(const blt_tensor *x, blt_tensor *out) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_amax: x and out must not be NULL");
    BLT_REQUIRE(x->dtype == BLT_DTYPE_FP32, "blt_amax: x must be FP32");
    BLT_REQUIRE(out->numel >= 1 && out->dtype == BLT_DTYPE_FP32, "blt_amax: out must hold at least one FP32 value");
    if (x->numel == 0) {
        return;
    }
    const int threads = 1024;
    amax_kernel<<<1, threads, threads * sizeof(float)>>>((const float *)x->data, (float *)out->data,
                                                         (long long)x->numel);
}

extern "C" void blt_quant_scale_cuda(const blt_tensor *amax_in, float max_value, float min_scale,
                                     blt_tensor *scale_out) {
    BLT_REQUIRE(amax_in != NULL && scale_out != NULL, "blt_quant_scale: inputs must not be NULL");
    BLT_REQUIRE(amax_in->numel >= 1 && scale_out->numel >= 1, "blt_quant_scale: inputs must hold one value");
    BLT_REQUIRE(max_value > 0.0f, "blt_quant_scale: max_value must be positive");
    quant_scale_kernel<<<1, 1>>>((const float *)amax_in->data, (float *)scale_out->data, max_value, min_scale);
}

extern "C" void blt_fp8_quantize_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_quantize: x and out must not be NULL");
    BLT_REQUIRE(x->ndim == 2 && out->ndim == 2, "blt_fp8_quantize: x and out must be 2D");
    BLT_REQUIRE(x->dtype == BLT_DTYPE_FP32, "blt_fp8_quantize: x must be FP32");
    BLT_REQUIRE(blt_dtype_is_fp8(out->dtype), "blt_fp8_quantize: out must be an FP8 dtype");

    const int rows = (int)x->shape[0];
    const int cols = (int)x->shape[1];
    const float inv = 1.0f / (scale ? ((const float *)scale->data)[0] : 1.0f);
    const int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>((const float *)x->data, (unsigned char *)out->data, inv, rows, cols, transpose,
                                         out->dtype == BLT_DTYPE_FP8_E5M2);
}

extern "C" void blt_fp8_dequantize_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_dequantize: x and out must not be NULL");
    BLT_REQUIRE(x->ndim == 2 && out->ndim == 2, "blt_fp8_dequantize: x and out must be 2D");
    BLT_REQUIRE(blt_dtype_is_fp8(x->dtype), "blt_fp8_dequantize: x must be an FP8 dtype");
    BLT_REQUIRE(out->dtype == BLT_DTYPE_FP32, "blt_fp8_dequantize: out must be FP32");

    const int rows = (int)(transpose ? x->shape[1] : x->shape[0]);
    const int cols = (int)(transpose ? x->shape[0] : x->shape[1]);
    const float s = scale ? ((const float *)scale->data)[0] : 1.0f;
    const int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    dequantize_kernel<<<blocks, threads>>>((const unsigned char *)x->data, (float *)out->data, s, rows, cols, transpose,
                                           x->dtype == BLT_DTYPE_FP8_E5M2);
}

extern "C" void blt_fp8_roundtrip_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_roundtrip: x and out must not be NULL");
    const float s = scale ? ((const float *)scale->data)[0] : 1.0f;
    const int threads = 256;
    int blocks = ((int)x->numel + threads - 1) / threads;
    roundtrip_kernel<<<blocks, threads>>>((const float *)x->data, (float *)out->data, s, (int)x->numel,
                                          out->dtype == BLT_DTYPE_FP8_E5M2);
}
