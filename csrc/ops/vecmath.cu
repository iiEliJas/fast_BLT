#include "ops/vecmath.h"
#include "core/backend.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int blt_cuda_launch_check(const char *what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

// One block reduces one dot product: strided partial sums in shared memory,
// then a deterministic tree reduction. Block size is fixed so the summation
// order is reproducible run to run.
#define BLT_VEC_DOT_BLOCK 256

__global__ void blt_vec_dot_kernel(const float *a, const float *b, size_t n, float *out) {
    __shared__ float partial[BLT_VEC_DOT_BLOCK];
    const size_t tid = threadIdx.x;

    float sum = 0.0f;
    for (size_t i = tid; i < n; i += BLT_VEC_DOT_BLOCK) {
        sum += a[i] * b[i];
    }
    partial[tid] = sum;
    __syncthreads();

    for (size_t stride = BLT_VEC_DOT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = partial[0];
    }
}

extern "C" float blt_vec_dot_cuda(blt_backend backend, const float *a, const float *b, size_t n) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_vec_dot: CUDA implementation called with non-CUDA backend");

    // Grow-only device scratch for the scalar result: grad clipping calls
    // this once per parameter tensor per step, and a cudaMalloc/cudaFree
    // round trip per call dominated that path on WSL2.
    static float *d_out = NULL;
    if (d_out == NULL) {
        cudaError_t err = cudaMalloc(&d_out, sizeof(float));
        if (err != cudaSuccess) {
            BLT_FATAL("blt_vec_dot: cudaMalloc failed: %s", cudaGetErrorString(err));
        }
    }
    blt_vec_dot_kernel<<<1, BLT_VEC_DOT_BLOCK>>>(a, b, n, d_out);
    blt_cuda_launch_check("blt_vec_dot_kernel");
    float result = 0.0f;
    cudaError_t err = cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        BLT_FATAL("blt_vec_dot: result copy failed: %s", cudaGetErrorString(err));
    }
    return result;
}

// Softmax over one contiguous row. Runs single-threaded and mirrors the CPU
// loop order exactly, which keeps masked rows bit-identical between backends.
__global__ void blt_softmax_masked_row_kernel(float *row, size_t row_len, size_t row_idx, int is_causal,
                                              const float *mask_row, float scale) {
    float max_val = -INFINITY;

    for (size_t col = 0; col < row_len; ++col) {
        if (mask_row != NULL) {
            row[col] = row[col] * scale + mask_row[col];
        } else if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
        }

        if (isfinite(row[col]) && row[col] > max_val) {
            max_val = row[col];
        }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < row_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    if (sum > 0.0f) {
        for (size_t col = 0; col < row_len; ++col) {
            row[col] /= sum;
        }
    }
}

extern "C" void blt_softmax_masked_row_inplace_cuda(blt_backend backend, float *row, size_t row_len, size_t row_idx,
                                                    int is_causal, const float *mask_row, float scale) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA,
                "blt_softmax_masked_row_inplace: CUDA implementation called with non-CUDA backend");

    blt_softmax_masked_row_kernel<<<1, 1>>>(row, row_len, row_idx, is_causal ? 1 : 0, mask_row, scale);
    blt_cuda_launch_check("blt_softmax_masked_row_inplace");
}

__global__ void blt_strided_copy_kernel(float *dst, size_t dst_stride, const float *src, size_t src_stride, size_t rows,
                                        size_t cols) {
    const size_t work = rows * cols;

    bool can_vectorize = (cols % 4 == 0) && (dst_stride == cols) && (src_stride == cols) &&
                         (reinterpret_cast<uintptr_t>(dst) % 16 == 0) && (reinterpret_cast<uintptr_t>(src) % 16 == 0);

    if (can_vectorize) {
        const float4 *src4 = reinterpret_cast<const float4 *>(src);
        float4 *dst4 = reinterpret_cast<float4 *>(dst);
        size_t cols4 = cols / 4;
        size_t work4 = rows * cols4;

        for (size_t off = (size_t)blockIdx.x * blockDim.x + threadIdx.x; off < work4;
             off += (size_t)gridDim.x * blockDim.x) {
            const size_t r = off / cols4;
            dst4[r * cols4 + (off % cols4)] = src4[r * cols4 + (off % cols4)];
        }
        return;
    }

    for (size_t off = (size_t)blockIdx.x * blockDim.x + threadIdx.x; off < work;
         off += (size_t)gridDim.x * blockDim.x) {
        const size_t r = off / cols;
        dst[r * dst_stride + (off % cols)] = src[r * src_stride + (off % cols)];
    }
}

extern "C" void blt_strided_copy_cuda(blt_backend backend, float *dst, size_t dst_stride, const float *src,
                                      size_t src_stride, size_t rows, size_t cols) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_strided_copy: CUDA implementation called with non-CUDA backend");
    {
        static int dump = -1;
        if (dump < 0) {
            dump = getenv("BLT_KVDBG") ? 1 : 0;
        }
        if (dump)
            fprintf(stderr, "[SC] dst=%p dstr=%zu src=%p sstr=%zu rows=%zu cols=%zu\n", (void *)dst, dst_stride,
                    (const void *)src, src_stride, rows, cols);
    }
    const size_t work = rows * cols;
    unsigned blocks = (unsigned)((work + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_strided_copy_kernel<<<blocks, 256>>>(dst, dst_stride, src, src_stride, rows, cols);
    blt_cuda_launch_check("blt_strided_copy");
}

__global__ void blt_fill_uniform_kernel(float *data, size_t n, uint64_t rng_state) {
    // xorshift64*
    uint64_t x = rng_state + threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        float val = (((float)(x >> 40) / 16777216.0f) * 2.0f - 1.0f);
        data[i] = val;
    }
}

__global__ void blt_fill_constant_kernel(float *data, size_t n, float v) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = v;
    }
}

extern "C" void blt_fill_uniform_cuda(blt_backend backend, float *data, size_t n, uint64_t *rng_state) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_fill_uniform: CUDA implementation called with non-CUDA backend");
    uint64_t seed = *rng_state;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_fill_uniform_kernel<<<blocks, 256>>>(data, n, seed);
    blt_cuda_launch_check("blt_fill_uniform");
    *rng_state = seed + n;
}

extern "C" void blt_fill_constant_cuda(blt_backend backend, float *data, size_t n, float v) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_fill_constant: CUDA implementation called with non-CUDA backend");
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_fill_constant_kernel<<<blocks, 256>>>(data, n, v);
    blt_cuda_launch_check("blt_fill_constant");
}

__global__ void blt_vec_scale_kernel(float *data, size_t n, float s) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= s;
}

extern "C" void blt_cuda_vec_scale(float *data, float s, size_t n) {
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_vec_scale_kernel<<<blocks, 256>>>(data, n, s);
    blt_cuda_launch_check("blt_vec_scale_kernel");
}

// ---------------------------------------------------------------------------
// SGD: w[i] -= lr * g[i]
// ---------------------------------------------------------------------------

__global__ void blt_visitor_sgd_kernel(float *w, const float *g, float lr, float grad_scale, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const float gi = g[i] * grad_scale;
        w[i] -= lr * gi;
    }
}

extern "C" void blt_cuda_sgd_step(float *w, const float *g, float lr, float grad_scale, size_t n) {
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_visitor_sgd_kernel<<<blocks, 256>>>(w, g, lr, grad_scale, n);
    blt_cuda_launch_check("blt_visitor_sgd_kernel");
}

// ---------------------------------------------------------------------------
// AdamW step: full update per element
// ---------------------------------------------------------------------------

__global__ void blt_visitor_adamw_kernel(float *w, const float *g, float *m, float *v, float lr, float b1, float b2,
                                         float eps, float wd, float bc1, float bc2, float grad_scale, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const float gi = g[i] * grad_scale;
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        float mhat = m[i] / bc1;
        float vhat = v[i] / bc2;
        w[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * w[i]);
    }
}

extern "C" void blt_cuda_adamw_step(float *w, const float *g, float *m, float *v, float lr, float b1, float b2,
                                    float eps, float wd, float bc1, float bc2, float grad_scale, size_t n) {
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    blt_visitor_adamw_kernel<<<blocks, 256>>>(w, g, m, v, lr, b1, b2, eps, wd, bc1, bc2, grad_scale, n);
    blt_cuda_launch_check("blt_visitor_adamw_kernel");
}

// ---------------------------------------------------------------------------
// AdamW squared-norm reduction: sum_i u_i^2 where
// u_i = m_hat/(sqrt(v_hat)+eps) + wd*w[i]
// Uses same single-block strided pattern as blt_vec_dot_kernel.
// ---------------------------------------------------------------------------

__global__ void blt_adamw_sqnorm_kernel(const float *w, const float *g, const float *m, const float *v, float b1,
                                        float b2, float eps, float wd, float bc1, float bc2, float grad_scale, size_t n,
                                        float *out) {
    __shared__ float partial[BLT_VEC_DOT_BLOCK];
    const size_t tid = threadIdx.x;

    float sum = 0.0f;
    for (size_t i = tid; i < n; i += BLT_VEC_DOT_BLOCK) {
        const float gi = g[i] * grad_scale;
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        float u = mhat / (sqrtf(vhat) + eps) + wd * w[i];
        sum += u * u;
    }
    partial[tid] = sum;
    __syncthreads();

    for (size_t stride = BLT_VEC_DOT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[0] = partial[0];
}

extern "C" float blt_cuda_adamw_sqnorm(const float *w, const float *g, const float *m, const float *v, float b1,
                                       float b2, float eps, float wd, float bc1, float bc2, float grad_scale,
                                       size_t n) {
    static float *d_out = NULL;
    if (d_out == NULL) {
        cudaError_t err = cudaMalloc(&d_out, sizeof(float));
        if (err != cudaSuccess) BLT_FATAL("blt_cuda_adamw_sqnorm: cudaMalloc failed: %s", cudaGetErrorString(err));
    }
    blt_adamw_sqnorm_kernel<<<1, BLT_VEC_DOT_BLOCK>>>(w, g, m, v, b1, b2, eps, wd, bc1, bc2, grad_scale, n, d_out);
    blt_cuda_launch_check("blt_adamw_sqnorm_kernel");
    float result = 0.0f;
    cudaError_t err = cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) BLT_FATAL("blt_cuda_adamw_sqnorm: copy failed: %s", cudaGetErrorString(err));
    return result;
}

extern "C" void blt_vec_dot_into_cuda(blt_backend backend, const float *a, const float *b, size_t n, float *out) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_vec_dot_into: CUDA implementation called with non-CUDA backend");
    blt_vec_dot_kernel<<<1, BLT_VEC_DOT_BLOCK>>>(a, b, n, out);
    blt_cuda_launch_check("blt_vec_dot_into_kernel");
}
