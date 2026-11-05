#include "test_helpers.h"
#include "test_suite.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/cuda_shim.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"
#include "ops/vecmath.h"
#include "train_diag.h"

#ifndef BLT_WITH_CUDA

int run_cuda_parity_clip_norm(void) { return 1; }

#else

#define CHECK_EMBED 16
#define CHECK_HIDDEN 32
#define CHECK_LAYERS 2
#define CHECK_MAX_SEQ 64
#define CHECK_MAX_PARAMS 128

struct pointer_ctx {
    float *ptrs[CHECK_MAX_PARAMS];
    size_t count;
};

static void record_ptr_fn(float *t, const blt_param_info *info, void *ctx) {
    struct pointer_ctx *c = (struct pointer_ctx *)ctx;
    BLT_REQUIRE(info->idx >= 0 && info->idx < CHECK_MAX_PARAMS, "clip parity: parameter index out of range");
    c->ptrs[info->idx] = t;
    c->count++;
}

struct upload_ctx {
    struct pointer_ctx *cpu;
    size_t count;
};

static void upload_fn(float *t, const blt_param_info *info, void *ctx) {
    struct upload_ctx *c = (struct upload_ctx *)ctx;
    BLT_REQUIRE(info->idx >= 0 && info->idx < CHECK_MAX_PARAMS, "clip parity: parameter index out of range");
    BLT_REQUIRE(c->cpu->ptrs[info->idx] != NULL, "clip parity: missing CPU source for parameter");
    blt_cuda_memcpy_h2d(t, c->cpu->ptrs[info->idx], info->numel * sizeof(float));
    c->count++;
}

static uint64_t next_rng(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x;
}

static void fill_grads_fn(float *t, const blt_param_info *info, void *ctx) {
    uint64_t *rng = (uint64_t *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        const int32_t r = (int32_t)(next_rng(rng) >> 40) % 2001;
        t[i] = ((float)r - 1000.0f) / 500.0f;
    }
}

struct sum_ctx {
    float sq;
    size_t count;
};

static void sum_dot_fn(float *t, const blt_param_info *info, void *ctx) {
    struct sum_ctx *c = (struct sum_ctx *)ctx;
    c->sq += blt_vec_dot(info->backend, t, t, info->numel);
    c->count++;
}

int run_cuda_parity_clip_norm(void) {
    blt_arena *cpu_arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    blt_arena *cuda_arena = blt_arena_create(64u << 20, BLT_BACKEND_CUDA);
    TEST_ASSERT(cpu_arena != NULL && cuda_arena != NULL);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, CHECK_EMBED, CHECK_HIDDEN, CHECK_LAYERS, CHECK_LAYERS, CHECK_LAYERS, CHECK_MAX_SEQ,
                              0);
    blt_model *cpu_model = blt_model_create(cpu_arena, &cfg);
    blt_model *cuda_model = blt_model_create(cuda_arena, &cfg);
    TEST_ASSERT(cpu_model != NULL && cuda_model != NULL);

    blt_model_grad *cpu_grad = blt_model_grad_create(cpu_arena, cpu_model);
    blt_model_grad *cuda_grad = blt_model_grad_create(cuda_arena, cuda_model);
    TEST_ASSERT(cpu_grad != NULL && cuda_grad != NULL);

    // Fill the CPU gradients, then upload the same bytes into the CUDA copies.
    uint64_t fill_rng = 0x0DDC0FFEEULL;
    blt_model_visit_params(cpu_model, cpu_grad, fill_grads_fn, &fill_rng, 1);

    struct pointer_ctx cpu_ptrs;
    memset(&cpu_ptrs, 0, sizeof(cpu_ptrs));
    blt_model_visit_params(cpu_model, cpu_grad, record_ptr_fn, &cpu_ptrs, 1);
    TEST_ASSERT(cpu_ptrs.count > 64);

    struct upload_ctx up = {&cpu_ptrs, 0};
    blt_model_visit_params(cuda_model, cuda_grad, upload_fn, &up, 1);
    TEST_ASSERT(up.count == cpu_ptrs.count);

    // Old path: one D2H per tensor, host sum in visitor order.
    struct sum_ctx cpu_ref = {0.0f, 0};
    blt_model_visit_params(cpu_model, cpu_grad, sum_dot_fn, &cpu_ref, 1);
    const float cpu_norm_ref = sqrtf(cpu_ref.sq);
    TEST_ASSERT(cpu_ref.count > 64);

    struct sum_ctx cuda_ref = {0.0f, 0};
    blt_model_visit_params(cuda_model, cuda_grad, sum_dot_fn, &cuda_ref, 1);
    const float cuda_norm_ref = sqrtf(cuda_ref.sq);
    TEST_ASSERT(cuda_ref.count == cpu_ref.count);

    // New path: one batched device-to-host download, same host summation order.
    const float max_norm = cpu_norm_ref * 0.25f;
    float cpu_scale = -1.0f;
    float cuda_scale = -1.0f;
    const float cpu_norm = blt_grad_clip_norm(cpu_model, cpu_grad, max_norm, &cpu_scale);
    const float cuda_norm = blt_grad_clip_norm(cuda_model, cuda_grad, max_norm, &cuda_scale);

    TEST_ASSERT(cpu_norm == cpu_norm_ref);
    TEST_ASSERT(cuda_norm == cuda_norm_ref);
    TEST_ASSERT(cpu_scale == max_norm / cpu_norm_ref);
    TEST_ASSERT(cuda_scale == max_norm / cuda_norm_ref);

    // GPU per-tensor dots use the block-reduction kernel, so their sum differs
    // from the CPU sequential sum by fp rounding only.
    const float rel = fabsf(cuda_norm_ref - cpu_norm_ref) / cpu_norm_ref;
    TEST_ASSERT(rel < 1e-4f);

    // A threshold above the norm must leave the unit scale in both backends.
    const float huge = cpu_norm_ref * 2.0f;
    TEST_ASSERT(blt_grad_clip_norm(cpu_model, cpu_grad, huge, &cpu_scale) == cpu_norm_ref);
    TEST_ASSERT(blt_grad_clip_norm(cuda_model, cuda_grad, huge, &cuda_scale) == cuda_norm_ref);
    TEST_ASSERT(cpu_scale == 1.0f && cuda_scale == 1.0f);

    blt_arena_destroy(cpu_arena);
    blt_arena_destroy(cuda_arena);
    return 1;
}

#endif
