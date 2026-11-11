#include "core/tensor.h"
#include "core/allocator.h"
#include "core/backend.h"
#include "core/cuda_shim.h"

size_t blt_tensor_compute_numel(const size_t *shape, size_t ndim) {
    if (ndim == 0) {
        return 0;
    }

    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }
    return numel;
}

void blt_tensor_compute_row_major_strides(const size_t *shape, size_t ndim, size_t *strides_out) {
    if (ndim == 0) {
        return;
    }

    size_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {
        strides_out[i] = stride;
        stride *= shape[i];
    }
}

size_t blt_tensor_bytes(const blt_tensor *t) { return t->numel * blt_dtype_sizeof(t->dtype); }

void view_1d(blt_tensor *view, void *data, size_t len, blt_dtype dtype, blt_backend backend) {
    view->data = data;
    for (size_t d = 0; d < BLT_MAX_NDIM; ++d) {
        view->shape[d] = 0;
        view->strides[d] = 0;
    }
    view->ndim = 1;
    view->shape[0] = len;
    view->strides[0] = 1;
    view->numel = len;
    view->dtype = dtype;
    view->backend = backend;
    view->is_view = true;
    view->scale = NULL;
    view->transposed = false;
}

void view_1d_offset(blt_tensor *view, const blt_tensor *src, size_t offset_elems, size_t len) {
    view_1d(view, (char *)src->data + offset_elems * blt_dtype_sizeof(src->dtype), len, src->dtype, src->backend);
}

void blt_tensor_view_2d(blt_tensor *t, void *data, size_t rows, size_t cols, blt_backend backend) {
    t->data = data;
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->shape[2] = 0;
    t->shape[3] = 0;
    t->ndim = 2;
    t->numel = blt_tensor_compute_numel(t->shape, 2);
    t->dtype = BLT_DTYPE_FP32;
    t->backend = backend;
    t->is_view = true;
    t->scale = NULL;
    t->transposed = false;
    blt_tensor_compute_row_major_strides(t->shape, 2, t->strides);
}

void blt_tensor_view_3d(blt_tensor *t, void *data, size_t d0, size_t d1, size_t d2, blt_backend backend) {
    t->data = data;
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    t->shape[3] = 0;
    t->ndim = 3;
    t->numel = blt_tensor_compute_numel(t->shape, 3);
    t->dtype = BLT_DTYPE_FP32;
    t->backend = backend;
    t->is_view = true;
    t->scale = NULL;
    t->transposed = false;
    blt_tensor_compute_row_major_strides(t->shape, 3, t->strides);
}

void zero_tensor(blt_tensor *t) {
    if (t->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_memset(t->data, 0, blt_tensor_bytes(t));
#endif
    } else {
        memset(t->data, 0, blt_tensor_bytes(t));
    }
}

blt_tensor blt_tensor_to_device(const blt_tensor *src, blt_arena *device_arena) {
    BLT_REQUIRE(src != NULL && device_arena != NULL && src->data != NULL,
                "blt_tensor_to_device: src and device_arena must be valid");
    BLT_REQUIRE(device_arena->backend == BLT_BACKEND_CUDA,
                "blt_tensor_to_device: destination arena must use the CUDA backend");

    blt_tensor dst = blt_tensor_create(device_arena, src->shape, src->ndim, src->dtype);
#ifdef BLT_WITH_CUDA
    if (src->backend == BLT_BACKEND_CUDA) {
        BLT_FATAL("blt_tensor_to_device: source tensor is already on the CUDA backend");
    }
    blt_cuda_memcpy_h2d(dst.data, src->data, blt_tensor_bytes(src));
#else
    BLT_FATAL("blt_tensor_to_device: built without BLT_WITH_CUDA");
#endif
    return dst;
}

blt_tensor blt_tensor_to_host(const blt_tensor *src, blt_arena *host_arena) {
    BLT_REQUIRE(src != NULL && host_arena != NULL && src->data != NULL,
                "blt_tensor_to_host: src and host_arena must be valid");
    BLT_REQUIRE(host_arena->backend == BLT_BACKEND_CPU,
                "blt_tensor_to_host: destination arena must use the CPU backend");

    blt_tensor dst = blt_tensor_create(host_arena, src->shape, src->ndim, src->dtype);
#ifdef BLT_WITH_CUDA
    if (src->backend != BLT_BACKEND_CUDA) {
        memcpy(dst.data, src->data, blt_tensor_bytes(src));
        return dst;
    }
    blt_cuda_memcpy_d2h(dst.data, src->data, blt_tensor_bytes(src));
#else
    memcpy(dst.data, src->data, blt_tensor_bytes(src));
#endif
    return dst;
}

void blt_tensor_upload(blt_tensor *dst, const void *host_src, size_t bytes) {
    BLT_REQUIRE(dst != NULL && host_src != NULL && dst->data != NULL,
                "blt_tensor_upload: dst and host_src must be valid");
    BLT_REQUIRE(bytes <= blt_tensor_bytes(dst), "blt_tensor_upload: byte count exceeds tensor storage");

#ifdef BLT_WITH_CUDA
    if (dst->backend == BLT_BACKEND_CUDA) {
        blt_cuda_memcpy_h2d(dst->data, host_src, bytes);
        return;
    }
#endif
    memcpy(dst->data, host_src, bytes);
}

void blt_tensor_download(const blt_tensor *src, void *host_dst, size_t bytes) {
    BLT_REQUIRE(src != NULL && host_dst != NULL && src->data != NULL,
                "blt_tensor_download: src and host_dst must be valid");
    BLT_REQUIRE(bytes <= blt_tensor_bytes(src), "blt_tensor_download: byte count exceeds tensor storage");

#ifdef BLT_WITH_CUDA
    if (src->backend == BLT_BACKEND_CUDA) {
        blt_cuda_memcpy_d2h(host_dst, src->data, bytes);
        return;
    }
#endif
    memcpy(host_dst, src->data, bytes);
}

void *blt_container_alloc_cpu(blt_arena *arena, size_t bytes) { return blt_arena_alloc(arena, bytes, 64); }

void *blt_container_alloc_cuda(blt_arena *arena, size_t bytes) {
    (void)arena;
    void *p = malloc(bytes);
    BLT_REQUIRE(p != NULL, "blt_container_alloc: host allocation failed");
    memset(p, 0, bytes);
    return p;
}

void blt_tensor_copy_from_host(blt_tensor *t, const void *host_src, size_t bytes) {
    BLT_REQUIRE(t != NULL && host_src != NULL, "blt_tensor_copy_from_host: arguments cannot be NULL");
    BLT_REQUIRE(bytes <= blt_tensor_bytes(t), "blt_tensor_copy_from_host: copy size exceeds tensor size");

    if (t->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_memcpy_h2d(t->data, host_src, bytes);
#else
        BLT_FATAL("blt_tensor_copy_from_host: CUDA backend not available in this build");
#endif
    } else {
        memcpy(t->data, host_src, bytes);
    }
}

void blt_tensor_copy_to_host(const blt_tensor *t, void *host_dst, size_t bytes) {
    BLT_REQUIRE(t != NULL && host_dst != NULL, "blt_tensor_copy_to_host: arguments cannot be NULL");
    BLT_REQUIRE(bytes <= blt_tensor_bytes(t), "blt_tensor_copy_to_host: copy size exceeds tensor size");

    if (t->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_memcpy_d2h(host_dst, t->data, bytes);
#else
        BLT_FATAL("blt_tensor_copy_to_host: CUDA backend not available in this build");
#endif
    } else {
        memcpy(host_dst, t->data, bytes);
    }
}
