cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(fBLT
  VERSION 1.0.0
  LANGUAGES C
  DESCRIPTION "Fast Byte Latent Transformer in pure C and CUDA"
)

# -- C Standard --------------------------------------------------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# -- Options -----------------------------------------------------------------
option(USE_CUDA "Enable CUDA backend (Linux + NVIDIA GPU required)" OFF)

# -- CUDA Setup --------------------------------------------------------------
if(USE_CUDA)
  # Prefer a standalone CUDA toolkit over the apt nvcc package.
  # A user-supplied -DCUDAToolkit_ROOT / -DCMAKE_CUDA_COMPILER still wins.
  if(NOT CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit install prefix")
  endif()
  if(NOT CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_ROOT)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(BLT_WITH_CUDA)
endif()

# -- Platform Detection ------------------------------------------------------
if(NOT WIN32)
  add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# -- Common Compile Options --------------------------------------------------
add_compile_options(-Wall -Wextra)

# -- Include Directories (global for all targets) ----------------------------
include_directories(csrc tests)

# ============================================================================
# CORE LIBRARY
# ============================================================================
set(BLT_CORE_SOURCES
  csrc/core/tensor.c
  csrc/core/json.c
  csrc/core/allocator.c
  csrc/core/backend.c
  csrc/ops/elementwise_cpu.c
  csrc/ops/linalg_cpu.c
  csrc/ops/reductions_cpu.c
  csrc/ops/optim_cpu.c
  csrc/ops/mask_builder_cpu.c
  csrc/ops/patch_pool_cpu.c
  csrc/ops/vecmath_cpu.c
  csrc/ops/attn_core_cpu.c
  csrc/ops/gather_scatter_cpu.c
  csrc/ops/row_stats_cpu.c
  csrc/ops/cast_cpu.c
  csrc/ops/quant_cpu.c
  csrc/models/entropy.c
  csrc/models/byte_embedding.c
  csrc/models/patcher.c
  csrc/models/attention.c
  csrc/models/cross_attention.c
  csrc/models/transformer.c
  csrc/models/transformer_stack.c
  csrc/models/entropy_lm.c
  csrc/models/hash_ngram.c
  csrc/models/local_common.c
  csrc/models/local_encoder.c
  csrc/models/local_decoder.c
  csrc/models/block_diffusion.c
  csrc/models/global_transformer.c
  csrc/models/model.c
  csrc/models/model_builder.c
  csrc/train_args.c
  csrc/train_optim.c
  csrc/train_diag.c
  csrc/train_eval.c
  csrc/train_schedule.c
  csrc/models/checkpoint.c
  csrc/models/param_visitor.c
  csrc/infer/stats.c
  csrc/infer/rope_gather.c
  csrc/infer/kv_cache.c
  csrc/infer/block_generation.c
  csrc/infer/self_speculation.c
  csrc/core/generate_greedy.c
  csrc/core/decode_select.c
  csrc/core/flops.c
  csrc/core/device_info.c
)

# CUDA sources (only when USE_CUDA is ON)
if(USE_CUDA)
  list(APPEND BLT_CORE_SOURCES
    csrc/ops/memory.cu
    csrc/ops/vecmath.cu
    csrc/ops/elementwise.cu
    csrc/ops/reductions.cu
    csrc/ops/linalg.cu
    csrc/ops/mask_builder.cu
    csrc/ops/patch_pool.cu
    csrc/ops/optim.cu
    csrc/ops/attn_core.cu
    csrc/ops/gather_scatter.cu
    csrc/ops/row_stats.cu
    csrc/ops/cast_cuda.cu
    csrc/ops/quant_cuda.cu
  )
endif()

add_library(blt_core STATIC ${BLT_CORE_SOURCES})

target_compile_options(blt_core PRIVATE -O2)

if(USE_CUDA)
  target_link_libraries(blt_core PRIVATE CUDA::cudart CUDA::cublas)
endif()

# ============================================================================
# BENCH HARNESS LIBRARY
# ============================================================================
add_library(bench_harness STATIC bench/harness.c)
target_include_directories(bench_harness PRIVATE bench)
target_compile_options(bench_harness PRIVATE -O2)

# ============================================================================
# EXECUTABLES
# ============================================================================

# -- Main --------------------------------------------------------------------
add_executable(main csrc/main.c)
target_link_libraries(main PRIVATE blt_core m)

# -- Test Suite --------------------------------------------------------------
file(GLOB_RECURSE TEST_UNIT_SOURCES CONFIGURE_DEPENDS tests/unit/*/*.c)
file(GLOB_RECURSE TEST_PARITY_SOURCES CONFIGURE_DEPENDS tests/parity/*.c)

add_executable(test_main
  ${TEST_UNIT_SOURCES}
  ${TEST_PARITY_SOURCES}
  tests/test_helpers.c
  tests/test_main.c
)
target_link_libraries(test_main PRIVATE blt_core m)

# -- Sandbox -----------------------------------------------------------------
add_executable(sandbox csrc/sandbox.c)
target_link_libraries(sandbox PRIVATE blt_core m)

# -- BLT-D Trainer ----------------------------------------------------------
add_executable(train_blt_d csrc/train_blt_d.c)
target_link_libraries(train_blt_d PRIVATE blt_core m)

# -- Inference Binary --------------------------------------------------------
add_executable(infer csrc/infer.c)
target_link_libraries(infer PRIVATE blt_core m)

# -- End-to-End Driver ------------------------------------------------------
add_executable(e2e_blt_dv csrc/e2e_blt_dv.c)
target_link_libraries(e2e_blt_dv PRIVATE blt_core m)

# -- Tools -------------------------------------------------------------------
add_executable(sanity_check csrc/tools/sanity_check.c)
target_link_libraries(sanity_check PRIVATE blt_core m)

add_executable(last_row_acc csrc/tools/last_row_acc.c)
target_link_libraries(last_row_acc PRIVATE blt_core m)

add_executable(pos_accuracy csrc/tools/pos_accuracy.c)
target_link_libraries(pos_accuracy PRIVATE blt_core m)

add_executable(masked_acc csrc/tools/masked_acc.c)
target_link_libraries(masked_acc PRIVATE blt_core m)

add_executable(selfspec_audit csrc/tools/selfspec_audit.c)
target_link_libraries(selfspec_audit PRIVATE blt_core m)

add_executable(patch_trunc_split csrc/tools/patch_trunc_split.c)
target_link_libraries(patch_trunc_split PRIVATE blt_core m)

add_executable(patch_final_split csrc/tools/patch_final_split.c)
target_link_libraries(patch_final_split PRIVATE blt_core m)

add_executable(greedy_tf_acc csrc/tools/greedy_tf_acc.c)
target_link_libraries(greedy_tf_acc PRIVATE blt_core m)

add_executable(eval_paper_rule_split csrc/tools/eval_paper_rule_split.c)
target_link_libraries(eval_paper_rule_split PRIVATE blt_core m)

add_executable(eval_isolate_redirect csrc/tools/eval_isolate_redirect.c)
target_link_libraries(eval_isolate_redirect PRIVATE blt_core m)

add_executable(ownlatent_value csrc/tools/ownlatent_value.c)
target_link_libraries(ownlatent_value PRIVATE blt_core m)

add_executable(leak_probe csrc/tools/leak_probe.c)
target_link_libraries(leak_probe PRIVATE blt_core m)

add_executable(prefix_invariance csrc/tools/prefix_invariance.c)
target_link_libraries(prefix_invariance PRIVATE blt_core m)

add_executable(entropy_lm_eval csrc/tools/entropy_lm_eval.c)
target_link_libraries(entropy_lm_eval PRIVATE blt_core m)

# -- Benchmarks --------------------------------------------------------------
add_executable(bench_patcher tests/bench/bench_patcher.c)
target_include_directories(bench_patcher PRIVATE bench)
target_link_libraries(bench_patcher PRIVATE blt_core m)

add_executable(bench_harness_selftest tests/bench/bench_harness_selftest.c)
target_include_directories(bench_harness_selftest PRIVATE ${CMAKE_SOURCE_DIR} bench)
target_link_libraries(bench_harness_selftest PRIVATE bench_harness m)

add_executable(infer_bench bench/infer_bench.c)
target_include_directories(infer_bench PRIVATE bench)
target_link_libraries(infer_bench PRIVATE blt_core bench_harness m)

add_executable(train_sweep csrc/train_sweep.c)
target_include_directories(train_sweep PRIVATE bench)
target_link_libraries(train_sweep PRIVATE blt_core bench_harness m)

# -- CUDA-Only Targets ------------------------------------------------------
if(USE_CUDA)
  add_executable(cuda_smoke csrc/cuda_smoke.c csrc/ops/smoke.cu)
  target_link_libraries(cuda_smoke PRIVATE blt_core CUDA::cudart CUDA::cublas m)

  add_executable(cuda_bench bench/cuda_bench.c)
  target_include_directories(cuda_bench PRIVATE bench)
  target_link_libraries(cuda_bench PRIVATE blt_core bench_harness CUDA::cudart CUDA::cublas m)

  # Host-compiled telemetry sources include <cuda_runtime.h>. Without the
  # toolkit include path gcc falls back to the apt package under /usr/include,
  # whose older header aliases cudaGetDeviceProperties to a _v2 symbol that the
  # installed libcudart does not export.
  target_include_directories(train_blt_d PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(infer_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ============================================================================
# CUSTOM TARGETS (parity data generation)
# ============================================================================
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/data/tests")
  add_custom_target(parity-data
    COMMAND python3 tests/py/parity_generate.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating golden test data..."
  )
  add_dependencies(test_main parity-data)
endif()

# ============================================================================
# CUSTOM TARGETS (build + run from project root)
# ============================================================================

# -- test: run full test suite -----------------------------------------------
add_custom_target(test
  COMMAND $<TARGET_FILE:test_main>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS test_main
  COMMENT "Running test suite..."
)

# -- sandbox: build + run scratch playground ---------------------------------
add_custom_target(profile-train
  COMMAND ${CMAKE_COMMAND} -E echo "usage: scripts/profile_train.sh build-cuda --corpus ..."
  COMMENT "Profiling wrapper")
add_custom_target(sandbox_run
  COMMAND $<TARGET_FILE:sandbox>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS sandbox
  COMMENT "Running sandbox..."
)

# -- bench-infer: build + run inference benchmark ----------------------------
add_custom_target(bench-infer
  COMMAND $<TARGET_FILE:infer_bench>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS infer_bench
  COMMENT "Running inference benchmark..."
)

# -- bench-cuda: build + run CUDA benchmark (CUDA only) ---------------------
if(USE_CUDA)
  add_custom_target(bench-cuda
    COMMAND $<TARGET_FILE:cuda_bench>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS cuda_bench
    COMMENT "Running CUDA benchmark..."
  )
endif()

# -- bench-harness: build + run harness self-test ----------------------------
add_custom_target(bench-harness
  COMMAND $<TARGET_FILE:bench_harness_selftest> bench/dummy_results.jsonl
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS bench_harness_selftest
  COMMENT "Running harness self-test..."
)

# -- cuda-smoke: build + run device sanity check (CUDA only) ----------------
if(USE_CUDA)
  add_custom_target(cuda-smoke
    COMMAND $<TARGET_FILE:cuda_smoke>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS cuda_smoke
    COMMENT "Running CUDA smoke test..."
  )
endif()

# -- e2e-dv: build + run end-to-end driver ----------------------------------
add_custom_target(e2e-dv
  COMMAND $<TARGET_FILE:e2e_blt_dv>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS e2e_blt_dv
  COMMENT "Running end-to-end driver..."
)

# -- cuda-sanitize: run tests under compute-sanitizer (Linux only) ----------
if(USE_CUDA AND NOT WIN32)
  find_program(COMPUTE_SANITIZER compute-sanitizer
    HINTS /usr/local/cuda/bin
  )
  if(COMPUTE_SANITIZER)
    add_custom_target(cuda-sanitize
      COMMAND ${COMPUTE_SANITIZER} --target-processes all --tool memcheck
              $<TARGET_FILE:test_main>
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      DEPENDS test_main
      COMMENT "Running CUDA sanitizer..."
    )
  endif()
endif()

# ============================================================================
# CLEAN TARGET
# ============================================================================
# CMake's built-in: cmake --build . --target clean (removes object files)
# Full clean: rm -rf build/ (from project root)
# This matches the old Makefile's `make clean` behavior.
