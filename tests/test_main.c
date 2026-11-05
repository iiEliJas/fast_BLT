#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "core/allocator.h"
#include "core/backend.h"

#include "test_helpers.h"
#include "test_suite.h"

const char *g_test_data_dir = "data/tests";

typedef struct {
    const char *name;
    int (*fn)(void);
} test_case;

int main(void) {
    const test_case core_tests[] = {
        {"tensor core helpers", run_tensor_core_tests},
        {"json parser", run_json_parser_tests},
        {"allocator core helpers", run_allocator_core_tests},
        {"backend dispatch", run_backend_core_tests},
        {"cuda arena and transfers", run_cuda_arena_transfer_tests},

        {"elementwise backend", run_elementwise_backend_tests},
        {"gelu backend", run_gelu_backend_tests},
        {"matmul backend", run_matmul_backend_tests},
        {"matmul backward backend", run_matmul_backward_backend_tests},
        {"softmax backend", run_softmax_backend_tests},
        {"cross entropy backend", run_cross_entropy_tests},
        {"rope backend", run_rope_backend_test},
        {"optimizer backend", run_optim_backend_tests},
        {"mask builder backend", run_mask_builder_backend_tests},
        {"bf16 cast", run_bf16_cast_tests},
        {"deterministic regression", run_deterministic_regression_test},
        {"topk from logits", run_topk_from_logits_tests},
        {"train window order", run_train_window_order_tests},
        {"train diagnostics", run_train_diag_tests},
        {"lr schedule", run_lr_schedule_tests},
        {"decode select sampling", run_decode_select_tests},
        {"patch capacity consistency", run_patch_capacity_consistency_tests},
        {"device info table", run_device_info_tests},
    };

    const test_case model_tests[] = {
        {"entropy", run_entropy_model_tests},
        {"patcher", run_patcher_model_tests},
        {"byte embedding", run_byte_embedding_model_tests},
        {"attention", run_attention_model_tests},
        {"attention masked", run_attention_masked_model_tests},
        {"attention backward", run_attention_backward_model_tests},
        {"cross attention", run_cross_attention_model_tests},
        {"cross attention masked", run_cross_attention_masked_model_tests},
        {"cross attention backward", run_cross_attention_backward_model_tests},
        {"entropy LM", run_elm_model_tests},
        {"hash ngram", run_hash_ngram_model_tests},
        {"local encoder mask", run_lencoder_mask_model_test},
        {"local encoder smoke", run_local_encoder_smoke_test},
        {"local encoder k-split", run_local_encoder_k_split},
        {"global transformer causal mask", run_global_transformer_causal_mask},
        {"global transformer doc boundary", run_global_transformer_doc_boundary},
        {"local decoder cross mask", run_local_decoder_cross_mask},
        {"local decoder paper rule", run_local_decoder_paper_rule},
        {"local decoder k-split", run_local_decoder_k_split},
        {"blt model generate sanity", run_blt_model_generate_sanity},
        {"model stage split", run_model_stage_split_equivalence},
        {"model multi-doc isolation", run_model_multi_doc_isolation},
        {"model decode nullable loss", run_model_decode_nullable_loss},
        {"decoder d0 modes", run_decoder_d0_modes},
        {"rope position gather", run_rope_position_gather_test},
        {"infer stats", run_infer_stats_test},
        {"verify draft directed", run_verify_draft_directed},
        {"selfspec structural random", run_selfspec_structural_random},
        {"selfspec equivalence trained", run_selfspec_equivalence_trained},
        {"kv cache logits equal dense", run_kv_cache_logits_equal_dense},
        {"kv cache rollback commit", run_kv_cache_rollback_commit},
        {"blockgen select kernels", run_blockgen_select_kernels},
        {"blockgen draft behavior", run_blockgen_draft_behavior},
        {"blockgen generation gates", run_blockgen_generation_gates},
        {"blockgen stage6 helpers", run_blockgen_stage6_helpers},
        {"block diffusion mask fixture", run_block_diffusion_mask_fixture},
        {"block diffusion mask has no leak", run_block_diffusion_mask_no_leak},
        {"block diffusion gradcheck", run_block_diffusion_gradcheck},
        {"block diffusion overfit gate", run_block_diffusion_overfit_gate},
        {"block diffusion has no label leak", run_block_diffusion_no_label_leak},
        {"block diffusion last-row upweight", run_block_diffusion_last_row_upweight},
        {"block diffusion draft trailing closed", run_block_diffusion_draft_trailing_closed},
    };

    const test_case parity_tests[] = {
        {"matmul parity", run_matmul_parity_test},
        {"softmax parity", run_softmax_parity_test},
        {"cuda elementwise parity", run_cuda_parity_elementwise},
        {"cuda reductions parity", run_cuda_parity_reductions},
        {"cuda matmul parity", run_cuda_parity_matmul},
        {"cuda composites parity", run_cuda_parity_composites},
        {"cuda training-step parity", run_cuda_parity_training_step},
        {"cuda clip-norm parity", run_cuda_parity_clip_norm},
        {"patcher parity", run_patcher_parity_tests},
        {"transformer parity", run_transformer_parity_tests},
        {"entropy LM overfit", run_elm_overfit_tests},
        {"local encoder overfit", run_lencoder_overfit_test},
        {"local encoder parity", run_local_encoder_parity},
        {"global transformer parity", run_global_transformer_parity},
        {"global transformer overfit", run_global_transformer_overfit},
        {"local decoder parity", run_local_decoder_parity},
        {"local decoder overfit", run_local_decoder_overfit},
        {"blt model overfit", run_blt_model_overfit},
    };

    int do_test[4] = {1, 1, 1}; // core, model, parity, tools
    size_t test_count = 0;
    int passed = 0;

    printf("\n------------------------------------------\n");
    printf("            Running FBLT tests...\n");
    printf("------------------------------------------\n");

    if (do_test[0]) {
        printf("\n------------------------\n");
        printf("Running core tests...\n\n");
        test_count += sizeof(core_tests) / sizeof(core_tests[0]);
        for (size_t i = 0; i < sizeof(core_tests) / sizeof(core_tests[0]); ++i) {
            if (core_tests[i].fn()) {
                printf("[PASS] %s\n", core_tests[i].name);
                passed++;
            } else {
                printf("[FAIL] %s\n", core_tests[i].name);
            }
        }
    }

    if (do_test[1]) {
        printf("\n------------------------\n");
        printf("Running model tests...\n\n");
        test_count += sizeof(model_tests) / sizeof(model_tests[0]);
        for (size_t i = 0; i < sizeof(model_tests) / sizeof(model_tests[0]); ++i) {
            if (model_tests[i].fn()) {
                printf("[PASS] %s\n", model_tests[i].name);
                passed++;
            } else {
                printf("[FAIL] %s\n", model_tests[i].name);
            }
        }
    }

    if (do_test[2]) {
        printf("\n------------------------\n");
        printf("Running parity tests...\n\n");
        test_count += sizeof(parity_tests) / sizeof(parity_tests[0]);
        for (size_t i = 0; i < sizeof(parity_tests) / sizeof(parity_tests[0]); ++i) {
            if (parity_tests[i].fn()) {
                printf("[PASS] %s\n", parity_tests[i].name);
                passed++;
            } else {
                printf("[FAIL] %s\n", parity_tests[i].name);
            }
        }
    }

    printf("\n------------------------------------------\n");
    printf("Summary: %d/%zu tests passed\n", passed, test_count);
    printf("------------------------------------------\n");

    return passed == (int)test_count ? 0 : 1;
}
