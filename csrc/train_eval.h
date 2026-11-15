#ifndef BLT_TRAIN_EVAL_H
#define BLT_TRAIN_EVAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "core/allocator.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/patcher.h"
#include "models/entropy_lm.h"
#include "models/block_diffusion.h"

// Per-window metrics from one decoder pass.
typedef struct {
    double causal_ce;    // mean next-byte CE (nats/byte) over clean rows [0, N-2]
    double masked_ce;    // mean CE over masked block cells; 0 when none
    size_t masked_hits;  // argmax hits over masked cells
    size_t masked_cells; // masked cell count
} blt_window_metrics;

// Fills out with the clean-row causal CE plus, when the diffusion batch is
// supplied, the masked-cell CE/accuracy of that same pass. Requires at least
// one clean-row prediction (N >= 2).
void window_metrics(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                    const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m, const blt_patch_info *patches,
                    size_t M, blt_window_metrics *out);

// Mean next-byte CE (nats/byte) over the clean rows of one window, computed
// from the decoder logits. For BOTH training modes: with the Figure-5 plain-
// causal TRAIN mask, clean-row logits never depend on block rows, so this is
// the true causal BPB of the model in either case.
double window_causal_ce(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                        const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m,
                        const blt_patch_info *patches, size_t M, size_t *masked_hits, size_t *masked_total);

// Metrics for one aligned-novel block.
typedef struct {
    double ce;    // mean CE (nats/byte) over the novel block's masked cells
    size_t hits;  // argmax hits
    size_t cells; // number of cells scored
} blt_novel_metrics;

// One aligned-novel block at the first interior patch start P with P >= 8 and
// P + B <= N: prefix [0,P) re-segmented on its own, one fully masked block
// P..P+B-1, cross-attention to the last closed latent. Returns 1 if a block was
// scored, 0 if the window has no suitable prefix start.
int window_novel_masked(blt_arena *scratch, blt_arena *seg_arena, const blt_model *model, blt_entropy_lm *lm,
                        const uint8_t *text, size_t N, const blt_patch_info *patches, size_t M, size_t B,
                        int use_entropy, blt_d0_mode d0m, blt_novel_metrics *out);

// Fixed-stride patching helper.
size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info *out);

// Top-k class indices of one logits row, written to idx_out[0..k-1] sorted by
// descending logit. Exact ties break toward the lower class index. Slots beyond
// the end of the row are set to -1, so callers can test every slot against a
// target class without knowing the vocab size.
void topk_from_logits(const float *row, int n, int k, int *idx_out);

// Entropy-LM patching (same numerics as the inference controllers).
// The patcher is host-only, so under a device backend the byte ids are
// uploaded and the entropies staged back through host memory.
// entropy_segment_ex uses threshold_global for the global entropy rule;
// entropy_segment forwards the standard 3.0f threshold.
size_t entropy_segment_ex(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                          size_t max_patches, float threshold_global);
size_t entropy_segment(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                       size_t max_patches);

#ifdef __cplusplus
}
#endif

#endif // BLT_TRAIN_EVAL_H
