#ifndef BLT_TRAIN_DIAG_H
#define BLT_TRAIN_DIAG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "core/tensor.h"
#include "models/block_diffusion.h"
#include "models/model.h"
#include "models/param_visitor.h"
#include "models/patcher.h"

// Compute the global L2 gradient norm in visitor order without modifying the
// gradients. When the norm exceeds max_norm, *scale_out receives
// max_norm / norm so the optimizer can fold the scale into its update; it is
// 1.0 otherwise. scale_out may be NULL when only the norm is needed.
float blt_grad_clip_norm(const blt_model *m, blt_model_grad *g, float max_norm, float *scale_out);

// Log per-component L2 gradient norms to the component-norm-log file.
void log_component_norms(FILE *fp, size_t step, blt_model *m, blt_model_grad *g);

// Log forward activation stats (before backward). NULL tensors are skipped.
void log_forward_activation_dump(FILE *fp, size_t step, const blt_tensor *P, const blt_tensor *h, const blt_tensor *O,
                                 const blt_tensor *logits);

// Log gradient activation dump for numerical anomaly detection.
void log_gradient_activation_dump(FILE *fp, size_t step, blt_model *m, blt_model_grad *g);

// Log per-step batch properties: window identity, patch stats, diffusion
// corruption stats, and the effective loss weight.
void log_batch_properties(FILE *fp, size_t step, size_t window_offset, const uint8_t *text, size_t window_size,
                          const blt_patch_info *patches, size_t num_patches, const blt_block_batch *batch);

// Visitor callback: zero out RMSNorm weights. Stays as declaration for use in train_blt_d.c.
void zero_norm_fn(float *p, const blt_param_info *info, void *ctx);

// Step-rate and ETA bookkeeping.
//
// The estimate state is owned here so it cannot be initialised incorrectly at
// the call site: the loop starts at start_step after a resume, so counting from
// zero divides one report interval's elapsed time by (start_step + interval) and
// yields a wildly optimistic rate that the exponential average then carries
// forward, making the ETA climb instead of fall.
typedef struct {
    double ema_sec_per_step;
    size_t last_step;
} blt_eta_state;

void blt_eta_init(blt_eta_state *st, size_t start_step);

// Advance the estimate and return the estimated seconds remaining.
// `elapsed_s` is the time since the previous report, not a running clock.
double blt_eta_update(blt_eta_state *st, size_t step, size_t end_step, double elapsed_s, double excluded_s);

// Create the parent directory of `path` (mkdir -p semantics) and verify it is
// writable. A missing save directory used to surface as a fatal at the first
// periodic checkpoint, which on a 10k-step run throws away all the work so far.
// Returns 0 on success, -1 if the directory is missing and cannot be created.
int train_ensure_parent_dir(const char *path);

// Deterministic training-window order. Each epoch walks every window exactly
// once, but in a different order, so resuming mid-run does not replay the same
// byte sequence it already trained on.
size_t train_window_index(size_t step, size_t num_windows, uint64_t seed);

#ifdef __cplusplus
}
#endif
#endif // BLT_TRAIN_DIAG_H
