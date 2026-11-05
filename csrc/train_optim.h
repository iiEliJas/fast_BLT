#ifndef BLT_TRAIN_OPTIM_H
#define BLT_TRAIN_OPTIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "models/model.h"
#include "models/param_visitor.h"
#include "ops/optim.h"
#include "core/allocator.h"

#ifndef BLT_MAX_PARAMS
#define BLT_MAX_PARAMS 4096
#endif

// SGD: w -= lr * (g * grad_scale) for every parameter. Folding the clip scale
// into the update removes a separate per-tensor gradient pass and is
// bit-identical to pre-scaling the gradient (same multiply order).
void sgd_all(blt_model *m, blt_model_grad *g, float lr, float grad_scale);

// AdamW state: one exp_avg + one exp_avg_sq per parameter, flat array indexed by param idx
typedef struct {
    blt_tensor em;
    blt_tensor esq;
} adamw_pair;

typedef struct {
    adamw_pair flat[BLT_MAX_PARAMS];
    size_t n;
    size_t step;
} adamw_state;

// Create adamw_state with flat pairs for every model parameter
adamw_state *adamw_state_create(blt_arena *arena, blt_model *m);

// AdamW update step: w -= lr * bias_corrected_adamw(w, g * grad_scale, m, v, ...)
void adamw_all(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg, float grad_scale);

// Compute ||update|| across all parameters (for gradient clipping diagnostics),
// using the same grad_scale the optimizer update will apply.
float adamw_update_norm(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg, float grad_scale);

// Write exp_avg, exp_avg_sq and the bias-correction step counter, plus the
// training step the snapshot was taken at. Fatal on any IO or shape mismatch.
void adamw_state_save(const adamw_state *s, size_t global_step, const char *path);

// Restore from adamw_state_save. Fatal if the pair count or any shape disagrees
// with the target model, which is what a config mismatch looks like.
// global_step_out may be NULL; when set it receives the stored training step.
void adamw_state_load(adamw_state *s, size_t *global_step_out, const char *path);

#ifdef __cplusplus
}
#endif
#endif // BLT_TRAIN_OPTIM_H
