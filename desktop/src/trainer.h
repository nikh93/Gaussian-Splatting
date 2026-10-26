/*
 * trainer.h - GPU (CUDA / OpenCL) gaussian splatting optimisation.
 */
#ifndef GS_TRAINER_H
#define GS_TRAINER_H

#include <stddef.h>

#include "gpu.h"
#include "dataset.h"
#include "model.h"

typedef struct {
    int iterations;
    int sh_degree;            /* max SH degree (0..3) */
    int sh_increase_interval; /* raise active SH degree every N iterations */
    float lambda_ssim;
    int densify_from;
    int densify_until;
    int densify_interval;
    int opacity_reset_interval;
    float densify_grad_threshold;
    float percent_dense;
    int max_gaussians;
    int max_dim;              /* longest image side used for training */
    int white_background;
    int snapshot_interval;    /* write intermediate model every N iterations (0 = off) */
    float lr_pos_init, lr_pos_final, lr_sh, lr_opacity, lr_scale, lr_rot;
    unsigned seed;
    const char *preview_dir;  /* if set: write side-by-side renderings (PPM) of some views after training */
} gs_train_params;

typedef struct {
    volatile int cancel; /* set to 1 from another thread to stop */
    /* called from the training thread */
    void (*progress)(void *user, int iter, int total, float loss, float psnr, int num_gaussians);
    void (*snapshot)(void *user, int iter); /* called after an intermediate model was written */
    void *user;
} gs_train_control;

void gs_train_params_default(gs_train_params *p);

/* Trains on `ds` and writes <out_dir>/model.ply and model.splat.
 * Returns 0 on success, 1 if cancelled, -1 on error. */
int gs_train(gs_gpu *gpu, const gs_dataset *ds, const gs_train_params *p, const char *out_dir,
             gs_train_control *ctl, char *err, size_t err_len);

/* Finite difference check of the GPU gradients. Returns 0 if it passed. */
int gs_selftest(gs_gpu *gpu);

#endif
