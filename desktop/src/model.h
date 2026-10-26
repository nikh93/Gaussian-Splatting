/*
 * model.h - host side gaussian model, initialisation and export.
 */
#ifndef GS_MODEL_H
#define GS_MODEL_H

#include <math.h>

#include "dataset.h"

#define GS_SH_COEFFS 16
#define GS_SH_STRIDE (GS_SH_COEFFS * 3)

typedef struct {
    int n;
    int cap;
    float *means;      /* 3n */
    float *log_scales; /* 3n */
    float *rots;       /* 4n (w,x,y,z) */
    float *opac;       /* n, logits */
    float *sh;         /* 48n */
} gs_model;

void gs_model_alloc(gs_model *m, int n);
void gs_model_free(gs_model *m);

/* Builds the initial gaussians from the dataset point cloud (or random points). */
int gs_model_init_from_dataset(gs_model *m, const gs_dataset *ds, unsigned seed);

/* Standard 3DGS PLY (compatible with SuperSplat, the reference viewers, ...). */
int gs_model_write_ply(const gs_model *m, int sh_degree, const char *path);
/* Compact 32 byte/splat ".splat" format (antimatter15 web viewer). */
int gs_model_write_splat(const gs_model *m, const char *path);

static inline float gs_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

#endif
