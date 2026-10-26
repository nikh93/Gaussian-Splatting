/*
 * dataset.h - capture loading (nerfstudio style transforms.json + images + points).
 */
#ifndef GS_DATASET_H
#define GS_DATASET_H

#include <stddef.h>

typedef struct {
    char name[256];
    int width, height;
    float fx, fy, cx, cy;
    float R[9];       /* world -> camera rotation (row major), OpenCV convention */
    float t[3];       /* world -> camera translation */
    float campos[3];  /* camera centre in world space */
    unsigned char *rgba; /* width*height*4 */
} gs_camera;

typedef struct {
    gs_camera *cams;
    int num_cams;
    float *points;          /* num_points * 3 */
    float *colors;          /* num_points * 3 in [0,1] */
    int num_points;
    int has_colors;
    float center[3];
    float extent;           /* radius of the camera rig, used to scale learning rates */
} gs_dataset;

int gs_dataset_load(gs_dataset *ds, const char *dir, int max_dim, char *err, size_t err_len);
void gs_dataset_free(gs_dataset *ds);

/* Loads x,y,z (+ optional red,green,blue) vertices from an ASCII or binary PLY. */
int gs_load_point_ply(const char *path, float **pts, float **cols, int *count, int *has_colors);

#endif
