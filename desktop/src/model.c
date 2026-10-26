/*
 * model.c - host side gaussian model: initialisation and file export.
 */
#include "model.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SH_C0 0.28209479177387814f

void gs_model_alloc(gs_model *m, int n)
{
    int cap = n > 0 ? n : 1;
    memset(m, 0, sizeof(*m));
    m->n = n;
    m->cap = cap;
    m->means = (float *)calloc((size_t)cap * 3, sizeof(float));
    m->log_scales = (float *)calloc((size_t)cap * 3, sizeof(float));
    m->rots = (float *)calloc((size_t)cap * 4, sizeof(float));
    m->opac = (float *)calloc((size_t)cap, sizeof(float));
    m->sh = (float *)calloc((size_t)cap * GS_SH_STRIDE, sizeof(float));
}

void gs_model_free(gs_model *m)
{
    free(m->means);
    free(m->log_scales);
    free(m->rots);
    free(m->opac);
    free(m->sh);
    memset(m, 0, sizeof(*m));
}

/* xorshift32 + Box-Muller helpers */
static float rnd_uniform(unsigned *s)
{
    unsigned x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return (float)((x >> 8) & 0xFFFFFF) / 16777216.0f;
}

/* ---------------------------------------------------------------------- */
/* mean squared distance to the 3 nearest neighbours (x-sorted sweep)     */
/* ---------------------------------------------------------------------- */

static const float *g_sort_pts;
static int cmp_x(const void *a, const void *b)
{
    float xa = g_sort_pts[*(const int *)a * 3], xb = g_sort_pts[*(const int *)b * 3];
    return xa < xb ? -1 : (xa > xb ? 1 : 0);
}

static void knn3_mean_dist2(const float *pts, int n, float *out)
{
    int *order = (int *)malloc(sizeof(int) * (size_t)n);
    int i;
    for (i = 0; i < n; i++)
        order[i] = i;
    g_sort_pts = pts;
    qsort(order, (size_t)n, sizeof(int), cmp_x);
    for (i = 0; i < n; i++) {
        const float *p = pts + order[i] * 3;
        float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        int dir;
        for (dir = -1; dir <= 1; dir += 2) {
            int j;
            for (j = i + dir; j >= 0 && j < n; j += dir) {
                const float *q = pts + order[j] * 3;
                float dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
                float d2;
                if (dx * dx >= best[2])
                    break;
                d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best[2]) {
                    if (d2 < best[0]) {
                        best[2] = best[1];
                        best[1] = best[0];
                        best[0] = d2;
                    } else if (d2 < best[1]) {
                        best[2] = best[1];
                        best[1] = d2;
                    } else {
                        best[2] = d2;
                    }
                }
            }
        }
        {
            float s = 0;
            int k, cnt = 0;
            for (k = 0; k < 3; k++)
                if (best[k] < FLT_MAX) {
                    s += best[k];
                    cnt++;
                }
            out[order[i]] = cnt ? s / cnt : 0.01f;
        }
    }
    free(order);
}

/* average colour of a point over the images it projects into */
static int sample_color(const gs_dataset *ds, const float *p, float *rgb)
{
    int step = ds->num_cams > 16 ? ds->num_cams / 16 : 1;
    int j, hits = 0;
    float acc[3] = {0, 0, 0};
    for (j = 0; j < ds->num_cams; j += step) {
        const gs_camera *c = &ds->cams[j];
        float x = c->R[0] * p[0] + c->R[1] * p[1] + c->R[2] * p[2] + c->t[0];
        float y = c->R[3] * p[0] + c->R[4] * p[1] + c->R[5] * p[2] + c->t[1];
        float z = c->R[6] * p[0] + c->R[7] * p[1] + c->R[8] * p[2] + c->t[2];
        int u, v;
        const unsigned char *px;
        if (z <= 0.05f)
            continue;
        u = (int)(c->fx * x / z + c->cx + 0.5f);
        v = (int)(c->fy * y / z + c->cy + 0.5f);
        if (u < 0 || v < 0 || u >= c->width || v >= c->height)
            continue;
        px = c->rgba + ((size_t)v * c->width + u) * 4;
        acc[0] += px[0] / 255.0f;
        acc[1] += px[1] / 255.0f;
        acc[2] += px[2] / 255.0f;
        hits++;
    }
    if (!hits)
        return 0;
    rgb[0] = acc[0] / hits;
    rgb[1] = acc[1] / hits;
    rgb[2] = acc[2] / hits;
    return 1;
}

/* point closest (least squares) to all optical axes - what the cameras look at */
static void estimate_focus(const gs_dataset *ds, float out[3])
{
    double A[9] = {0}, b[3] = {0}, det;
    int j, r, k;
    for (j = 0; j < ds->num_cams; j++) {
        const gs_camera *c = &ds->cams[j];
        double d[3] = {c->R[6], c->R[7], c->R[8]}; /* camera z axis in world */
        for (r = 0; r < 3; r++)
            for (k = 0; k < 3; k++) {
                double v = (r == k ? 1.0 : 0.0) - d[r] * d[k];
                A[r * 3 + k] += v;
                b[r] += v * c->campos[k];
            }
    }
    det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
          A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (fabs(det) < 1e-9) {
        memcpy(out, ds->center, sizeof(float) * 3);
        return;
    }
    {
        double inv[9];
        inv[0] = (A[4] * A[8] - A[5] * A[7]) / det;
        inv[1] = (A[2] * A[7] - A[1] * A[8]) / det;
        inv[2] = (A[1] * A[5] - A[2] * A[4]) / det;
        inv[3] = (A[5] * A[6] - A[3] * A[8]) / det;
        inv[4] = (A[0] * A[8] - A[2] * A[6]) / det;
        inv[5] = (A[2] * A[3] - A[0] * A[5]) / det;
        inv[6] = (A[3] * A[7] - A[4] * A[6]) / det;
        inv[7] = (A[1] * A[6] - A[0] * A[7]) / det;
        inv[8] = (A[0] * A[4] - A[1] * A[3]) / det;
        for (r = 0; r < 3; r++)
            out[r] = (float)(inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]);
    }
}

int gs_model_init_from_dataset(gs_model *m, const gs_dataset *ds, unsigned seed)
{
    const int max_points = 300000;
    int n, i;
    float *pts, *cols, *d2;
    unsigned rng = seed ? seed : 0x1234567u;

    if (ds->num_points >= 100) {
        int step = ds->num_points > max_points ? (ds->num_points + max_points - 1) / max_points : 1;
        n = (ds->num_points + step - 1) / step;
        pts = (float *)malloc(sizeof(float) * 3 * (size_t)n);
        cols = (float *)malloc(sizeof(float) * 3 * (size_t)n);
        for (i = 0; i < n; i++) {
            int src = i * step;
            memcpy(pts + i * 3, ds->points + src * 3, sizeof(float) * 3);
            if (ds->has_colors)
                memcpy(cols + i * 3, ds->colors + src * 3, sizeof(float) * 3);
            else if (!sample_color(ds, pts + i * 3, cols + i * 3))
                cols[i * 3] = cols[i * 3 + 1] = cols[i * 3 + 2] = 0.5f;
        }
        printf("[init] %d points from the capture point cloud\n", n);
    } else {
        float focus[3];
        float radius = ds->extent * 0.75f;
        estimate_focus(ds, focus);
        n = 20000;
        pts = (float *)malloc(sizeof(float) * 3 * (size_t)n);
        cols = (float *)malloc(sizeof(float) * 3 * (size_t)n);
        for (i = 0; i < n; i++) {
            float v[3], l;
            do {
                v[0] = rnd_uniform(&rng) * 2 - 1;
                v[1] = rnd_uniform(&rng) * 2 - 1;
                v[2] = rnd_uniform(&rng) * 2 - 1;
                l = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            } while (l > 1.0f);
            pts[i * 3] = focus[0] + v[0] * radius;
            pts[i * 3 + 1] = focus[1] + v[1] * radius;
            pts[i * 3 + 2] = focus[2] + v[2] * radius;
            if (!sample_color(ds, pts + i * 3, cols + i * 3))
                cols[i * 3] = cols[i * 3 + 1] = cols[i * 3 + 2] = 0.5f;
        }
        printf("[init] no point cloud - %d random points around (%.2f %.2f %.2f) r=%.2f\n", n, focus[0],
               focus[1], focus[2], radius);
    }

    d2 = (float *)malloc(sizeof(float) * (size_t)n);
    knn3_mean_dist2(pts, n, d2);

    gs_model_alloc(m, n);
    for (i = 0; i < n; i++) {
        float s = logf(sqrtf(d2[i] > 1e-7f ? d2[i] : 1e-7f));
        int c;
        memcpy(m->means + i * 3, pts + i * 3, sizeof(float) * 3);
        m->log_scales[i * 3] = m->log_scales[i * 3 + 1] = m->log_scales[i * 3 + 2] = s;
        m->rots[i * 4] = 1.0f;
        m->opac[i] = logf(0.1f / 0.9f);
        for (c = 0; c < 3; c++)
            m->sh[(size_t)i * GS_SH_STRIDE + c] = (cols[i * 3 + c] - 0.5f) / SH_C0;
    }
    free(pts);
    free(cols);
    free(d2);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* export                                                                  */
/* ---------------------------------------------------------------------- */

int gs_model_write_ply(const gs_model *m, int sh_degree, const char *path)
{
    char tmp[1100];
    FILE *f;
    int i, k, c;
    int ncoef = (sh_degree + 1) * (sh_degree + 1);
    int nrest = (ncoef - 1) * 3;
    float *rec;
    int nfloat = 3 + 3 + 3 + nrest + 1 + 3 + 4;

    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f)
        return -1;
    fprintf(f, "ply\nformat binary_little_endian 1.0\nelement vertex %d\n", m->n);
    fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    fprintf(f, "property float nx\nproperty float ny\nproperty float nz\n");
    for (k = 0; k < 3; k++)
        fprintf(f, "property float f_dc_%d\n", k);
    for (k = 0; k < nrest; k++)
        fprintf(f, "property float f_rest_%d\n", k);
    fprintf(f, "property float opacity\n");
    for (k = 0; k < 3; k++)
        fprintf(f, "property float scale_%d\n", k);
    for (k = 0; k < 4; k++)
        fprintf(f, "property float rot_%d\n", k);
    fprintf(f, "end_header\n");

    rec = (float *)malloc(sizeof(float) * (size_t)nfloat);
    for (i = 0; i < m->n; i++) {
        const float *sh = m->sh + (size_t)i * GS_SH_STRIDE;
        const float *q = m->rots + i * 4;
        float ql = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        int o = 0;
        if (ql < 1e-12f)
            ql = 1;
        for (k = 0; k < 3; k++)
            rec[o++] = m->means[i * 3 + k];
        rec[o++] = 0;
        rec[o++] = 0;
        rec[o++] = 0;
        for (c = 0; c < 3; c++)
            rec[o++] = sh[c];
        /* reference layout: channel major, f_rest_{c*(ncoef-1) + (k-1)} */
        for (c = 0; c < 3; c++)
            for (k = 1; k < ncoef; k++)
                rec[o++] = sh[k * 3 + c];
        rec[o++] = m->opac[i];
        for (k = 0; k < 3; k++)
            rec[o++] = m->log_scales[i * 3 + k];
        for (k = 0; k < 4; k++)
            rec[o++] = q[k] / ql;
        fwrite(rec, sizeof(float), (size_t)nfloat, f);
    }
    free(rec);
    fclose(f);
    remove(path);
    return rename(tmp, path);
}

static const float *g_sort_key;
static int cmp_key_desc(const void *a, const void *b)
{
    float ka = g_sort_key[*(const int *)a], kb = g_sort_key[*(const int *)b];
    return ka > kb ? -1 : (ka < kb ? 1 : 0);
}

int gs_model_write_splat(const gs_model *m, const char *path)
{
    char tmp[1100];
    FILE *f;
    int i, k;
    int *order = (int *)malloc(sizeof(int) * (size_t)(m->n ? m->n : 1));
    float *key = (float *)malloc(sizeof(float) * (size_t)(m->n ? m->n : 1));
    for (i = 0; i < m->n; i++) {
        order[i] = i;
        key[i] = expf(m->log_scales[i * 3] + m->log_scales[i * 3 + 1] + m->log_scales[i * 3 + 2]) *
                 gs_sigmoid(m->opac[i]);
    }
    g_sort_key = key;
    qsort(order, (size_t)m->n, sizeof(int), cmp_key_desc);

    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f) {
        free(order);
        free(key);
        return -1;
    }
    for (i = 0; i < m->n; i++) {
        int g = order[i];
        float pos[3], sc[3];
        unsigned char rgba[4], rot[4];
        const float *q = m->rots + g * 4;
        float ql = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (ql < 1e-12f)
            ql = 1;
        for (k = 0; k < 3; k++) {
            float col = 0.5f + SH_C0 * m->sh[(size_t)g * GS_SH_STRIDE + k];
            pos[k] = m->means[g * 3 + k];
            sc[k] = expf(m->log_scales[g * 3 + k]);
            col = col < 0 ? 0 : (col > 1 ? 1 : col);
            rgba[k] = (unsigned char)(col * 255.0f + 0.5f);
        }
        rgba[3] = (unsigned char)(gs_sigmoid(m->opac[g]) * 255.0f + 0.5f);
        for (k = 0; k < 4; k++) {
            float v = q[k] / ql * 128.0f + 128.0f;
            rot[k] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        fwrite(pos, sizeof(float), 3, f);
        fwrite(sc, sizeof(float), 3, f);
        fwrite(rgba, 1, 4, f);
        fwrite(rot, 1, 4, f);
    }
    fclose(f);
    free(order);
    free(key);
    remove(path);
    return rename(tmp, path);
}
