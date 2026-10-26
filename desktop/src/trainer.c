/*
 * trainer.c - GPU gaussian splatting optimisation loop (CUDA or OpenCL, see gpu.h).
 *
 * Per iteration:
 *   1. pick a training view, upload its image
 *   2. preprocess (GPU): project gaussians, evaluate colours
 *   3. binning (CPU): depth sort visible gaussians, build per tile lists
 *   4. render_fwd (GPU), loss + gradient (GPU, L1 + D-SSIM)
 *   5. render_bwd + preprocess_bwd (GPU), Adam step (GPU)
 *   6. every `densify_interval` iterations: clone / split / prune (CPU)
 */
#include "trainer.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

#define TILE 16
#define LOCAL_1D 256
#define NPARAM 5
static const int DIMS[NPARAM] = {3, 3, 4, 1, GS_SH_STRIDE}; /* means, scales, rots, opacity, sh */

enum { P_MEAN = 0, P_SCALE = 1, P_ROT = 2, P_OPAC = 3, P_SH = 4 };

typedef struct {
    gs_gpu *gpu;
    gs_kernel k_pre, k_fwd, k_ssim_h, k_ssim_v, k_ssim_bh, k_loss, k_bwd, k_pre_bwd, k_adam, k_stats;
    int n, cap;
    gs_buf p[NPARAM], m[NPARAM], v[NPARAM], g[NPARAM];
    gs_buf mean2d, depths, radii, conic, colors;
    gs_buf dmean2d, dconic, dopa, dcolor;
    gs_buf accum, denom, maxr;
    /* image sized buffers */
    int max_w, max_h;
    gs_buf img, final_t, ncontrib, dimg, gt, tmp5, efh, ssim_map, tmp3, cam;
    /* tiles */
    gs_buf ranges, plist;
    size_t plist_cap;
    /* host scratch for binning */
    int *h_radii;
    float *h_mean2d, *h_depth;
    unsigned *h_keys, *h_keys2;
    int *h_vals, *h_vals2;
    int *h_rect;
    int *h_tile_count;
    int *h_ranges;
    int *h_plist;
    size_t h_plist_cap;
    int host_cap;
} trainer;

typedef struct {
    int n;
    float *a[3][NPARAM]; /* [param, adam m, adam v][buffer] */
} host_state;

/* GPU call that must succeed, otherwise jump to the local `fail` label */
#define CK(call)                                                                       \
    do {                                                                               \
        if ((call) != 0) {                                                             \
            fprintf(stderr, "[gpu] %s failed (%s:%d)\n", #call, __FILE__, __LINE__); \
            goto fail;                                                                 \
        }                                                                              \
    } while (0)

/* kernel argument builders */
#define BUF(b) gs_arg_buf(tr->gpu, &(b))
#define VAL(v) val_arg(&(v), sizeof(v))
#define NARGS(a) ((int)(sizeof(a) / sizeof((a)[0])))

static gs_arg val_arg(const void *p, size_t size)
{
    gs_arg a;
    a.ptr = p;
    a.size = size;
    return a;
}

static int launch1d(trainer *tr, gs_kernel k, size_t n, int nargs, const gs_arg *args)
{
    size_t global[3] = {n, 1, 1}, local[3] = {LOCAL_1D, 1, 1};
    return tr->gpu->ops->launch(tr->gpu, k, 1, global, local, nargs, args);
}

static int launch_image(trainer *tr, gs_kernel k, int W, int H, int depth, int nargs, const gs_arg *args)
{
    size_t global[3], local[3] = {TILE, TILE, 1};
    global[0] = (size_t)W;
    global[1] = (size_t)H;
    global[2] = (size_t)depth;
    return tr->gpu->ops->launch(tr->gpu, k, depth > 1 ? 3 : 2, global, local, nargs, args);
}

#define G (tr->gpu->ops)

void gs_train_params_default(gs_train_params *p)
{
    memset(p, 0, sizeof(*p));
    p->iterations = 30000;
    p->sh_degree = 3;
    p->sh_increase_interval = 1000;
    p->lambda_ssim = 0.2f;
    p->densify_from = 500;
    p->densify_until = 15000;
    p->densify_interval = 100;
    p->opacity_reset_interval = 3000;
    p->densify_grad_threshold = 0.0002f;
    p->percent_dense = 0.01f;
    p->max_gaussians = 3000000;
    p->max_dim = 1600;
    p->white_background = 0;
    p->snapshot_interval = 5000;
    p->lr_pos_init = 0.00016f;
    p->lr_pos_final = 0.0000016f;
    p->lr_sh = 0.0025f;
    p->lr_opacity = 0.025f;
    p->lr_scale = 0.005f;
    p->lr_rot = 0.001f;
    p->seed = 42;
}

static int mkbuf(trainer *tr, gs_buf *b, size_t bytes)
{
    if (G->alloc(tr->gpu, b, bytes) != 0) {
        fprintf(stderr, "[gpu] allocation of %lu bytes failed\n", (unsigned long)bytes);
        return -1;
    }
    return 0;
}

static void rel(trainer *tr, gs_buf *b)
{
    if (b->u.dev || b->u.ptr)
        G->free(tr->gpu, b);
    memset(b, 0, sizeof(*b));
}

static int fill_zero(trainer *tr, gs_buf *b, size_t bytes)
{
    return G->zero(tr->gpu, b, bytes);
}

static void release_gauss(trainer *tr)
{
    int i;
    for (i = 0; i < NPARAM; i++) {
        rel(tr, &tr->p[i]);
        rel(tr, &tr->m[i]);
        rel(tr, &tr->v[i]);
        rel(tr, &tr->g[i]);
    }
    rel(tr, &tr->mean2d);
    rel(tr, &tr->depths);
    rel(tr, &tr->radii);
    rel(tr, &tr->conic);
    rel(tr, &tr->colors);
    rel(tr, &tr->dmean2d);
    rel(tr, &tr->dconic);
    rel(tr, &tr->dopa);
    rel(tr, &tr->dcolor);
    rel(tr, &tr->accum);
    rel(tr, &tr->denom);
    rel(tr, &tr->maxr);
}

/* (Re)allocates the per gaussian GPU buffers and host scratch for `cap` gaussians. */
static int alloc_gauss(trainer *tr, int cap)
{
    size_t c = (size_t)cap;
    int i;
    release_gauss(tr);
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = c * DIMS[i] * sizeof(float);
        if (mkbuf(tr, &tr->p[i], bytes) || mkbuf(tr, &tr->m[i], bytes) || mkbuf(tr, &tr->v[i], bytes) ||
            mkbuf(tr, &tr->g[i], bytes))
            return -1;
    }
    if (mkbuf(tr, &tr->mean2d, c * 8) || mkbuf(tr, &tr->depths, c * 4) || mkbuf(tr, &tr->radii, c * 4) ||
        mkbuf(tr, &tr->conic, c * 16) || mkbuf(tr, &tr->colors, c * 12) || mkbuf(tr, &tr->dmean2d, c * 8) ||
        mkbuf(tr, &tr->dconic, c * 12) || mkbuf(tr, &tr->dopa, c * 4) || mkbuf(tr, &tr->dcolor, c * 12) ||
        mkbuf(tr, &tr->accum, c * 4) || mkbuf(tr, &tr->denom, c * 4) || mkbuf(tr, &tr->maxr, c * 4))
        return -1;
    tr->cap = cap;
    if (cap > tr->host_cap) {
        free(tr->h_radii);
        free(tr->h_mean2d);
        free(tr->h_depth);
        free(tr->h_keys);
        free(tr->h_keys2);
        free(tr->h_vals);
        free(tr->h_vals2);
        free(tr->h_rect);
        tr->h_radii = (int *)malloc(c * sizeof(int));
        tr->h_mean2d = (float *)malloc(c * 2 * sizeof(float));
        tr->h_depth = (float *)malloc(c * sizeof(float));
        tr->h_keys = (unsigned *)malloc(c * sizeof(unsigned));
        tr->h_keys2 = (unsigned *)malloc(c * sizeof(unsigned));
        tr->h_vals = (int *)malloc(c * sizeof(int));
        tr->h_vals2 = (int *)malloc(c * sizeof(int));
        tr->h_rect = (int *)malloc(c * 4 * sizeof(int));
        tr->host_cap = cap;
    }
    return 0;
}

static void state_alloc(host_state *s, int n)
{
    int k, i;
    s->n = n;
    for (k = 0; k < 3; k++)
        for (i = 0; i < NPARAM; i++)
            s->a[k][i] = (float *)calloc((size_t)(n ? n : 1) * DIMS[i], sizeof(float));
}

static void state_free(host_state *s)
{
    int k, i;
    for (k = 0; k < 3; k++)
        for (i = 0; i < NPARAM; i++)
            free(s->a[k][i]);
    memset(s, 0, sizeof(*s));
}

static int state_download(trainer *tr, host_state *s)
{
    int i;
    state_alloc(s, tr->n);
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = (size_t)tr->n * DIMS[i] * sizeof(float);
        if (G->read(tr->gpu, &tr->p[i], 0, bytes, s->a[0][i]) || G->read(tr->gpu, &tr->m[i], 0, bytes, s->a[1][i]) ||
            G->read(tr->gpu, &tr->v[i], 0, bytes, s->a[2][i]))
            return -1;
    }
    return 0;
}

static int state_upload(trainer *tr, const host_state *s)
{
    int i;
    if (s->n > tr->cap) {
        int cap = s->n + s->n / 2 + 1024;
        if (alloc_gauss(tr, cap) != 0)
            return -1;
    }
    tr->n = s->n;
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = (size_t)s->n * DIMS[i] * sizeof(float);
        if (G->write(tr->gpu, &tr->p[i], 0, bytes, s->a[0][i]) || G->write(tr->gpu, &tr->m[i], 0, bytes, s->a[1][i]) ||
            G->write(tr->gpu, &tr->v[i], 0, bytes, s->a[2][i]))
            return -1;
    }
    if (fill_zero(tr, &tr->accum, (size_t)tr->cap * 4) || fill_zero(tr, &tr->denom, (size_t)tr->cap * 4) ||
        fill_zero(tr, &tr->maxr, (size_t)tr->cap * 4))
        return -1;
    return 0;
}

static void state_to_model(const host_state *s, gs_model *m)
{
    gs_model_alloc(m, s->n);
    memcpy(m->means, s->a[0][P_MEAN], sizeof(float) * 3 * (size_t)s->n);
    memcpy(m->log_scales, s->a[0][P_SCALE], sizeof(float) * 3 * (size_t)s->n);
    memcpy(m->rots, s->a[0][P_ROT], sizeof(float) * 4 * (size_t)s->n);
    memcpy(m->opac, s->a[0][P_OPAC], sizeof(float) * (size_t)s->n);
    memcpy(m->sh, s->a[0][P_SH], sizeof(float) * GS_SH_STRIDE * (size_t)s->n);
}

/* copy gaussian `src` of `a` to slot `dst` of `b`; adam moments optionally cleared */
static void state_copy(const host_state *a, int src, host_state *b, int dst, int keep_moments)
{
    int i, k;
    for (k = 0; k < 3; k++)
        for (i = 0; i < NPARAM; i++) {
            float *d = b->a[k][i] + (size_t)dst * DIMS[i];
            if (k == 0 || keep_moments)
                memcpy(d, a->a[k][i] + (size_t)src * DIMS[i], sizeof(float) * DIMS[i]);
            else
                memset(d, 0, sizeof(float) * DIMS[i]);
        }
}

/* ---------------------------------------------------------------------- */
/* binning                                                                 */
/* ---------------------------------------------------------------------- */

static void radix_sort(unsigned *keys, int *vals, unsigned *k2, int *v2, int n)
{
    int pass, i;
    for (pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        int count[257];
        memset(count, 0, sizeof(count));
        for (i = 0; i < n; i++)
            count[((keys[i] >> shift) & 0xFF) + 1]++;
        for (i = 0; i < 256; i++)
            count[i + 1] += count[i];
        for (i = 0; i < n; i++) {
            int b = (keys[i] >> shift) & 0xFF;
            int o = count[b]++;
            k2[o] = keys[i];
            v2[o] = vals[i];
        }
        {
            unsigned *tk = keys;
            int *tv = vals;
            keys = k2;
            vals = v2;
            k2 = tk;
            v2 = tv;
        }
    }
    /* 4 passes -> result is back in the original arrays */
}

static int bin_gaussians(trainer *tr, int W, int H, int *num_visible)
{
    int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE, ntiles = gx * gy;
    int n = tr->n, i, nv = 0;
    size_t total = 0;

    if (n > 0) {
        CK(G->read(tr->gpu, &tr->radii, 0, (size_t)n * 4, tr->h_radii));
        CK(G->read(tr->gpu, &tr->mean2d, 0, (size_t)n * 8, tr->h_mean2d));
        CK(G->read(tr->gpu, &tr->depths, 0, (size_t)n * 4, tr->h_depth));
    }
    for (i = 0; i < n; i++) {
        if (tr->h_radii[i] > 0) {
            unsigned key;
            memcpy(&key, &tr->h_depth[i], 4); /* positive floats sort like their bit patterns */
            tr->h_keys[nv] = key;
            tr->h_vals[nv] = i;
            nv++;
        }
    }
    radix_sort(tr->h_keys, tr->h_vals, tr->h_keys2, tr->h_vals2, nv);

    memset(tr->h_tile_count, 0, sizeof(int) * (size_t)(ntiles + 1));
    for (i = 0; i < nv; i++) {
        int id = tr->h_vals[i];
        float u = tr->h_mean2d[2 * id], v = tr->h_mean2d[2 * id + 1];
        float r = (float)tr->h_radii[id];
        int x0, y0, x1, y1, x, y;
        int *rc = tr->h_rect + 4 * i;
        x0 = (int)((u - r) / TILE); x0 = x0 < 0 ? 0 : (x0 > gx ? gx : x0);
        y0 = (int)((v - r) / TILE); y0 = y0 < 0 ? 0 : (y0 > gy ? gy : y0);
        x1 = (int)((u + r + TILE - 1) / TILE); x1 = x1 < 0 ? 0 : (x1 > gx ? gx : x1);
        y1 = (int)((v + r + TILE - 1) / TILE); y1 = y1 < 0 ? 0 : (y1 > gy ? gy : y1);
        rc[0] = x0; rc[1] = y0; rc[2] = x1; rc[3] = y1;
        for (y = y0; y < y1; y++)
            for (x = x0; x < x1; x++)
                tr->h_tile_count[y * gx + x]++;
        total += (size_t)(x1 - x0) * (y1 - y0);
    }
    {
        int acc = 0;
        for (i = 0; i < ntiles; i++) {
            tr->h_ranges[2 * i] = acc;
            acc += tr->h_tile_count[i];
            tr->h_ranges[2 * i + 1] = acc;
            tr->h_tile_count[i] = tr->h_ranges[2 * i]; /* becomes write cursor */
        }
    }
    if (total > tr->h_plist_cap) {
        free(tr->h_plist);
        tr->h_plist_cap = total + total / 2 + 1024;
        tr->h_plist = (int *)malloc(tr->h_plist_cap * sizeof(int));
    }
    for (i = 0; i < nv; i++) {
        int id = tr->h_vals[i];
        int *rc = tr->h_rect + 4 * i;
        int x, y;
        for (y = rc[1]; y < rc[3]; y++)
            for (x = rc[0]; x < rc[2]; x++)
                tr->h_plist[tr->h_tile_count[y * gx + x]++] = id;
    }
    if (total > tr->plist_cap) {
        rel(tr, &tr->plist);
        tr->plist_cap = total + total / 2 + 1024;
        CK(mkbuf(tr, &tr->plist, tr->plist_cap * sizeof(int)));
    }
    CK(G->write(tr->gpu, &tr->ranges, 0, sizeof(int) * 2 * (size_t)ntiles, tr->h_ranges));
    CK(G->write(tr->gpu, &tr->plist, 0, sizeof(int) * total, tr->h_plist));
    *num_visible = nv;
    return 0;
fail:
    return -1;
}

/* ---------------------------------------------------------------------- */

static int create_kernels(trainer *tr)
{
#define MK(var, name)                                                       \
    tr->var = G->kernel(tr->gpu, name);                                     \
    if (!tr->var) {                                                         \
        fprintf(stderr, "[gpu] kernel %s not found\n", name);               \
        return -1;                                                          \
    }
    MK(k_pre, "preprocess");
    MK(k_fwd, "render_fwd");
    MK(k_ssim_h, "ssim_h");
    MK(k_ssim_v, "ssim_v");
    MK(k_ssim_bh, "ssim_bh");
    MK(k_loss, "loss_grad");
    MK(k_bwd, "render_bwd");
    MK(k_pre_bwd, "preprocess_bwd");
    MK(k_adam, "adam");
    MK(k_stats, "accum_stats");
#undef MK
    return 0;
}

static void trainer_release(trainer *tr)
{
    gs_kernel *ks[] = {&tr->k_pre, &tr->k_fwd, &tr->k_ssim_h, &tr->k_ssim_v, &tr->k_ssim_bh,
                       &tr->k_loss, &tr->k_bwd, &tr->k_pre_bwd, &tr->k_adam, &tr->k_stats};
    size_t i;
    for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
        if (*ks[i])
            G->kernel_release(tr->gpu, *ks[i]);
    release_gauss(tr);
    rel(tr, &tr->img);
    rel(tr, &tr->final_t);
    rel(tr, &tr->ncontrib);
    rel(tr, &tr->dimg);
    rel(tr, &tr->gt);
    rel(tr, &tr->tmp5);
    rel(tr, &tr->efh);
    rel(tr, &tr->ssim_map);
    rel(tr, &tr->tmp3);
    rel(tr, &tr->cam);
    rel(tr, &tr->ranges);
    rel(tr, &tr->plist);
    free(tr->h_radii);
    free(tr->h_mean2d);
    free(tr->h_depth);
    free(tr->h_keys);
    free(tr->h_keys2);
    free(tr->h_vals);
    free(tr->h_vals2);
    free(tr->h_rect);
    free(tr->h_tile_count);
    free(tr->h_ranges);
    free(tr->h_plist);
}

static void fill_cam(const gs_camera *c, float *out)
{
    memset(out, 0, sizeof(float) * 24);
    memcpy(out, c->R, sizeof(float) * 9);
    memcpy(out + 9, c->t, sizeof(float) * 3);
    memcpy(out + 12, c->campos, sizeof(float) * 3);
    out[15] = c->fx;
    out[16] = c->fy;
    out[17] = c->cx;
    out[18] = c->cy;
    out[19] = (float)c->width;
    out[20] = (float)c->height;
}

/* Forward pass for one view: preprocess + binning + render. */
static int forward_ex(trainer *tr, const gs_camera *c, int sh_deg, const float bg[3], int render_only)
{
    float camv[24];
    int N = tr->n, W = c->width, H = c->height, nv;

    fill_cam(c, camv);
    if (!render_only)
        CK(G->write(tr->gpu, &tr->cam, 0, sizeof(camv), camv));

    if (N > 0 && !render_only) {
        gs_arg a[] = {VAL(N),          BUF(tr->p[P_MEAN]), BUF(tr->p[P_SCALE]), BUF(tr->p[P_ROT]),
                      BUF(tr->p[P_OPAC]), BUF(tr->p[P_SH]), VAL(sh_deg),       BUF(tr->cam),
                      BUF(tr->mean2d),  BUF(tr->depths),   BUF(tr->radii),     BUF(tr->conic),
                      BUF(tr->colors)};
        CK(launch1d(tr, tr->k_pre, (size_t)N, NARGS(a), a));
    }
    if (!render_only && bin_gaussians(tr, W, H, &nv) != 0)
        goto fail;
    {
        gs_arg a[] = {BUF(tr->ranges), BUF(tr->plist), VAL(W),          VAL(H),           BUF(tr->mean2d),
                      BUF(tr->conic),  BUF(tr->colors), VAL(bg[0]),     VAL(bg[1]),       VAL(bg[2]),
                      BUF(tr->img),    BUF(tr->final_t), BUF(tr->ncontrib)};
        CK(launch_image(tr, tr->k_fwd, W, H, 1, NARGS(a), a));
    }
    return 0;
fail:
    return -1;
}

static int forward(trainer *tr, const gs_camera *c, int sh_deg, const float bg[3])
{
    return forward_ex(tr, c, sh_deg, bg, 0);
}

static int loss_and_backward(trainer *tr, const gs_camera *c, int sh_deg, const float bg[3], float lambda)
{
    int N = tr->n, W = c->width, H = c->height;
    {
        gs_arg a1[] = {VAL(W), VAL(H), BUF(tr->img), BUF(tr->gt), BUF(tr->tmp5)};
        gs_arg a2[] = {VAL(W), VAL(H), BUF(tr->tmp5), BUF(tr->efh), BUF(tr->ssim_map)};
        gs_arg a3[] = {VAL(W), VAL(H), BUF(tr->efh), BUF(tr->tmp3)};
        gs_arg a4[] = {VAL(W), VAL(H), VAL(lambda), BUF(tr->img), BUF(tr->gt), BUF(tr->tmp3), BUF(tr->dimg)};
        CK(launch_image(tr, tr->k_ssim_h, W, H, 3, NARGS(a1), a1));
        CK(launch_image(tr, tr->k_ssim_v, W, H, 3, NARGS(a2), a2));
        CK(launch_image(tr, tr->k_ssim_bh, W, H, 3, NARGS(a3), a3));
        CK(launch_image(tr, tr->k_loss, W, H, 3, NARGS(a4), a4));
    }
    if (N == 0)
        return 0;

    CK(fill_zero(tr, &tr->dmean2d, (size_t)N * 8));
    CK(fill_zero(tr, &tr->dconic, (size_t)N * 12));
    CK(fill_zero(tr, &tr->dopa, (size_t)N * 4));
    CK(fill_zero(tr, &tr->dcolor, (size_t)N * 12));
    {
        gs_arg a[] = {BUF(tr->ranges),  BUF(tr->plist),   VAL(W),           VAL(H),
                      BUF(tr->mean2d),  BUF(tr->conic),   BUF(tr->colors),  VAL(bg[0]),
                      VAL(bg[1]),       VAL(bg[2]),       BUF(tr->final_t), BUF(tr->ncontrib),
                      BUF(tr->dimg),    BUF(tr->dmean2d), BUF(tr->dconic),  BUF(tr->dopa),
                      BUF(tr->dcolor)};
        CK(launch_image(tr, tr->k_bwd, W, H, 1, NARGS(a), a));
    }
    {
        gs_arg a[] = {VAL(N),             BUF(tr->p[P_MEAN]), BUF(tr->p[P_SCALE]), BUF(tr->p[P_ROT]),
                      BUF(tr->p[P_OPAC]), BUF(tr->p[P_SH]),   VAL(sh_deg),         BUF(tr->cam),
                      BUF(tr->radii),     BUF(tr->dmean2d),   BUF(tr->dconic),     BUF(tr->dopa),
                      BUF(tr->dcolor),    BUF(tr->g[P_MEAN]), BUF(tr->g[P_SCALE]), BUF(tr->g[P_ROT]),
                      BUF(tr->g[P_OPAC]), BUF(tr->g[P_SH])};
        CK(launch1d(tr, tr->k_pre_bwd, (size_t)N, NARGS(a), a));
    }
    return 0;
fail:
    return -1;
}

static int adam_step(trainer *tr, int which, float lr1, float lr2, int group, int first, int step)
{
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-15f;
    float bc1 = 1.f - powf(b1, (float)step), bc2 = 1.f - powf(b2, (float)step);
    int n = tr->n * DIMS[which];
    if (n == 0)
        return 0;
    {
        gs_arg a[] = {VAL(n),     BUF(tr->p[which]), BUF(tr->g[which]), BUF(tr->m[which]), BUF(tr->v[which]),
                      VAL(lr1),   VAL(lr2),          VAL(group),        VAL(first),        VAL(b1),
                      VAL(b2),    VAL(eps),          VAL(bc1),          VAL(bc2)};
        CK(launch1d(tr, tr->k_adam, (size_t)n, NARGS(a), a));
    }
    return 0;
fail:
    return -1;
}

static int accumulate_stats(trainer *tr, int W, int H)
{
    float sx = 0.5f * W, sy = 0.5f * H;
    if (tr->n == 0)
        return 0;
    {
        gs_arg a[] = {VAL(tr->n), BUF(tr->radii), BUF(tr->dmean2d), VAL(sx),
                      VAL(sy),    BUF(tr->accum), BUF(tr->denom),   BUF(tr->maxr)};
        CK(launch1d(tr, tr->k_stats, (size_t)tr->n, NARGS(a), a));
    }
    return 0;
fail:
    return -1;
}

/* ---------------------------------------------------------------------- */
/* densification                                                          */
/* ---------------------------------------------------------------------- */

static float rnd01(unsigned *s)
{
    unsigned x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return ((x >> 8) + 0.5f) / 16777216.0f;
}

static float rnd_normal(unsigned *s)
{
    float u1 = rnd01(s), u2 = rnd01(s);
    return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

static void quat_rot(const float *q4, float *R)
{
    float l = sqrtf(q4[0] * q4[0] + q4[1] * q4[1] + q4[2] * q4[2] + q4[3] * q4[3]);
    float w, x, y, z;
    if (l < 1e-12f)
        l = 1;
    w = q4[0] / l; x = q4[1] / l; y = q4[2] / l; z = q4[3] / l;
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}

static float max_scale(const host_state *s, int i)
{
    const float *ls = s->a[0][P_SCALE] + 3 * (size_t)i;
    float m = ls[0] > ls[1] ? ls[0] : ls[1];
    m = m > ls[2] ? m : ls[2];
    return expf(m);
}

static int densify_and_prune(trainer *tr, const gs_train_params *p, float extent, int prune_big,
                             int allow_growth, unsigned *rng)
{
    host_state s, out;
    float *accum = NULL, *denom = NULL, *maxr = NULL, *out_maxr = NULL;
    unsigned char *mode = NULL; /* 0 keep, 1 clone, 2 split */
    int n = tr->n, i, n_clone = 0, n_split = 0, o, kept;
    int ret = -1;
    float thr_scale = p->percent_dense * extent;

    memset(&s, 0, sizeof(s));
    memset(&out, 0, sizeof(out));
    if (n == 0)
        return 0;
    if (state_download(tr, &s) != 0)
        goto done;
    accum = (float *)malloc(sizeof(float) * (size_t)n);
    denom = (float *)malloc(sizeof(float) * (size_t)n);
    maxr = (float *)malloc(sizeof(float) * (size_t)n);
    mode = (unsigned char *)calloc((size_t)n, 1);
    if (G->read(tr->gpu, &tr->accum, 0, (size_t)n * 4, accum) ||
        G->read(tr->gpu, &tr->denom, 0, (size_t)n * 4, denom) ||
        G->read(tr->gpu, &tr->maxr, 0, (size_t)n * 4, maxr))
        goto done;

    if (allow_growth) {
        for (i = 0; i < n; i++) {
            float g = denom[i] > 0 ? accum[i] / denom[i] : 0.f;
            if (g < p->densify_grad_threshold)
                continue;
            if (max_scale(&s, i) <= thr_scale) {
                mode[i] = 1;
                n_clone++;
            } else {
                mode[i] = 2;
                n_split++;
            }
        }
        if (n + n_clone + n_split > p->max_gaussians) {
            /* budget exhausted: no growth this round */
            memset(mode, 0, (size_t)n);
            n_clone = n_split = 0;
        }
    }

    state_alloc(&out, n + n_clone + 2 * n_split);
    out_maxr = (float *)calloc((size_t)out.n, sizeof(float));
    o = 0;
    for (i = 0; i < n; i++) {
        if (mode[i] == 2)
            continue; /* replaced by its two children */
        state_copy(&s, i, &out, o, 1);
        out_maxr[o] = maxr[i];
        o++;
    }
    for (i = 0; i < n; i++)
        if (mode[i] == 1) {
            state_copy(&s, i, &out, o, 0);
            o++;
        }
    for (i = 0; i < n; i++) {
        int c;
        float R[9], sc[3];
        const float *mean = s.a[0][P_MEAN] + 3 * (size_t)i;
        const float *ls = s.a[0][P_SCALE] + 3 * (size_t)i;
        if (mode[i] != 2)
            continue;
        quat_rot(s.a[0][P_ROT] + 4 * (size_t)i, R);
        for (c = 0; c < 3; c++)
            sc[c] = expf(ls[c]);
        for (c = 0; c < 2; c++) {
            float z[3], *nm, *nls;
            int k;
            for (k = 0; k < 3; k++)
                z[k] = rnd_normal(rng) * sc[k];
            state_copy(&s, i, &out, o, 0);
            nm = out.a[0][P_MEAN] + 3 * (size_t)o;
            nls = out.a[0][P_SCALE] + 3 * (size_t)o;
            for (k = 0; k < 3; k++) {
                nm[k] = mean[k] + R[k * 3] * z[0] + R[k * 3 + 1] * z[1] + R[k * 3 + 2] * z[2];
                nls[k] = logf(sc[k] / 1.6f);
            }
            o++;
        }
    }

    /* prune transparent and (optionally) oversized gaussians */
    kept = 0;
    for (i = 0; i < out.n; i++) {
        float op = gs_sigmoid(out.a[0][P_OPAC][i]);
        int prune = op < 0.005f;
        if (prune_big && !prune)
            prune = out_maxr[i] > 20.f || max_scale(&out, i) > 0.1f * extent;
        if (!prune) {
            if (kept != i)
                state_copy(&out, i, &out, kept, 1);
            kept++;
        }
    }
    if (kept == 0 && out.n > 0)
        kept = 1; /* never remove everything */
    out.n = kept;
    if (state_upload(tr, &out) != 0)
        goto done;
    ret = 0;
done:
    free(accum);
    free(denom);
    free(maxr);
    free(out_maxr);
    free(mode);
    state_free(&s);
    state_free(&out);
    return ret;
}

static int reset_opacity(trainer *tr)
{
    host_state s;
    int i;
    const float cap = logf(0.01f / 0.99f);
    memset(&s, 0, sizeof(s));
    if (tr->n == 0)
        return 0;
    if (state_download(tr, &s) != 0) {
        state_free(&s);
        return -1;
    }
    for (i = 0; i < s.n; i++) {
        if (s.a[0][P_OPAC][i] > cap)
            s.a[0][P_OPAC][i] = cap;
        s.a[1][P_OPAC][i] = 0;
        s.a[2][P_OPAC][i] = 0;
    }
    i = state_upload(tr, &s);
    state_free(&s);
    return i;
}

static int save_model(trainer *tr, int sh_deg, const char *out_dir)
{
    host_state s;
    gs_model m;
    char path[1100], path2[1100];
    int r;
    memset(&s, 0, sizeof(s));
    if (state_download(tr, &s) != 0) {
        state_free(&s);
        return -1;
    }
    state_to_model(&s, &m);
    state_free(&s);
    snprintf(path, sizeof(path), "%s/model.ply", out_dir);
    snprintf(path2, sizeof(path2), "%s/model.splat", out_dir);
    r = gs_model_write_ply(&m, sh_deg, path);
    if (r == 0)
        r = gs_model_write_splat(&m, path2);
    gs_model_free(&m);
    return r;
}

/* L1 and PSNR of the current rendering (host side, for logging). */
static void image_metrics(trainer *tr, const gs_camera *c, float *l1_out, float *psnr_out)
{
    size_t hw = (size_t)c->width * c->height, i;
    float *img = (float *)malloc(sizeof(float) * 3 * hw);
    double l1 = 0, mse = 0;
    int ch;
    if (!img)
        return;
    if (G->read(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img) == 0) {
        for (ch = 0; ch < 3; ch++)
            for (i = 0; i < hw; i++) {
                float a = img[ch * hw + i];
                float b = c->rgba[i * 4 + ch] / 255.f;
                a = a < 0 ? 0 : (a > 1 ? 1 : a);
                l1 += fabs(a - b);
                mse += (a - b) * (a - b);
            }
        l1 /= 3.0 * hw;
        mse /= 3.0 * hw;
        *l1_out = (float)l1;
        *psnr_out = (float)(mse > 1e-10 ? -10.0 * log10(mse) : 100.0);
    }
    free(img);
}

/* Writes "<dir>/view_XXX.ppm": photo (left) and rendering (right) for a few views. */
static void write_previews(trainer *tr, const gs_dataset *ds, int sh_deg, const float bg[3], const char *dir)
{
    int v, step = ds->num_cams > 4 ? ds->num_cams / 4 : 1;
    gs_mkdirs(dir);
    for (v = 0; v < ds->num_cams; v += step) {
        const gs_camera *c = &ds->cams[v];
        size_t hw = (size_t)c->width * c->height, i;
        float *img = (float *)malloc(sizeof(float) * 3 * hw);
        unsigned char *row = (unsigned char *)malloc((size_t)c->width * 6);
        char path[1100];
        FILE *f;
        int x, y, ch;
        if (forward(tr, c, sh_deg, bg) != 0 ||
            G->read(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img) != 0) {
            free(img);
            free(row);
            return;
        }
        snprintf(path, sizeof(path), "%s/view_%03d.ppm", dir, v);
        f = fopen(path, "wb");
        if (f) {
            fprintf(f, "P6\n%d %d\n255\n", c->width * 2, c->height);
            for (y = 0; y < c->height; y++) {
                for (x = 0; x < c->width; x++) {
                    i = (size_t)y * c->width + x;
                    for (ch = 0; ch < 3; ch++) {
                        float val = img[ch * hw + i];
                        row[x * 3 + ch] = c->rgba[i * 4 + ch];
                        row[(c->width + x) * 3 + ch] = (unsigned char)(val < 0 ? 0 : (val > 1 ? 255 : val * 255.f + 0.5f));
                    }
                }
                fwrite(row, 1, (size_t)c->width * 6, f);
            }
            fclose(f);
        }
        free(img);
        free(row);
    }
}

int gs_train(gs_gpu *gpu, const gs_dataset *ds, const gs_train_params *p, const char *out_dir,
             gs_train_control *ctl, char *err, size_t err_len)
{
    trainer trs, *tr = &trs;
    gs_model init;
    host_state s;
    int i, step, ret = -1, sh_deg = 0;
    int *perm = NULL, perm_pos = 0;
    unsigned rng = p->seed ? p->seed : 1;
    float bg[3];
    float ema_loss = 0, ema_psnr = 0;
    double t0 = gs_now_seconds();
    int densify_until = p->densify_until;

    memset(tr, 0, sizeof(*tr));
    memset(&s, 0, sizeof(s));
    tr->gpu = gpu;
    bg[0] = bg[1] = bg[2] = p->white_background ? 1.f : 0.f;
    if (densify_until > p->iterations / 2 && p->iterations < 2 * p->densify_until)
        densify_until = p->iterations / 2;

    /* CUDA contexts are per thread: training runs in the server's worker thread */
    if (G->bind_thread(gpu) != 0) {
        snprintf(err, err_len, "cannot use the GPU from this thread (%s)", gpu->error);
        return -1;
    }
    if (create_kernels(tr) != 0) {
        snprintf(err, err_len, "failed to create kernels");
        goto done;
    }

    /* image sized buffers */
    for (i = 0; i < ds->num_cams; i++) {
        if (ds->cams[i].width > tr->max_w)
            tr->max_w = ds->cams[i].width;
        if (ds->cams[i].height > tr->max_h)
            tr->max_h = ds->cams[i].height;
    }
    {
        size_t hw = (size_t)tr->max_w * tr->max_h;
        int tiles = ((tr->max_w + TILE - 1) / TILE) * ((tr->max_h + TILE - 1) / TILE);
        if (mkbuf(tr, &tr->img, hw * 12) || mkbuf(tr, &tr->final_t, hw * 4) ||
            mkbuf(tr, &tr->ncontrib, hw * 4) || mkbuf(tr, &tr->dimg, hw * 12) || mkbuf(tr, &tr->gt, hw * 4) ||
            mkbuf(tr, &tr->tmp5, hw * 4 * 15) || mkbuf(tr, &tr->efh, hw * 4 * 9) ||
            mkbuf(tr, &tr->ssim_map, hw * 12) || mkbuf(tr, &tr->tmp3, hw * 4 * 9) || mkbuf(tr, &tr->cam, 24 * 4) ||
            mkbuf(tr, &tr->ranges, (size_t)tiles * 8)) {
            snprintf(err, err_len, "out of GPU memory (image buffers %dx%d)", tr->max_w, tr->max_h);
            goto done;
        }
        tr->h_tile_count = (int *)malloc(sizeof(int) * (size_t)(tiles + 1));
        tr->h_ranges = (int *)malloc(sizeof(int) * 2 * (size_t)tiles);
        tr->plist_cap = 1 << 20;
        if (mkbuf(tr, &tr->plist, tr->plist_cap * 4)) {
            snprintf(err, err_len, "out of GPU memory");
            goto done;
        }
    }

    /* initial model */
    gs_model_init_from_dataset(&init, ds, rng);
    state_alloc(&s, init.n);
    memcpy(s.a[0][P_MEAN], init.means, sizeof(float) * 3 * (size_t)init.n);
    memcpy(s.a[0][P_SCALE], init.log_scales, sizeof(float) * 3 * (size_t)init.n);
    memcpy(s.a[0][P_ROT], init.rots, sizeof(float) * 4 * (size_t)init.n);
    memcpy(s.a[0][P_OPAC], init.opac, sizeof(float) * (size_t)init.n);
    memcpy(s.a[0][P_SH], init.sh, sizeof(float) * GS_SH_STRIDE * (size_t)init.n);
    gs_model_free(&init);
    if (alloc_gauss(tr, s.n + s.n / 2 + 1024) != 0 || state_upload(tr, &s) != 0) {
        snprintf(err, err_len, "out of GPU memory (gaussians)");
        goto done;
    }
    state_free(&s);

    printf("[train] %d views, %d gaussians, extent %.3f, %d iterations on %s\n", ds->num_cams, tr->n, ds->extent,
           p->iterations, gpu->device_name);

    perm = (int *)malloc(sizeof(int) * (size_t)ds->num_cams);
    perm_pos = ds->num_cams;

    for (step = 1; step <= p->iterations; step++) {
        const gs_camera *cam;
        float t, lr_pos;
        if (ctl && ctl->cancel) {
            ret = 1;
            snprintf(err, err_len, "cancelled");
            goto done;
        }
        if (perm_pos >= ds->num_cams) {
            for (i = 0; i < ds->num_cams; i++)
                perm[i] = i;
            for (i = ds->num_cams - 1; i > 0; i--) {
                int j = (int)(rnd01(&rng) * (i + 1));
                int tmp;
                if (j > i)
                    j = i;
                tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
            }
            perm_pos = 0;
        }
        cam = &ds->cams[perm[perm_pos++]];
        if (p->sh_increase_interval > 0 && step % p->sh_increase_interval == 0 && sh_deg < p->sh_degree)
            sh_deg++;

        if (G->write(tr->gpu, &tr->gt, 0, (size_t)cam->width * cam->height * 4, cam->rgba) != 0 ||
            forward(tr, cam, sh_deg, bg) != 0 || loss_and_backward(tr, cam, sh_deg, bg, p->lambda_ssim) != 0) {
            snprintf(err, err_len, "GPU error during iteration %d (%s)", step, gpu->error);
            goto done;
        }

        t = (float)step / (float)p->iterations;
        if (t > 1.f)
            t = 1.f;
        lr_pos = expf(logf(p->lr_pos_init * ds->extent) * (1.f - t) + logf(p->lr_pos_final * ds->extent) * t);
        if (adam_step(tr, P_MEAN, lr_pos, lr_pos, 1, 1, step) || adam_step(tr, P_SCALE, p->lr_scale, p->lr_scale, 1, 1, step) ||
            adam_step(tr, P_ROT, p->lr_rot, p->lr_rot, 1, 1, step) ||
            adam_step(tr, P_OPAC, p->lr_opacity, p->lr_opacity, 1, 1, step) ||
            adam_step(tr, P_SH, p->lr_sh, p->lr_sh / 20.f, GS_SH_STRIDE, 3, step)) {
            snprintf(err, err_len, "GPU error in optimiser step %d (%s)", step, gpu->error);
            goto done;
        }

        if (step < densify_until) {
            if (accumulate_stats(tr, cam->width, cam->height) != 0) {
                snprintf(err, err_len, "OpenCL error in stats");
                goto done;
            }
            if (step > p->densify_from && step % p->densify_interval == 0) {
                if (densify_and_prune(tr, p, ds->extent, step > p->opacity_reset_interval, 1, &rng) != 0) {
                    snprintf(err, err_len, "densification failed (out of memory?)");
                    goto done;
                }
            }
            if (p->opacity_reset_interval > 0 && step % p->opacity_reset_interval == 0) {
                if (reset_opacity(tr) != 0) {
                    snprintf(err, err_len, "opacity reset failed");
                    goto done;
                }
            }
        }

        if (step % 25 == 0 || step == 1) {
            float l1 = 0, psnr = 0;
            image_metrics(tr, cam, &l1, &psnr);
            ema_loss = step == 1 ? l1 : 0.8f * ema_loss + 0.2f * l1;
            ema_psnr = step == 1 ? psnr : 0.8f * ema_psnr + 0.2f * psnr;
            if (ctl && ctl->progress)
                ctl->progress(ctl->user, step, p->iterations, ema_loss, ema_psnr, tr->n);
        }
        if (step % 500 == 0) {
            double el = gs_now_seconds() - t0;
            printf("[train] it %6d/%d  L1 %.4f  PSNR %.2f dB  gaussians %d  %.1f it/s\n", step, p->iterations,
                   ema_loss, ema_psnr, tr->n, step / (el > 0 ? el : 1));
            fflush(stdout);
        }
        if (p->snapshot_interval > 0 && step % p->snapshot_interval == 0 && step < p->iterations) {
            if (save_model(tr, sh_deg, out_dir) == 0 && ctl && ctl->snapshot)
                ctl->snapshot(ctl->user, step);
        }
    }

    if (p->preview_dir)
        write_previews(tr, ds, sh_deg, bg, p->preview_dir);
    if (save_model(tr, sh_deg, out_dir) != 0) {
        snprintf(err, err_len, "could not write model to %s", out_dir);
        goto done;
    }
    if (ctl && ctl->progress)
        ctl->progress(ctl->user, p->iterations, p->iterations, ema_loss, ema_psnr, tr->n);
    printf("[train] done in %.1f s, %d gaussians\n", gs_now_seconds() - t0, tr->n);
    ret = 0;
done:
    G->finish(gpu);
    state_free(&s);
    free(perm);
    trainer_release(tr);
    return ret;
}

/* ====================================================================== */
/* self test: analytic vs. finite difference gradients                   */
/* ====================================================================== */

static double host_loss(trainer *tr, const gs_camera *c, float lambda)
{
    size_t hw = (size_t)c->width * c->height, i;
    float *img = (float *)malloc(sizeof(float) * 3 * hw);
    float *ss = (float *)malloc(sizeof(float) * 3 * hw);
    double l1 = 0, ssim = 0;
    int ch;
    G->read(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img);
    G->read(tr->gpu, &tr->ssim_map, 0, sizeof(float) * 3 * hw, ss);
    for (ch = 0; ch < 3; ch++)
        for (i = 0; i < hw; i++) {
            l1 += fabs(img[ch * hw + i] - c->rgba[i * 4 + ch] / 255.0);
            ssim += ss[ch * hw + i];
        }
    free(img);
    free(ss);
    l1 /= 3.0 * hw;
    ssim /= 3.0 * hw;
    return (1.0 - lambda) * l1 + lambda * (1.0 - ssim);
}

int gs_selftest(gs_gpu *gpu)
{
    trainer trs, *tr = &trs;
    gs_camera cam;
    host_state s;
    float *grads[NPARAM];
    const int N = 24;
    const int sh_deg = 0; /* the SH view-direction gradient w.r.t. the mean is not modelled */
    const float lambda = 0.2f, eps = 3e-3f;
    const float bg[3] = {0.1f, 0.2f, 0.3f};
    unsigned rng = 7;
    int i, b, failures = 0, checked = 0;
    double base;

    memset(tr, 0, sizeof(*tr));
    memset(&cam, 0, sizeof(cam));
    tr->gpu = gpu;
    if (G->bind_thread(gpu) != 0 || create_kernels(tr) != 0)
        return -1;
    cam.width = 72;
    cam.height = 56;
    cam.fx = cam.fy = 70.f;
    cam.cx = 36.f;
    cam.cy = 28.f;
    /* slightly rotated camera */
    {
        float a = 0.2f;
        float R[9];
        R[0] = cosf(a); R[1] = 0; R[2] = sinf(a);
        R[3] = 0;       R[4] = 1; R[5] = 0;
        R[6] = -sinf(a); R[7] = 0; R[8] = cosf(a);
        memcpy(cam.R, R, sizeof(R));
        cam.t[0] = 0.1f;
        cam.t[1] = -0.05f;
        cam.t[2] = 0.2f;
        for (i = 0; i < 3; i++)
            cam.campos[i] = -(R[i] * cam.t[0] + R[3 + i] * cam.t[1] + R[6 + i] * cam.t[2]);
    }
    cam.rgba = (unsigned char *)malloc((size_t)cam.width * cam.height * 4);
    for (i = 0; i < cam.width * cam.height; i++) {
        int x = i % cam.width, y = i / cam.width;
        cam.rgba[i * 4 + 0] = (unsigned char)(127 + 120 * sinf(x * 0.2f));
        cam.rgba[i * 4 + 1] = (unsigned char)(127 + 120 * cosf(y * 0.15f));
        cam.rgba[i * 4 + 2] = (unsigned char)((x * 3 + y * 2) & 255);
        cam.rgba[i * 4 + 3] = 255;
    }
    tr->max_w = cam.width;
    tr->max_h = cam.height;
    {
        size_t hw = (size_t)cam.width * cam.height;
        int tiles = ((cam.width + TILE - 1) / TILE) * ((cam.height + TILE - 1) / TILE);
        mkbuf(tr, &tr->img, hw * 12);
        mkbuf(tr, &tr->final_t, hw * 4);
        mkbuf(tr, &tr->ncontrib, hw * 4);
        mkbuf(tr, &tr->dimg, hw * 12);
        mkbuf(tr, &tr->gt, hw * 4);
        mkbuf(tr, &tr->tmp5, hw * 60);
        mkbuf(tr, &tr->efh, hw * 36);
        mkbuf(tr, &tr->ssim_map, hw * 12);
        mkbuf(tr, &tr->tmp3, hw * 36);
        mkbuf(tr, &tr->cam, 96);
        mkbuf(tr, &tr->ranges, (size_t)tiles * 8);
        tr->h_tile_count = (int *)malloc(sizeof(int) * (size_t)(tiles + 1));
        tr->h_ranges = (int *)malloc(sizeof(int) * 2 * (size_t)tiles);
        tr->plist_cap = 4096;
        mkbuf(tr, &tr->plist, tr->plist_cap * 4);
        G->write(tr->gpu, &tr->gt, 0, hw * 4, cam.rgba);
    }
    state_alloc(&s, N);
    for (i = 0; i < N; i++) {
        int k;
        s.a[0][P_MEAN][3 * i] = (rnd01(&rng) - 0.5f) * 2.0f;
        s.a[0][P_MEAN][3 * i + 1] = (rnd01(&rng) - 0.5f) * 1.5f;
        s.a[0][P_MEAN][3 * i + 2] = 2.5f + rnd01(&rng) * 2.f;
        for (k = 0; k < 3; k++)
            s.a[0][P_SCALE][3 * i + k] = logf(0.08f + 0.25f * rnd01(&rng));
        for (k = 0; k < 4; k++)
            s.a[0][P_ROT][4 * i + k] = rnd01(&rng) - 0.5f + (k == 0 ? 1.f : 0.f);
        s.a[0][P_OPAC][i] = (rnd01(&rng) - 0.3f) * 3.f;
        for (k = 0; k < 12; k++)
            s.a[0][P_SH][(size_t)i * GS_SH_STRIDE + k] = (rnd01(&rng) - 0.5f) * (k < 3 ? 2.f : 0.5f);
    }
    alloc_gauss(tr, N);
    state_upload(tr, &s);

    forward(tr, &cam, sh_deg, bg);
    loss_and_backward(tr, &cam, sh_deg, bg, lambda);
    base = host_loss(tr, &cam, lambda);
    for (b = 0; b < NPARAM; b++) {
        grads[b] = (float *)malloc(sizeof(float) * (size_t)N * DIMS[b]);
        G->read(tr->gpu, &tr->g[b], 0, sizeof(float) * (size_t)N * DIMS[b], grads[b]);
    }
    printf("[selftest] base loss %.6f\n", base);

    /* d loss / d image at random pixels (checks the L1 + SSIM gradient kernels) */
    {
        size_t hw = (size_t)cam.width * cam.height;
        float *img = (float *)malloc(sizeof(float) * 3 * hw);
        float *dimg = (float *)malloc(sizeof(float) * 3 * hw);
        int bad = 0, cnt = 0, t;
        double max_rel = 0;
        G->read(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img);
        G->read(tr->gpu, &tr->dimg, 0, sizeof(float) * 3 * hw, dimg);
        for (t = 0; t < 40; t++) {
            size_t idx = (size_t)(rnd01(&rng) * 3 * hw) % (3 * hw);
            float orig = img[idx], e = 1e-2f;
            double lp, lm, num, rel;
            img[idx] = orig + e;
            G->write(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img);
            loss_and_backward(tr, &cam, sh_deg, bg, lambda);
            lp = host_loss(tr, &cam, lambda);
            img[idx] = orig - e;
            G->write(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img);
            loss_and_backward(tr, &cam, sh_deg, bg, lambda);
            lm = host_loss(tr, &cam, lambda);
            img[idx] = orig;
            num = (lp - lm) / (2.0 * e);
            rel = fabs(num - dimg[idx]) / (fabs(num) + fabs(dimg[idx]) + 1e-9);
            cnt++;
            if (rel > max_rel)
                max_rel = rel;
            if (rel > 0.1) {
                bad++;
                if (bad <= 3)
                    printf("  image[%u]: analytic %.4e numeric %.4e\n", (unsigned)idx, dimg[idx], num);
            }
        }
        printf("[selftest] %-9s checked %3d  mismatches(>10%%) %2d  max rel err %.3f\n", "image", cnt, bad, max_rel);
        if (bad > cnt / 10 + 1)
            failures++;
        G->write(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img);
        free(img);
        free(dimg);
    }

    /* d loss / d (mean2d, conic): perturb the projected quantities directly (checks render_bwd) */
    forward(tr, &cam, sh_deg, bg);
    loss_and_backward(tr, &cam, sh_deg, bg, lambda);
    {
        float *m2 = (float *)malloc(sizeof(float) * 2 * N), *co = (float *)malloc(sizeof(float) * 4 * N);
        float *gm2 = (float *)malloc(sizeof(float) * 2 * N), *gco = (float *)malloc(sizeof(float) * 3 * N);
        int which;
        G->read(tr->gpu, &tr->mean2d, 0, sizeof(float) * 2 * N, m2);
        G->read(tr->gpu, &tr->conic, 0, sizeof(float) * 4 * N, co);
        G->read(tr->gpu, &tr->dmean2d, 0, sizeof(float) * 2 * N, gm2);
        G->read(tr->gpu, &tr->dconic, 0, sizeof(float) * 3 * N, gco);
        for (which = 0; which < 2; which++) {
            int bad = 0, cnt = 0, k, nel = which == 0 ? 2 : 3;
            double max_rel = 0;
            for (i = 0; i < N; i++)
                for (k = 0; k < nel; k++) {
                    float *arr = which == 0 ? m2 : co;
                    size_t idx = which == 0 ? (size_t)2 * i + k : (size_t)4 * i + k;
                    float orig = arr[idx];
                    float e = which == 0 ? 0.05f : fabsf(orig) * 0.01f + 1e-6f;
                    double lp, lm, num, ana, rel;
                    gs_buf *buf = which == 0 ? &tr->mean2d : &tr->conic;
                    size_t bytes = sizeof(float) * (which == 0 ? 2 : 4) * N;
                    arr[idx] = orig + e;
                    G->write(tr->gpu, buf, 0, bytes, arr);
                    forward_ex(tr, &cam, sh_deg, bg, 1);
                    loss_and_backward(tr, &cam, sh_deg, bg, lambda);
                    lp = host_loss(tr, &cam, lambda);
                    arr[idx] = orig - e;
                    G->write(tr->gpu, buf, 0, bytes, arr);
                    forward_ex(tr, &cam, sh_deg, bg, 1);
                    loss_and_backward(tr, &cam, sh_deg, bg, lambda);
                    lm = host_loss(tr, &cam, lambda);
                    arr[idx] = orig;
                    G->write(tr->gpu, buf, 0, bytes, arr);
                    num = (lp - lm) / (2.0 * e);
                    ana = which == 0 ? gm2[2 * i + k] : gco[3 * i + k];
                    if (fabs(num) < 1e-7 && fabs(ana) < 1e-7)
                        continue;
                    rel = fabs(num - ana) / (fabs(num) + fabs(ana) + 1e-9);
                    cnt++;
                    if (rel > max_rel)
                        max_rel = rel;
                    if (rel > 0.1) {
                        bad++;
                        if (bad <= 3)
                            printf("  %s[%d][%d]: analytic %.4e numeric %.4e\n", which ? "conic" : "mean2d", i, k, ana, num);
                    }
                }
            printf("[selftest] %-9s checked %3d  mismatches(>10%%) %2d  max rel err %.3f\n", which ? "conic" : "mean2d",
                   cnt, bad, max_rel);
            if (bad > cnt / 10 + 1)
                failures++;
        }
        free(m2);
        free(co);
        free(gm2);
        free(gco);
    }

    for (b = 0; b < NPARAM; b++) {
        static const char *names[NPARAM] = {"mean", "log_scale", "rot", "opacity", "sh"};
        int nel = b == P_SH ? 12 : DIMS[b];
        double max_rel = 0;
        int bad = 0, cnt = 0;
        for (i = 0; i < N; i++) {
            int k;
            for (k = 0; k < nel; k++) {
                size_t idx = (size_t)i * DIMS[b] + k;
                float orig = s.a[0][b][idx];
                double lp, lm, num, ana, rel;
                s.a[0][b][idx] = orig + eps;
                state_upload(tr, &s);
                forward(tr, &cam, sh_deg, bg);
                loss_and_backward(tr, &cam, sh_deg, bg, lambda);
                lp = host_loss(tr, &cam, lambda);
                s.a[0][b][idx] = orig - eps;
                state_upload(tr, &s);
                forward(tr, &cam, sh_deg, bg);
                loss_and_backward(tr, &cam, sh_deg, bg, lambda);
                lm = host_loss(tr, &cam, lambda);
                s.a[0][b][idx] = orig;
                num = (lp - lm) / (2.0 * eps);
                ana = grads[b][idx];
                if (fabs(num) < 1e-6 && fabs(ana) < 1e-6)
                    continue;
                rel = fabs(num - ana) / (fabs(num) + fabs(ana) + 1e-6);
                cnt++;
                if (rel > max_rel)
                    max_rel = rel;
                if (rel > 0.1) {
                    bad++;
                    if (bad <= 3)
                        printf("  %s[%d][%d]: analytic %.4e numeric %.4e\n", names[b], i, k, ana, num);
                }
            }
        }
        printf("[selftest] %-9s checked %3d  mismatches(>10%%) %2d  max rel err %.3f\n", names[b], cnt, bad,
               max_rel);
        checked += cnt;
        /* tolerate a few outliers caused by tile/threshold discontinuities */
        if (bad > cnt / 10 + 1)
            failures++;
    }
    for (b = 0; b < NPARAM; b++)
        free(grads[b]);
    state_free(&s);
    free(cam.rgba);
    trainer_release(tr);
    printf("[selftest] %s (%d gradients checked)\n", failures ? "FAILED" : "PASSED", checked);
    return failures ? 1 : 0;
}
