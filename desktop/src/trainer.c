/*
 * trainer.c - GPU gaussian splatting optimisation loop (CUDA or OpenCL, see gpu.h).
 *
 * Everything that runs per iteration lives on the GPU:
 *   1. preprocess: project gaussians, evaluate colours, count covered tiles
 *   2. binning: radix sort by depth, emit (tile, gaussian) pairs, radix sort by tile,
 *      tile ranges (only the number of pairs is read back)
 *   3. render_fwd, loss + gradient (L1 + D-SSIM), render_bwd, preprocess_bwd, Adam
 *   4. every `densify_interval` iterations: clone / split / prune (GPU, scan based
 *      compaction), periodic opacity reset (GPU)
 * All training photos are kept in GPU memory when they fit. The host only
 * schedules kernels, reads a few counters and downloads the model for saving.
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
#define SCAN_BLOCK_ELEMS 1024
#define SCAN_LEVELS 6
#define NPARAM 5
static const int DIMS[NPARAM] = {3, 3, 4, 1, GS_SH_STRIDE}; /* means, scales, rots, opacity, sh */

enum { P_MEAN = 0, P_SCALE = 1, P_ROT = 2, P_OPAC = 3, P_SH = 4 };

typedef struct {
    gs_gpu *gpu;
    /* kernels */
    gs_kernel k_pre, k_fwd, k_ssim_h, k_ssim_v, k_ssim_bh, k_loss, k_bwd, k_pre_bwd, k_adam, k_stats;
    gs_kernel k_scan_block, k_scan_add, k_hist, k_scatter, k_gather, k_dup, k_ranges, k_metrics;
    gs_kernel k_dcount, k_dscatter, k_dsplit, k_oreset;
    /* parameters + Adam moments (exactly `cap` gaussians) */
    int n, cap;
    gs_buf p[NPARAM], m[NPARAM], v[NPARAM];
    /* per gaussian work buffers (aux_cap gaussians) */
    int aux_cap;
    gs_buf g[NPARAM];
    gs_buf mean2d, radii, conic, colors;
    gs_buf dmean2d, dconic, dopa, dcolor;
    gs_buf accum, denom, maxr;
    gs_buf tiles_touched, dkey, did, dkey2, did2, tt_sorted, offsets;
    gs_buf dflags, dcounts, doffsets;
    /* (tile, gaussian) pairs */
    size_t pair_cap;
    gs_buf pkey, pid, pkey2, pid2;
    gs_buf *plist; /* buffer holding the tile sorted gaussian ids */
    int num_pairs;
    /* radix sort histograms and scan temporaries */
    size_t hist_cap;
    gs_buf hist, hoff;
    size_t scan_cap[SCAN_LEVELS];
    gs_buf scan_sum[SCAN_LEVELS], scan_out[SCAN_LEVELS];
    /* image sized buffers */
    int max_w, max_h;
    gs_buf img, final_t, ncontrib, dimg, tmp5, efh, ssim_map, tmp3, cam, ranges, partial;
    float *h_partial;
    /* ground truth photos: all resident (gt_off per view) or one at a time */
    gs_buf gt;
    int gt_resident;
    int *gt_off;
    int cur_gt_off;
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
#define G (tr->gpu->ops)
#define HAS(b) ((b).u.dev != 0 || (b).u.ptr != NULL)

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
    return G->launch(tr->gpu, k, 1, global, local, nargs, args);
}

static int launch_image(trainer *tr, gs_kernel k, int W, int H, int depth, int nargs, const gs_arg *args)
{
    size_t global[3], local[3] = {TILE, TILE, 1};
    global[0] = (size_t)W;
    global[1] = (size_t)H;
    global[2] = (size_t)depth;
    return G->launch(tr->gpu, k, depth > 1 ? 3 : 2, global, local, nargs, args);
}

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

/* ---------------------------------------------------------------------- */
/* buffers                                                                 */
/* ---------------------------------------------------------------------- */

static int mkbuf(trainer *tr, gs_buf *b, size_t bytes)
{
    if (G->alloc(tr->gpu, b, bytes) != 0) {
        fprintf(stderr, "[gpu] allocation of %.1f MB failed\n", bytes / (1024.0 * 1024.0));
        return -1;
    }
    return 0;
}

static void rel(trainer *tr, gs_buf *b)
{
    if (HAS(*b))
        G->free(tr->gpu, b);
    memset(b, 0, sizeof(*b));
}

static int fill_zero(trainer *tr, gs_buf *b, size_t bytes)
{
    return G->zero(tr->gpu, b, bytes);
}

static void release_params(trainer *tr)
{
    int i;
    for (i = 0; i < NPARAM; i++) {
        rel(tr, &tr->p[i]);
        rel(tr, &tr->m[i]);
        rel(tr, &tr->v[i]);
    }
    tr->cap = 0;
}

static void release_aux(trainer *tr)
{
    gs_buf *bufs[] = {&tr->mean2d, &tr->radii, &tr->conic, &tr->colors, &tr->dmean2d, &tr->dconic, &tr->dopa,
                      &tr->dcolor, &tr->accum, &tr->denom, &tr->maxr, &tr->tiles_touched, &tr->dkey, &tr->did,
                      &tr->dkey2, &tr->did2, &tr->tt_sorted, &tr->offsets, &tr->dflags, &tr->dcounts, &tr->doffsets};
    int i;
    for (i = 0; i < NPARAM; i++)
        rel(tr, &tr->g[i]);
    for (i = 0; i < (int)(sizeof(bufs) / sizeof(bufs[0])); i++)
        rel(tr, bufs[i]);
    tr->aux_cap = 0;
}

/* per gaussian work buffers for at least n gaussians (contents are lost on growth) */
static int ensure_aux(trainer *tr, int n)
{
    size_t c;
    int i;
    if (n <= tr->aux_cap && tr->aux_cap > 0)
        return 0;
    release_aux(tr);
    c = (size_t)n + (size_t)n / 2 + 1024;
    for (i = 0; i < NPARAM; i++)
        if (mkbuf(tr, &tr->g[i], c * DIMS[i] * sizeof(float)))
            return -1;
    if (mkbuf(tr, &tr->mean2d, c * 8) || mkbuf(tr, &tr->radii, c * 4) || mkbuf(tr, &tr->conic, c * 16) ||
        mkbuf(tr, &tr->colors, c * 12) || mkbuf(tr, &tr->dmean2d, c * 8) || mkbuf(tr, &tr->dconic, c * 12) ||
        mkbuf(tr, &tr->dopa, c * 4) || mkbuf(tr, &tr->dcolor, c * 12) || mkbuf(tr, &tr->accum, c * 4) ||
        mkbuf(tr, &tr->denom, c * 4) || mkbuf(tr, &tr->maxr, c * 4) || mkbuf(tr, &tr->tiles_touched, c * 4) ||
        mkbuf(tr, &tr->dkey, c * 4) || mkbuf(tr, &tr->did, c * 4) || mkbuf(tr, &tr->dkey2, c * 4) ||
        mkbuf(tr, &tr->did2, c * 4) || mkbuf(tr, &tr->tt_sorted, c * 4) || mkbuf(tr, &tr->offsets, c * 4) ||
        mkbuf(tr, &tr->dflags, c * 4) || mkbuf(tr, &tr->dcounts, c * 4) || mkbuf(tr, &tr->doffsets, c * 4))
        return -1;
    tr->aux_cap = (int)c;
    if (fill_zero(tr, &tr->accum, c * 4) || fill_zero(tr, &tr->denom, c * 4) || fill_zero(tr, &tr->maxr, c * 4))
        return -1;
    return 0;
}

static int alloc_params(trainer *tr, gs_buf *p, gs_buf *m, gs_buf *v, int cap)
{
    int i;
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = (size_t)(cap > 0 ? cap : 1) * DIMS[i] * sizeof(float);
        if (mkbuf(tr, &p[i], bytes) || mkbuf(tr, &m[i], bytes) || mkbuf(tr, &v[i], bytes))
            return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* host <-> GPU model transfer (initialisation, saving, self test)         */
/* ---------------------------------------------------------------------- */

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

static int state_download(trainer *tr, host_state *s, int with_moments)
{
    int i;
    state_alloc(s, tr->n);
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = (size_t)tr->n * DIMS[i] * sizeof(float);
        if (G->read(tr->gpu, &tr->p[i], 0, bytes, s->a[0][i]))
            return -1;
        if (with_moments &&
            (G->read(tr->gpu, &tr->m[i], 0, bytes, s->a[1][i]) || G->read(tr->gpu, &tr->v[i], 0, bytes, s->a[2][i])))
            return -1;
    }
    return 0;
}

static int state_upload(trainer *tr, const host_state *s)
{
    int i;
    if (s->n > tr->cap) {
        release_params(tr);
        if (alloc_params(tr, tr->p, tr->m, tr->v, s->n) != 0)
            return -1;
        tr->cap = s->n;
    }
    if (ensure_aux(tr, s->n) != 0)
        return -1;
    tr->n = s->n;
    for (i = 0; i < NPARAM; i++) {
        size_t bytes = (size_t)s->n * DIMS[i] * sizeof(float);
        if (G->write(tr->gpu, &tr->p[i], 0, bytes, s->a[0][i]) || G->write(tr->gpu, &tr->m[i], 0, bytes, s->a[1][i]) ||
            G->write(tr->gpu, &tr->v[i], 0, bytes, s->a[2][i]))
            return -1;
    }
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

/* ---------------------------------------------------------------------- */
/* GPU primitives: exclusive scan and stable radix sort                    */
/* ---------------------------------------------------------------------- */

static int read_int(trainer *tr, gs_buf *b, int index, int *out)
{
    return G->read(tr->gpu, b, (size_t)index * sizeof(int), sizeof(int), out);
}

/* exclusive prefix sum of n ints: out[i] = in[0] + ... + in[i-1] (in != out) */
static int gpu_scan(trainer *tr, gs_buf *in, gs_buf *out, int n, int level)
{
    int nb = (n + SCAN_BLOCK_ELEMS - 1) / SCAN_BLOCK_ELEMS;
    if (n <= 0)
        return 0;
    if (level >= SCAN_LEVELS)
        return -1;
    if ((size_t)nb * sizeof(int) > tr->scan_cap[level] || !HAS(tr->scan_sum[level])) {
        rel(tr, &tr->scan_sum[level]);
        rel(tr, &tr->scan_out[level]);
        tr->scan_cap[level] = (size_t)nb * sizeof(int) * 2 + 4096;
        CK(mkbuf(tr, &tr->scan_sum[level], tr->scan_cap[level]));
        CK(mkbuf(tr, &tr->scan_out[level], tr->scan_cap[level]));
    }
    {
        gs_arg a[] = {VAL(n), BUF(*in), BUF(*out), BUF(tr->scan_sum[level])};
        CK(launch1d(tr, tr->k_scan_block, (size_t)nb * LOCAL_1D, NARGS(a), a));
    }
    if (nb > 1) {
        CK(gpu_scan(tr, &tr->scan_sum[level], &tr->scan_out[level], nb, level + 1));
        {
            gs_arg a[] = {VAL(n), BUF(*out), BUF(tr->scan_out[level])};
            CK(launch1d(tr, tr->k_scan_add, (size_t)n, NARGS(a), a));
        }
    }
    return 0;
fail:
    return -1;
}

/* Stable LSD radix sort of (key, value) pairs on the low `bits` bits of the key,
 * 4 bits per pass. On return *result is 0 if the sorted data is in (k0, v0), 1 if in (k1, v1). */
static int gpu_radix_sort(trainer *tr, gs_buf *k0, gs_buf *v0, gs_buf *k1, gs_buf *v1, int n, int bits, int *result)
{
    int nblocks = (n + LOCAL_1D - 1) / LOCAL_1D, shift, cur = 0;
    gs_buf *ks[2], *vs[2];
    size_t hbytes = (size_t)nblocks * 16 * sizeof(int);
    ks[0] = k0;
    ks[1] = k1;
    vs[0] = v0;
    vs[1] = v1;
    *result = 0;
    if (n <= 1)
        return 0;
    if (hbytes > tr->hist_cap || !HAS(tr->hist)) {
        rel(tr, &tr->hist);
        rel(tr, &tr->hoff);
        tr->hist_cap = hbytes + hbytes / 2 + 4096;
        CK(mkbuf(tr, &tr->hist, tr->hist_cap));
        CK(mkbuf(tr, &tr->hoff, tr->hist_cap));
    }
    for (shift = 0; shift < bits; shift += 4) {
        int nh = nblocks * 16;
        gs_arg ah[] = {VAL(n), BUF(*ks[cur]), VAL(shift), VAL(nblocks), BUF(tr->hist)};
        gs_arg as[] = {VAL(n),     BUF(*ks[cur]), BUF(*vs[cur]),  BUF(*ks[1 - cur]),
                       BUF(*vs[1 - cur]), VAL(shift), VAL(nblocks), BUF(tr->hoff)};
        CK(launch1d(tr, tr->k_hist, (size_t)nblocks * LOCAL_1D, NARGS(ah), ah));
        CK(gpu_scan(tr, &tr->hist, &tr->hoff, nh, 0));
        CK(launch1d(tr, tr->k_scatter, (size_t)nblocks * LOCAL_1D, NARGS(as), as));
        cur = 1 - cur;
    }
    *result = cur;
    return 0;
fail:
    return -1;
}

/* ---------------------------------------------------------------------- */
/* binning                                                                 */
/* ---------------------------------------------------------------------- */

static int bits_for(int v)
{
    int b = 1;
    while ((1 << b) < v && b < 31)
        b++;
    return b;
}

static int bin_gaussians(trainer *tr, int W, int H)
{
    int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE, ntiles = gx * gy;
    int N = tr->n, total = 0, which = 0;
    gs_buf *order;

    CK(fill_zero(tr, &tr->ranges, (size_t)ntiles * 2 * sizeof(int)));
    tr->num_pairs = 0;
    if (!HAS(tr->pid)) {
        tr->pair_cap = 4096;
        CK(mkbuf(tr, &tr->pkey, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pid, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pkey2, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pid2, tr->pair_cap * 4));
    }
    tr->plist = &tr->pid;
    if (N == 0)
        return 0;

    /* 1. gaussians front to back (depth keys from preprocess, culled ones last) */
    CK(gpu_radix_sort(tr, &tr->dkey, &tr->did, &tr->dkey2, &tr->did2, N, 32, &which));
    order = which ? &tr->did2 : &tr->did;
    {
        gs_arg ag[] = {VAL(N), BUF(*order), BUF(tr->tiles_touched), BUF(tr->tt_sorted)};
        int last_off = 0, last_cnt = 0;
        CK(launch1d(tr, tr->k_gather, (size_t)N, NARGS(ag), ag));
        /* 2. output offset of every gaussian's pairs; the total is the only readback */
        CK(gpu_scan(tr, &tr->tt_sorted, &tr->offsets, N, 0));
        CK(read_int(tr, &tr->offsets, N - 1, &last_off));
        CK(read_int(tr, &tr->tt_sorted, N - 1, &last_cnt));
        total = last_off + last_cnt;
    }
    if (total <= 0)
        return 0;
    if ((size_t)total > tr->pair_cap) {
        rel(tr, &tr->pkey);
        rel(tr, &tr->pid);
        rel(tr, &tr->pkey2);
        rel(tr, &tr->pid2);
        tr->pair_cap = (size_t)total + (size_t)total / 2 + 4096;
        CK(mkbuf(tr, &tr->pkey, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pid, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pkey2, tr->pair_cap * 4));
        CK(mkbuf(tr, &tr->pid2, tr->pair_cap * 4));
        tr->plist = &tr->pid;
    }
    /* 3. pairs in depth order */
    {
        gs_arg ad[] = {VAL(N),  BUF(*order), BUF(tr->offsets), BUF(tr->mean2d), BUF(tr->radii),
                       VAL(gx), VAL(gy),     BUF(tr->pkey),    BUF(tr->pid)};
        CK(launch1d(tr, tr->k_dup, (size_t)N, NARGS(ad), ad));
    }
    /* 4. stable sort by tile keeps the depth order inside every tile */
    CK(gpu_radix_sort(tr, &tr->pkey, &tr->pid, &tr->pkey2, &tr->pid2, total, bits_for(ntiles), &which));
    {
        gs_buf *keys = which ? &tr->pkey2 : &tr->pkey;
        gs_arg ar[] = {VAL(total), BUF(*keys), BUF(tr->ranges)};
        tr->plist = which ? &tr->pid2 : &tr->pid;
        CK(launch1d(tr, tr->k_ranges, (size_t)total, NARGS(ar), ar));
    }
    tr->num_pairs = total;
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
    MK(k_scan_block, "scan_block");
    MK(k_scan_add, "scan_add");
    MK(k_hist, "radix_hist");
    MK(k_scatter, "radix_scatter");
    MK(k_gather, "gather_counts");
    MK(k_dup, "duplicate_pairs");
    MK(k_ranges, "tile_ranges");
    MK(k_metrics, "image_metrics");
    MK(k_dcount, "densify_count");
    MK(k_dscatter, "densify_scatter");
    MK(k_dsplit, "densify_split");
    MK(k_oreset, "opacity_reset");
#undef MK
    return 0;
}

static void trainer_release(trainer *tr)
{
    gs_kernel *ks[] = {&tr->k_pre,        &tr->k_fwd,      &tr->k_ssim_h,  &tr->k_ssim_v,  &tr->k_ssim_bh,
                       &tr->k_loss,       &tr->k_bwd,      &tr->k_pre_bwd, &tr->k_adam,    &tr->k_stats,
                       &tr->k_scan_block, &tr->k_scan_add, &tr->k_hist,    &tr->k_scatter, &tr->k_gather,
                       &tr->k_dup,        &tr->k_ranges,   &tr->k_metrics, &tr->k_dcount,  &tr->k_dscatter,
                       &tr->k_dsplit,     &tr->k_oreset};
    gs_buf *bufs[] = {&tr->pkey,    &tr->pid,      &tr->pkey2, &tr->pid2, &tr->hist, &tr->hoff,
                      &tr->img,     &tr->final_t,  &tr->ncontrib, &tr->dimg, &tr->tmp5, &tr->efh,
                      &tr->ssim_map, &tr->tmp3,    &tr->cam,   &tr->ranges, &tr->partial, &tr->gt};
    size_t i;
    if (!tr->gpu || !tr->gpu->ops)
        return;
    for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
        if (*ks[i])
            G->kernel_release(tr->gpu, *ks[i]);
    release_params(tr);
    release_aux(tr);
    for (i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        rel(tr, bufs[i]);
    for (i = 0; i < SCAN_LEVELS; i++) {
        rel(tr, &tr->scan_sum[i]);
        rel(tr, &tr->scan_out[i]);
    }
    free(tr->h_partial);
    free(tr->gt_off);
}

/* image sized buffers for views up to w x h */
static int alloc_image_buffers(trainer *tr, int w, int h)
{
    size_t hw = (size_t)w * h;
    int tiles = ((w + TILE - 1) / TILE) * ((h + TILE - 1) / TILE);
    size_t nb = (3 * hw + LOCAL_1D - 1) / LOCAL_1D;
    tr->max_w = w;
    tr->max_h = h;
    if (mkbuf(tr, &tr->img, hw * 12) || mkbuf(tr, &tr->final_t, hw * 4) || mkbuf(tr, &tr->ncontrib, hw * 4) ||
        mkbuf(tr, &tr->dimg, hw * 12) || mkbuf(tr, &tr->tmp5, hw * 4 * 15) || mkbuf(tr, &tr->efh, hw * 4 * 9) ||
        mkbuf(tr, &tr->ssim_map, hw * 12) || mkbuf(tr, &tr->tmp3, hw * 4 * 9) || mkbuf(tr, &tr->cam, 24 * 4) ||
        mkbuf(tr, &tr->ranges, (size_t)tiles * 8) || mkbuf(tr, &tr->partial, nb * 2 * sizeof(float)))
        return -1;
    tr->h_partial = (float *)malloc(nb * 2 * sizeof(float));
    return 0;
}

/* uploads all photos once if they fit in a part of the GPU memory, else one per iteration */
static int setup_ground_truth(trainer *tr, const gs_dataset *ds)
{
    size_t total = 0, pix = 0;
    int i;
    tr->gt_off = (int *)calloc((size_t)ds->num_cams, sizeof(int));
    for (i = 0; i < ds->num_cams; i++)
        total += (size_t)ds->cams[i].width * ds->cams[i].height * 4;
    tr->gt_resident = total <= (size_t)((double)tr->gpu->global_mem * 0.3) && total / 4 < 0x7fffffff;
    if (!tr->gt_resident) {
        printf("[train] photos (%.0f MB) are streamed to the GPU each iteration\n", total / (1024.0 * 1024.0));
        return mkbuf(tr, &tr->gt, (size_t)tr->max_w * tr->max_h * 4);
    }
    if (mkbuf(tr, &tr->gt, total))
        return -1;
    for (i = 0; i < ds->num_cams; i++) {
        size_t bytes = (size_t)ds->cams[i].width * ds->cams[i].height * 4;
        tr->gt_off[i] = (int)pix;
        if (G->write(tr->gpu, &tr->gt, pix * 4, bytes, ds->cams[i].rgba))
            return -1;
        pix += bytes / 4;
    }
    printf("[train] %d photos resident in GPU memory (%.0f MB)\n", ds->num_cams, total / (1024.0 * 1024.0));
    return 0;
}

static int select_ground_truth(trainer *tr, const gs_dataset *ds, int view)
{
    const gs_camera *c = &ds->cams[view];
    if (tr->gt_resident) {
        tr->cur_gt_off = tr->gt_off[view];
        return 0;
    }
    tr->cur_gt_off = 0;
    return G->write(tr->gpu, &tr->gt, 0, (size_t)c->width * c->height * 4, c->rgba);
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

/* ---------------------------------------------------------------------- */
/* forward / backward                                                      */
/* ---------------------------------------------------------------------- */

/* Forward pass for one view: preprocess + binning + render. */
static int forward_ex(trainer *tr, const gs_camera *c, int sh_deg, const float bg[3], int render_only)
{
    float camv[24];
    int N = tr->n, W = c->width, H = c->height;

    fill_cam(c, camv);
    if (!render_only) {
        CK(G->write(tr->gpu, &tr->cam, 0, sizeof(camv), camv));
        if (N > 0) {
            gs_arg a[] = {VAL(N),              BUF(tr->p[P_MEAN]), BUF(tr->p[P_SCALE]), BUF(tr->p[P_ROT]),
                          BUF(tr->p[P_OPAC]),  BUF(tr->p[P_SH]),   VAL(sh_deg),         BUF(tr->cam),
                          BUF(tr->mean2d),     BUF(tr->radii),     BUF(tr->conic),      BUF(tr->colors),
                          BUF(tr->tiles_touched), BUF(tr->dkey),   BUF(tr->did)};
            CK(launch1d(tr, tr->k_pre, (size_t)N, NARGS(a), a));
        }
        CK(bin_gaussians(tr, W, H));
    }
    {
        gs_arg a[] = {BUF(tr->ranges), BUF(*tr->plist),  VAL(W),     VAL(H),     BUF(tr->mean2d),
                      BUF(tr->conic),  BUF(tr->colors),  VAL(bg[0]), VAL(bg[1]), VAL(bg[2]),
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
    int N = tr->n, W = c->width, H = c->height, go = tr->cur_gt_off;
    {
        gs_arg a1[] = {VAL(W), VAL(H), BUF(tr->img), BUF(tr->gt), VAL(go), BUF(tr->tmp5)};
        gs_arg a2[] = {VAL(W), VAL(H), BUF(tr->tmp5), BUF(tr->efh), BUF(tr->ssim_map)};
        gs_arg a3[] = {VAL(W), VAL(H), BUF(tr->efh), BUF(tr->tmp3)};
        gs_arg a4[] = {VAL(W),      VAL(H),  VAL(lambda),   BUF(tr->img),
                       BUF(tr->gt), VAL(go), BUF(tr->tmp3), BUF(tr->dimg)};
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
        gs_arg a[] = {BUF(tr->ranges), BUF(*tr->plist),  VAL(W),           VAL(H),
                      BUF(tr->mean2d), BUF(tr->conic),   BUF(tr->colors),  VAL(bg[0]),
                      VAL(bg[1]),      VAL(bg[2]),       BUF(tr->final_t), BUF(tr->ncontrib),
                      BUF(tr->dimg),   BUF(tr->dmean2d), BUF(tr->dconic),  BUF(tr->dopa),
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
        gs_arg a[] = {VAL(n),   BUF(tr->p[which]), BUF(tr->g[which]), BUF(tr->m[which]), BUF(tr->v[which]),
                      VAL(lr1), VAL(lr2),          VAL(group),        VAL(first),        VAL(b1),
                      VAL(b2),  VAL(eps),          VAL(bc1),          VAL(bc2)};
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
/* densification (GPU)                                                     */
/* ---------------------------------------------------------------------- */

static int densify_count(trainer *tr, const gs_train_params *p, float extent, int prune_big, int allow_growth,
                         int *total)
{
    int n = tr->n, last_off = 0, last_cnt = 0;
    float grad_thr = p->densify_grad_threshold, clone_max = p->percent_dense * extent, big = 0.1f * extent;
    gs_arg a[] = {VAL(n),            BUF(tr->p[P_SCALE]), BUF(tr->p[P_OPAC]), BUF(tr->accum), BUF(tr->denom),
                  BUF(tr->maxr),     VAL(grad_thr),       VAL(clone_max),     VAL(prune_big), VAL(big),
                  VAL(allow_growth), BUF(tr->dflags),     BUF(tr->dcounts)};
    CK(launch1d(tr, tr->k_dcount, (size_t)n, NARGS(a), a));
    CK(gpu_scan(tr, &tr->dcounts, &tr->doffsets, n, 0));
    CK(read_int(tr, &tr->doffsets, n - 1, &last_off));
    CK(read_int(tr, &tr->dcounts, n - 1, &last_cnt));
    *total = last_off + last_cnt;
    return 0;
fail:
    return -1;
}

static int reset_stats(trainer *tr)
{
    size_t bytes = (size_t)tr->n * 4;
    return fill_zero(tr, &tr->accum, bytes) || fill_zero(tr, &tr->denom, bytes) || fill_zero(tr, &tr->maxr, bytes);
}

/* clone / split / prune entirely on the GPU: decide, scan, scatter into new buffers */
static int densify_and_prune(trainer *tr, const gs_train_params *p, float extent, int prune_big, unsigned seed)
{
    gs_buf np[NPARAM], nm[NPARAM], nv[NPARAM];
    int n = tr->n, total = 0, b;
    if (n == 0)
        return 0;
    memset(np, 0, sizeof(np));
    memset(nm, 0, sizeof(nm));
    memset(nv, 0, sizeof(nv));
    CK(densify_count(tr, p, extent, prune_big, 1, &total));
    if (total > p->max_gaussians) /* budget exhausted: prune only */
        CK(densify_count(tr, p, extent, prune_big, 0, &total));
    if (total <= 0) /* never remove everything */
        return reset_stats(tr);

    if (alloc_params(tr, np, nm, nv, total) != 0) {
        fprintf(stderr, "[train] not enough GPU memory to densify to %d gaussians, skipping\n", total);
        for (b = 0; b < NPARAM; b++) {
            rel(tr, &np[b]);
            rel(tr, &nm[b]);
            rel(tr, &nv[b]);
        }
        return reset_stats(tr);
    }
    for (b = 0; b < NPARAM; b++) {
        int dim = DIMS[b];
        gs_arg a[] = {VAL(n),        VAL(dim),      BUF(tr->dflags), BUF(tr->doffsets), BUF(tr->p[b]),
                      BUF(tr->m[b]), BUF(tr->v[b]), BUF(np[b]),      BUF(nm[b]),        BUF(nv[b])};
        CK(launch1d(tr, tr->k_dscatter, (size_t)n, NARGS(a), a));
    }
    {
        gs_arg a[] = {VAL(n),              BUF(tr->dflags),   BUF(tr->doffsets), BUF(tr->p[P_MEAN]),
                      BUF(tr->p[P_SCALE]), BUF(tr->p[P_ROT]), BUF(np[P_MEAN]),   BUF(np[P_SCALE]),
                      VAL(seed)};
        CK(launch1d(tr, tr->k_dsplit, (size_t)n, NARGS(a), a));
    }
    release_params(tr);
    memcpy(tr->p, np, sizeof(np));
    memcpy(tr->m, nm, sizeof(nm));
    memcpy(tr->v, nv, sizeof(nv));
    tr->cap = total;
    tr->n = total;
    CK(ensure_aux(tr, total));
    CK(reset_stats(tr));
    return 0;
fail:
    return -1;
}

static int reset_opacity(trainer *tr)
{
    const float cap = logf(0.01f / 0.99f);
    int n = tr->n;
    if (n == 0)
        return 0;
    {
        gs_arg a[] = {VAL(n), BUF(tr->p[P_OPAC]), BUF(tr->m[P_OPAC]), BUF(tr->v[P_OPAC]), VAL(cap)};
        CK(launch1d(tr, tr->k_oreset, (size_t)n, NARGS(a), a));
    }
    return 0;
fail:
    return -1;
}

/* ---------------------------------------------------------------------- */
/* output                                                                  */
/* ---------------------------------------------------------------------- */

static int save_model(trainer *tr, int sh_deg, const char *out_dir)
{
    host_state s;
    gs_model m;
    char path[1100], path2[1100];
    int r;
    memset(&s, 0, sizeof(s));
    if (state_download(tr, &s, 0) != 0) {
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

/* L1 and PSNR of the current rendering, reduced on the GPU (logging only) */
static void image_metrics(trainer *tr, const gs_camera *c, float *l1_out, float *psnr_out)
{
    int hw = c->width * c->height, n3 = 3 * hw, go = tr->cur_gt_off;
    int nb = (n3 + LOCAL_1D - 1) / LOCAL_1D, i;
    double l1 = 0, se = 0;
    gs_arg a[] = {VAL(n3), VAL(hw), BUF(tr->img), BUF(tr->gt), VAL(go), BUF(tr->partial)};
    if (launch1d(tr, tr->k_metrics, (size_t)nb * LOCAL_1D, NARGS(a), a) != 0 ||
        G->read(tr->gpu, &tr->partial, 0, (size_t)nb * 2 * sizeof(float), tr->h_partial) != 0)
        return;
    for (i = 0; i < nb; i++) {
        l1 += tr->h_partial[2 * i];
        se += tr->h_partial[2 * i + 1];
    }
    l1 /= n3;
    se /= n3;
    *l1_out = (float)l1;
    *psnr_out = (float)(se > 1e-10 ? -10.0 * log10(se) : 100.0);
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
        if (forward(tr, c, sh_deg, bg) != 0 || G->read(tr->gpu, &tr->img, 0, sizeof(float) * 3 * hw, img) != 0) {
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
                        row[(c->width + x) * 3 + ch] =
                            (unsigned char)(val < 0 ? 0 : (val > 1 ? 255 : val * 255.f + 0.5f));
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

static float rnd01(unsigned *s)
{
    unsigned x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return ((x >> 8) + 0.5f) / 16777216.0f;
}

/* ---------------------------------------------------------------------- */
/* training loop                                                           */
/* ---------------------------------------------------------------------- */

int gs_train(gs_gpu *gpu, const gs_dataset *ds, const gs_train_params *p, const char *out_dir,
             gs_train_control *ctl, char *err, size_t err_len)
{
    trainer trs, *tr = &trs;
    gs_model init;
    host_state s;
    int i, step, ret = -1, sh_deg = 0, max_w = 0, max_h = 0;
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
    for (i = 0; i < ds->num_cams; i++) {
        if (ds->cams[i].width > max_w)
            max_w = ds->cams[i].width;
        if (ds->cams[i].height > max_h)
            max_h = ds->cams[i].height;
    }
    if (alloc_image_buffers(tr, max_w, max_h) != 0 || setup_ground_truth(tr, ds) != 0) {
        snprintf(err, err_len, "out of GPU memory (image buffers %dx%d)", max_w, max_h);
        goto done;
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
    if (state_upload(tr, &s) != 0) {
        snprintf(err, err_len, "out of GPU memory (gaussians)");
        goto done;
    }
    state_free(&s);

    printf("[train] %d views, %d gaussians, extent %.3f, %d iterations on %s (%s)\n", ds->num_cams, tr->n,
           ds->extent, p->iterations, gpu->device_name, gs_backend_name(gpu->backend));

    perm = (int *)malloc(sizeof(int) * (size_t)ds->num_cams);
    perm_pos = ds->num_cams;

    for (step = 1; step <= p->iterations; step++) {
        const gs_camera *cam;
        int view;
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
        view = perm[perm_pos++];
        cam = &ds->cams[view];
        if (p->sh_increase_interval > 0 && step % p->sh_increase_interval == 0 && sh_deg < p->sh_degree)
            sh_deg++;

        if (select_ground_truth(tr, ds, view) != 0 || forward(tr, cam, sh_deg, bg) != 0 ||
            loss_and_backward(tr, cam, sh_deg, bg, p->lambda_ssim) != 0) {
            snprintf(err, err_len, "GPU error during iteration %d (%s)", step, gpu->error);
            goto done;
        }

        t = (float)step / (float)p->iterations;
        if (t > 1.f)
            t = 1.f;
        lr_pos = expf(logf(p->lr_pos_init * ds->extent) * (1.f - t) + logf(p->lr_pos_final * ds->extent) * t);
        if (adam_step(tr, P_MEAN, lr_pos, lr_pos, 1, 1, step) ||
            adam_step(tr, P_SCALE, p->lr_scale, p->lr_scale, 1, 1, step) ||
            adam_step(tr, P_ROT, p->lr_rot, p->lr_rot, 1, 1, step) ||
            adam_step(tr, P_OPAC, p->lr_opacity, p->lr_opacity, 1, 1, step) ||
            adam_step(tr, P_SH, p->lr_sh, p->lr_sh / 20.f, GS_SH_STRIDE, 3, step)) {
            snprintf(err, err_len, "GPU error in optimiser step %d (%s)", step, gpu->error);
            goto done;
        }

        if (step < densify_until) {
            if (accumulate_stats(tr, cam->width, cam->height) != 0) {
                snprintf(err, err_len, "GPU error in densification statistics (%s)", gpu->error);
                goto done;
            }
            if (step > p->densify_from && step % p->densify_interval == 0) {
                if (densify_and_prune(tr, p, ds->extent, step > p->opacity_reset_interval, rng + (unsigned)step) != 0) {
                    snprintf(err, err_len, "densification failed (%s)", gpu->error);
                    goto done;
                }
            }
            if (p->opacity_reset_interval > 0 && step % p->opacity_reset_interval == 0) {
                if (reset_opacity(tr) != 0) {
                    snprintf(err, err_len, "opacity reset failed (%s)", gpu->error);
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
/* self tests                                                             */
/* ====================================================================== */

static unsigned st_rng = 12345u;
static unsigned st_rand(void)
{
    st_rng ^= st_rng << 13;
    st_rng ^= st_rng >> 17;
    st_rng ^= st_rng << 5;
    return st_rng;
}

/* scan and radix sort against CPU references */
static int selftest_primitives(trainer *tr)
{
    const int n = 300001;
    int *in = (int *)malloc(sizeof(int) * n), *out = (int *)malloc(sizeof(int) * n);
    unsigned *keys = (unsigned *)malloc(sizeof(unsigned) * n), *kres = (unsigned *)malloc(sizeof(unsigned) * n);
    int *vres = (int *)malloc(sizeof(int) * n), *ref_v = (int *)malloc(sizeof(int) * n);
    gs_buf a, b, c, d;
    int i, bad = 0, which = 0, acc = 0;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&c, 0, sizeof(c));
    memset(&d, 0, sizeof(d));
    for (i = 0; i < n; i++)
        in[i] = (int)(st_rand() % 50);
    if (mkbuf(tr, &a, n * 4) || mkbuf(tr, &b, n * 4) || mkbuf(tr, &c, n * 4) || mkbuf(tr, &d, n * 4))
        return 1;
    G->write(tr->gpu, &a, 0, n * 4, in);
    if (gpu_scan(tr, &a, &b, n, 0) != 0)
        bad++;
    G->read(tr->gpu, &b, 0, n * 4, out);
    for (i = 0; i < n; i++) {
        if (out[i] != acc)
            bad++;
        acc += in[i];
    }
    printf("[selftest] scan      %d elements, %d mismatches\n", n, bad);

    /* sort: 32 bit keys with many duplicates (stability matters) */
    for (i = 0; i < n; i++) {
        keys[i] = (st_rand() % 1000u) * 4000037u;
        in[i] = i;
    }
    G->write(tr->gpu, &a, 0, n * 4, keys);
    G->write(tr->gpu, &b, 0, n * 4, in);
    if (gpu_radix_sort(tr, &a, &b, &c, &d, n, 32, &which) != 0)
        bad++;
    G->read(tr->gpu, which ? &c : &a, 0, n * 4, kres);
    G->read(tr->gpu, which ? &d : &b, 0, n * 4, vres);
    {
        /* reference: stable counting by key value via index order */
        int sbad = 0;
        for (i = 1; i < n; i++)
            if (kres[i - 1] > kres[i] || (kres[i - 1] == kres[i] && vres[i - 1] > vres[i]))
                sbad++;
        for (i = 0; i < n; i++)
            if (vres[i] < 0 || vres[i] >= n || keys[vres[i]] != kres[i])
                sbad++;
        (void)ref_v;
        printf("[selftest] sort      %d keys, %d order/stability errors\n", n, sbad);
        bad += sbad;
    }
    rel(tr, &a);
    rel(tr, &b);
    rel(tr, &c);
    rel(tr, &d);
    free(in);
    free(out);
    free(keys);
    free(kres);
    free(vres);
    free(ref_v);
    return bad ? 1 : 0;
}

/* GPU binning against the straightforward CPU construction */
static int selftest_binning(trainer *tr, const gs_camera *cam, int sh_deg, const float bg[3])
{
    int N = tr->n, W = cam->width, H = cam->height;
    int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE, ntiles = gx * gy;
    int *radii = (int *)malloc(sizeof(int) * N), *ranges = (int *)malloc(sizeof(int) * 2 * ntiles);
    float *m2 = (float *)malloc(sizeof(float) * 2 * N);
    unsigned *dk = (unsigned *)malloc(sizeof(unsigned) * N);
    int *order = (int *)malloc(sizeof(int) * N), *cnt = (int *)calloc((size_t)ntiles + 1, sizeof(int));
    int *gpu_list, *ref, i, t, bad = 0, total = 0;
    forward(tr, cam, sh_deg, bg);
    G->read(tr->gpu, &tr->radii, 0, sizeof(int) * N, radii);
    G->read(tr->gpu, &tr->mean2d, 0, sizeof(float) * 2 * N, m2);
    G->read(tr->gpu, &tr->ranges, 0, sizeof(int) * 2 * ntiles, ranges);
    /* depth keys were sorted in place: recompute from the camera */
    for (i = 0; i < N; i++) {
        float z;
        float mx[3];
        G->read(tr->gpu, &tr->p[P_MEAN], (size_t)i * 12, 12, mx);
        z = cam->R[6] * mx[0] + cam->R[7] * mx[1] + cam->R[8] * mx[2] + cam->t[2];
        memcpy(&dk[i], &z, 4);
        if (radii[i] <= 0)
            dk[i] = 0xFFFFFFFFu;
        order[i] = i;
    }
    /* stable insertion sort by depth (N is small) */
    for (i = 1; i < N; i++) {
        int j = i, o = order[i];
        while (j > 0 && dk[order[j - 1]] > dk[o]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = o;
    }
    for (i = 0; i < N; i++) {
        int id = order[i], x0, y0, x1, y1, x, y;
        float u = m2[2 * id], v = m2[2 * id + 1], r = (float)radii[id];
        if (radii[id] <= 0)
            continue;
        x0 = (int)((u - r) / TILE); x0 = x0 < 0 ? 0 : (x0 > gx ? gx : x0);
        y0 = (int)((v - r) / TILE); y0 = y0 < 0 ? 0 : (y0 > gy ? gy : y0);
        x1 = (int)((u + r + TILE - 1) / TILE); x1 = x1 < 0 ? 0 : (x1 > gx ? gx : x1);
        y1 = (int)((v + r + TILE - 1) / TILE); y1 = y1 < 0 ? 0 : (y1 > gy ? gy : y1);
        for (y = y0; y < y1; y++)
            for (x = x0; x < x1; x++)
                cnt[y * gx + x]++;
        total += (x1 - x0) * (y1 - y0);
    }
    ref = (int *)malloc(sizeof(int) * (size_t)(total + 1));
    {
        int *cur = (int *)malloc(sizeof(int) * (size_t)(ntiles + 1)), acc = 0;
        for (t = 0; t < ntiles; t++) {
            cur[t] = acc;
            if ((cnt[t] ? ranges[2 * t] != acc || ranges[2 * t + 1] != acc + cnt[t] : ranges[2 * t] != ranges[2 * t + 1]))
                bad++;
            acc += cnt[t];
        }
        for (i = 0; i < N; i++) {
            int id = order[i], x0, y0, x1, y1, x, y;
            float u = m2[2 * id], v = m2[2 * id + 1], r = (float)radii[id];
            if (radii[id] <= 0)
                continue;
            x0 = (int)((u - r) / TILE); x0 = x0 < 0 ? 0 : (x0 > gx ? gx : x0);
            y0 = (int)((v - r) / TILE); y0 = y0 < 0 ? 0 : (y0 > gy ? gy : y0);
            x1 = (int)((u + r + TILE - 1) / TILE); x1 = x1 < 0 ? 0 : (x1 > gx ? gx : x1);
            y1 = (int)((v + r + TILE - 1) / TILE); y1 = y1 < 0 ? 0 : (y1 > gy ? gy : y1);
            for (y = y0; y < y1; y++)
                for (x = x0; x < x1; x++)
                    ref[cur[y * gx + x]++] = id;
        }
        free(cur);
    }
    gpu_list = (int *)malloc(sizeof(int) * (size_t)(total + 1));
    if (tr->num_pairs != total)
        bad++;
    else if (total > 0) {
        G->read(tr->gpu, tr->plist, 0, sizeof(int) * (size_t)total, gpu_list);
        for (i = 0; i < total; i++)
            if (gpu_list[i] != ref[i])
                bad++;
    }
    printf("[selftest] binning   %d gaussians, %d pairs (GPU %d), %d mismatches\n", N, total, tr->num_pairs, bad);
    free(radii);
    free(ranges);
    free(m2);
    free(dk);
    free(order);
    free(cnt);
    free(ref);
    free(gpu_list);
    return bad ? 1 : 0;
}

/* GPU densification against the reference rules */
static int selftest_densify(trainer *tr, host_state *s)
{
    gs_train_params p;
    const float extent = 10.f; /* clone below 0.1, split above, prune above 1.0 */
    int n = s->n, i, k, bad = 0, expect = 0, o;
    float *accum = (float *)malloc(sizeof(float) * n), *denom = (float *)malloc(sizeof(float) * n);
    float *maxr = (float *)malloc(sizeof(float) * n);
    int *flag = (int *)malloc(sizeof(int) * n);
    host_state out;
    gs_train_params_default(&p);
    memset(&out, 0, sizeof(out));
    for (i = 0; i < n; i++) {
        /* a third of the gaussians gets a large gradient, a few are huge or transparent */
        accum[i] = (i % 3 == 0) ? 0.01f : 0.f;
        denom[i] = 10.f;
        maxr[i] = (i % 7 == 0) ? 50.f : 5.f;
        for (k = 0; k < 3; k++)
            s->a[0][P_SCALE][3 * i + k] = logf((i % 2) ? 0.05f : (i % 4 == 0 ? 0.5f : 0.08f));
        s->a[0][P_OPAC][i] = (i % 11 == 5) ? -8.f : 0.5f;
        for (k = 0; k < GS_SH_STRIDE; k++)
            s->a[1][P_SH][(size_t)i * GS_SH_STRIDE + k] = 0.25f; /* Adam moments must survive for kept ones */
    }
    state_upload(tr, s);
    G->write(tr->gpu, &tr->accum, 0, sizeof(float) * n, accum);
    G->write(tr->gpu, &tr->denom, 0, sizeof(float) * n, denom);
    G->write(tr->gpu, &tr->maxr, 0, sizeof(float) * n, maxr);
    /* reference decision */
    for (i = 0; i < n; i++) {
        float ms = expf(fmaxf(fmaxf(s->a[0][P_SCALE][3 * i], s->a[0][P_SCALE][3 * i + 1]), s->a[0][P_SCALE][3 * i + 2]));
        float g = accum[i] / denom[i], op = gs_sigmoid(s->a[0][P_OPAC][i]);
        int opok = op >= 0.005f, keep = opok && !(maxr[i] > 20.f || ms > 0.1f * extent);
        int f;
        if (g >= p.densify_grad_threshold) {
            if (ms <= p.percent_dense * extent)
                f = (keep ? 1 : 0) | ((opok && !(ms > 0.1f * extent)) ? 2 : 0);
            else
                f = (opok && !(ms / 1.6f > 0.1f * extent)) ? 4 : 0;
        } else {
            f = keep ? 1 : 0;
        }
        flag[i] = f;
        expect += ((f & 1) ? 1 : 0) + ((f & 2) ? 1 : 0) + ((f & 4) ? 2 : 0);
    }
    if (densify_and_prune(tr, &p, extent, 1, 99u) != 0)
        bad++;
    if (tr->n != expect)
        bad++;
    else {
        state_download(tr, &out, 1);
        o = 0;
        for (i = 0; i < n; i++) {
            if (flag[i] & 1) { /* exact copy with moments */
                for (k = 0; k < 3; k++)
                    if (out.a[0][P_MEAN][3 * o + k] != s->a[0][P_MEAN][3 * i + k])
                        bad++;
                if (out.a[1][P_SH][(size_t)o * GS_SH_STRIDE] != 0.25f)
                    bad++;
                o++;
            }
            if (flag[i] & 2) { /* clone: same values, fresh moments */
                if (out.a[0][P_OPAC][o] != s->a[0][P_OPAC][i] || out.a[1][P_SH][(size_t)o * GS_SH_STRIDE] != 0.f)
                    bad++;
                o++;
            }
            if (flag[i] & 4) { /* split: scale / 1.6, position near the parent */
                int c;
                for (c = 0; c < 2; c++, o++) {
                    float want = logf(expf(s->a[0][P_SCALE][3 * i]) / 1.6f), d2 = 0.f;
                    if (fabsf(out.a[0][P_SCALE][3 * o] - want) > 1e-4f)
                        bad++;
                    for (k = 0; k < 3; k++) {
                        float dd = out.a[0][P_MEAN][3 * o + k] - s->a[0][P_MEAN][3 * i + k];
                        d2 += dd * dd;
                    }
                    if (!(d2 < 2.25f)) /* parent scale <= 0.5: within 3 sigma */
                        bad++;
                }
            }
        }
    }
    printf("[selftest] densify   %d -> %d gaussians (expected %d), %d mismatches\n", n, tr->n, expect, bad);
    state_free(&out);
    free(accum);
    free(denom);
    free(maxr);
    free(flag);
    return bad ? 1 : 0;
}

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
    if (alloc_image_buffers(tr, cam.width, cam.height) != 0 || mkbuf(tr, &tr->gt, (size_t)cam.width * cam.height * 4))
        return -1;
    G->write(tr->gpu, &tr->gt, 0, (size_t)cam.width * cam.height * 4, cam.rgba);
    tr->cur_gt_off = 0;

    failures += selftest_primitives(tr);

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
    state_upload(tr, &s);
    failures += selftest_binning(tr, &cam, sh_deg, bg);

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
    failures += selftest_densify(tr, &s);
    state_free(&s);
    free(cam.rgba);
    trainer_release(tr);
    printf("[selftest] %s (%d gradients checked)\n", failures ? "FAILED" : "PASSED", checked);
    return failures ? 1 : 0;
}
