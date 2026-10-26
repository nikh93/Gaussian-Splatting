/*
 * gsplat.cl - OpenCL kernels for differentiable 3D Gaussian Splatting.
 *
 * Follows the tile based rasteriser of Kerbl et al. 2023 ("3D Gaussian
 * Splatting for Real-Time Radiance Field Rendering"):
 *   preprocess      project 3D gaussians to 2D conics, evaluate SH colour
 *   render_fwd      alpha-composite depth sorted splats per 16x16 tile
 *   ssim_* / loss   0.8 * L1 + 0.2 * (1 - SSIM) and its image gradient
 *   render_bwd      back-propagate through the compositing
 *   preprocess_bwd  back-propagate through projection / covariance / SH
 *   adam            optimiser step
 *
 * Camera convention (OpenCV): x right, y down, z forward. Pixel coordinates
 * of a point are u = fx * x / z + cx, v = fy * y / z + cy.
 *
 * Parameter layout (structure of arrays, N gaussians):
 *   means [3N], log_scales [3N], rots [4N] (w,x,y,z), opacity logits [N],
 *   sh [48N] (16 coefficients x RGB, coefficient major).
 */

#ifndef TILE
#define TILE 16
#endif
#define BLOCK (TILE * TILE)
#define SH_STRIDE 48

#ifdef GS_FULL_TILES
#define ALPHA_MIN 0.f /* testing only: makes the loss smooth for finite differences */
#else
#define ALPHA_MIN (1.f / 255.f)
#endif

/* camera constant buffer layout */
#define CAM_R 0        /* 9 floats, world->camera rotation, row major */
#define CAM_T 9        /* 3 floats */
#define CAM_POS 12     /* 3 floats, camera centre */
#define CAM_FX 15
#define CAM_FY 16
#define CAM_CX 17
#define CAM_CY 18
#define CAM_W 19
#define CAM_H 20
#define CAM_SIZE 24

__constant float SH_C0 = 0.28209479177387814f;
__constant float SH_C1 = 0.4886025119029199f;
__constant float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                             -1.0925484305920792f, 0.5462742152960396f};
__constant float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                             0.3731763325901154f,  -0.4570457994644658f, 1.445305721320277f,
                             -0.5900435899266435f};

inline void atomic_add_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u;
        float f;
    } prev, next, cur;
    cur.f = *addr;
    do {
        prev.f = cur.f;
        next.f = prev.f + val;
        cur.u = atomic_cmpxchg((volatile __global unsigned int *)addr, prev.u, next.u);
    } while (cur.u != prev.u);
}

/* SH basis for a unit direction; returns the number of coefficients written. */
inline int sh_basis(int deg, float3 d, float *b)
{
    float x = d.x, y = d.y, z = d.z;
    b[0] = SH_C0;
    if (deg < 1)
        return 1;
    b[1] = -SH_C1 * y;
    b[2] = SH_C1 * z;
    b[3] = -SH_C1 * x;
    if (deg < 2)
        return 4;
    {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        b[4] = SH_C2[0] * xy;
        b[5] = SH_C2[1] * yz;
        b[6] = SH_C2[2] * (2.0f * zz - xx - yy);
        b[7] = SH_C2[3] * xz;
        b[8] = SH_C2[4] * (xx - yy);
        if (deg < 3)
            return 9;
        b[9] = SH_C3[0] * y * (3.0f * xx - yy);
        b[10] = SH_C3[1] * xy * z;
        b[11] = SH_C3[2] * y * (4.0f * zz - xx - yy);
        b[12] = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        b[13] = SH_C3[4] * x * (4.0f * zz - xx - yy);
        b[14] = SH_C3[5] * z * (xx - yy);
        b[15] = SH_C3[6] * x * (xx - 3.0f * yy);
    }
    return 16;
}

/* Rotation matrix (row major) from a normalised quaternion (w,x,y,z). */
inline void quat_to_rot(float4 q, float *R)
{
    float w = q.x, x = q.y, y = q.z, z = q.w;
    R[0] = 1.f - 2.f * (y * y + z * z);
    R[1] = 2.f * (x * y - w * z);
    R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z);
    R[4] = 1.f - 2.f * (x * x + z * z);
    R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y);
    R[7] = 2.f * (y * z + w * x);
    R[8] = 1.f - 2.f * (x * x + y * y);
}

/* Shared forward geometry of one gaussian. */
typedef struct {
    float3 tc;     /* camera space centre */
    float txc, tyc; /* clamped x, y used for the Jacobian */
    int clamp_x, clamp_y;
    float Rq[9];   /* rotation from quaternion */
    float3 s;      /* activated scales */
    float Sig[9];  /* 3D covariance (full symmetric) */
    float T[6];    /* J * W (2x3) */
    float a, b, c; /* 2D covariance incl. low pass */
} geom;

inline int compute_geom(int i, __global const float *means, __global const float *log_scales,
                        __global const float *rots, __constant float *cam, geom *g, float4 *qn_out,
                        float *qlen_out)
{
    __constant float *W = cam + CAM_R;
    float3 m = (float3)(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    float fx = cam[CAM_FX], fy = cam[CAM_FY];
    float limx = 1.3f * 0.5f * cam[CAM_W] / fx;
    float limy = 1.3f * 0.5f * cam[CAM_H] / fy;
    float4 q;
    float qlen;
    float M[9];
    float J00, J02, J11, J12;
    float TS[6];
    int r, k;

    g->tc.x = W[0] * m.x + W[1] * m.y + W[2] * m.z + cam[CAM_T + 0];
    g->tc.y = W[3] * m.x + W[4] * m.y + W[5] * m.z + cam[CAM_T + 1];
    g->tc.z = W[6] * m.x + W[7] * m.y + W[8] * m.z + cam[CAM_T + 2];
    if (g->tc.z <= 0.2f)
        return 0;

    {
        float xz = g->tc.x / g->tc.z, yz = g->tc.y / g->tc.z;
        g->clamp_x = xz < -limx || xz > limx;
        g->clamp_y = yz < -limy || yz > limy;
        g->txc = clamp(xz, -limx, limx) * g->tc.z;
        g->tyc = clamp(yz, -limy, limy) * g->tc.z;
    }

    q = (float4)(rots[4 * i], rots[4 * i + 1], rots[4 * i + 2], rots[4 * i + 3]);
    qlen = sqrt(dot(q, q));
    if (qlen < 1e-12f) {
        q = (float4)(1.f, 0.f, 0.f, 0.f);
        qlen = 1.f;
    } else {
        q /= qlen;
    }
    *qn_out = q;
    *qlen_out = qlen;
    quat_to_rot(q, g->Rq);
    g->s = exp((float3)(log_scales[3 * i], log_scales[3 * i + 1], log_scales[3 * i + 2]));

    /* Sigma = M M^T, M = Rq diag(s) */
    for (r = 0; r < 3; r++) {
        M[r * 3 + 0] = g->Rq[r * 3 + 0] * g->s.x;
        M[r * 3 + 1] = g->Rq[r * 3 + 1] * g->s.y;
        M[r * 3 + 2] = g->Rq[r * 3 + 2] * g->s.z;
    }
    for (r = 0; r < 3; r++)
        for (k = 0; k < 3; k++)
            g->Sig[r * 3 + k] = M[r * 3] * M[k * 3] + M[r * 3 + 1] * M[k * 3 + 1] + M[r * 3 + 2] * M[k * 3 + 2];

    {
        float z = g->tc.z, z2 = z * z;
        J00 = fx / z;
        J02 = -fx * g->txc / z2;
        J11 = fy / z;
        J12 = -fy * g->tyc / z2;
    }
    for (k = 0; k < 3; k++) {
        g->T[k] = J00 * W[k] + J02 * W[6 + k];
        g->T[3 + k] = J11 * W[3 + k] + J12 * W[6 + k];
    }
    for (r = 0; r < 2; r++)
        for (k = 0; k < 3; k++)
            TS[r * 3 + k] = g->T[r * 3] * g->Sig[k] + g->T[r * 3 + 1] * g->Sig[3 + k] + g->T[r * 3 + 2] * g->Sig[6 + k];
    g->a = TS[0] * g->T[0] + TS[1] * g->T[1] + TS[2] * g->T[2] + 0.3f;
    g->b = TS[0] * g->T[3] + TS[1] * g->T[4] + TS[2] * g->T[5];
    g->c = TS[3] * g->T[3] + TS[4] * g->T[4] + TS[5] * g->T[5] + 0.3f;
    return 1;
}

/* ===================================================================== */
/* forward                                                               */
/* ===================================================================== */

__kernel void preprocess(int N, __global const float *means, __global const float *log_scales,
                         __global const float *rots, __global const float *opac,
                         __global const float *sh, int sh_deg, __constant float *cam,
                         __global float2 *mean2d, __global float *depths, __global int *radii,
                         __global float4 *conic_opac, __global float *colors)
{
    int i = get_global_id(0);
    geom g;
    float4 qn;
    float qlen, det, mid, lambda1, radius;
    float2 uv;
    int gx, gy;
    if (i >= N)
        return;
    radii[i] = 0;
    if (!compute_geom(i, means, log_scales, rots, cam, &g, &qn, &qlen))
        return;
    det = g.a * g.c - g.b * g.b;
    if (det <= 0.f)
        return;
    mid = 0.5f * (g.a + g.c);
    lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    radius = ceil(3.f * sqrt(lambda1));
#ifdef GS_FULL_TILES
    radius = 4096.f; /* testing only: every gaussian touches every tile */
#endif
    uv.x = cam[CAM_FX] * g.tc.x / g.tc.z + cam[CAM_CX];
    uv.y = cam[CAM_FY] * g.tc.y / g.tc.z + cam[CAM_CY];

    gx = ((int)cam[CAM_W] + TILE - 1) / TILE;
    gy = ((int)cam[CAM_H] + TILE - 1) / TILE;
    {
        int x0 = min(gx, max(0, (int)((uv.x - radius) / TILE)));
        int y0 = min(gy, max(0, (int)((uv.y - radius) / TILE)));
        int x1 = min(gx, max(0, (int)((uv.x + radius + TILE - 1) / TILE)));
        int y1 = min(gy, max(0, (int)((uv.y + radius + TILE - 1) / TILE)));
        if ((x1 - x0) * (y1 - y0) == 0)
            return;
    }

    {
        float3 m = (float3)(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
        float3 cp = (float3)(cam[CAM_POS], cam[CAM_POS + 1], cam[CAM_POS + 2]);
        float3 dir = normalize(m - cp);
        float basis[16];
        int n = sh_basis(sh_deg, dir, basis), k;
        float3 col = (float3)(0.5f, 0.5f, 0.5f);
        __global const float *f = sh + (size_t)i * SH_STRIDE;
        for (k = 0; k < n; k++)
            col += basis[k] * (float3)(f[3 * k], f[3 * k + 1], f[3 * k + 2]);
        col = max(col, 0.f);
        colors[3 * i] = col.x;
        colors[3 * i + 1] = col.y;
        colors[3 * i + 2] = col.z;
    }

    mean2d[i] = uv;
    depths[i] = g.tc.z;
    radii[i] = (int)radius;
    conic_opac[i] = (float4)(g.c / det, -g.b / det, g.a / det, 1.f / (1.f + exp(-opac[i])));
}

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void render_fwd(__global const int2 *ranges, __global const int *point_list, int W, int H,
                __global const float2 *mean2d, __global const float4 *conic_opac,
                __global const float *colors, float bg_r, float bg_g, float bg_b,
                __global float *out_img, __global float *final_T, __global int *n_contrib)
{
    __local int l_id[BLOCK];
    __local float2 l_xy[BLOCK];
    __local float4 l_co[BLOCK];
    __local int l_done;
    int tiles_x = (W + TILE - 1) / TILE;
    int tile = get_group_id(1) * tiles_x + get_group_id(0);
    int px = get_group_id(0) * TILE + get_local_id(0);
    int py = get_group_id(1) * TILE + get_local_id(1);
    int tid = get_local_id(1) * TILE + get_local_id(0);
    int inside = px < W && py < H;
    int done = !inside;
    int2 range = ranges[tile];
    int todo = range.y - range.x;
    int rounds = (todo + BLOCK - 1) / BLOCK;
    float T = 1.f;
    float3 C = (float3)(0.f, 0.f, 0.f);
    int contributor = 0, last_contributor = 0;
    float2 pf = (float2)((float)px, (float)py);
    int r, j;

    for (r = 0; r < rounds; r++, todo -= BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid == 0)
            l_done = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (done)
            atomic_inc(&l_done);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_done == BLOCK)
            break;
        {
            int progress = range.x + r * BLOCK + tid;
            if (progress < range.y) {
                int id = point_list[progress];
                l_id[tid] = id;
                l_xy[tid] = mean2d[id];
                l_co[tid] = conic_opac[id];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (j = 0; !done && j < min(BLOCK, todo); j++) {
            float2 d = l_xy[j] - pf;
            float4 co = l_co[j];
            float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            float alpha, test_T;
            int id;
            contributor++;
            if (power > 0.f)
                continue;
            alpha = min(0.99f, co.w * exp(power));
            if (alpha < ALPHA_MIN)
                continue;
            test_T = T * (1.f - alpha);
            if (test_T < 0.0001f) {
                done = 1;
                continue;
            }
            id = l_id[j];
            C += (float3)(colors[3 * id], colors[3 * id + 1], colors[3 * id + 2]) * (alpha * T);
            T = test_T;
            last_contributor = contributor;
        }
    }
    if (inside) {
        int p = py * W + px;
        int hw = W * H;
        final_T[p] = T;
        n_contrib[p] = last_contributor;
        out_img[p] = C.x + T * bg_r;
        out_img[hw + p] = C.y + T * bg_g;
        out_img[2 * hw + p] = C.z + T * bg_b;
    }
}

/* ===================================================================== */
/* loss: 0.8 * L1 + 0.2 * (1 - SSIM), 11x11 gaussian window, sigma 1.5    */
/* ===================================================================== */

__constant float GW[11] = {0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f, 0.21300553f, 0.26601172f,
                           0.21300553f, 0.10936069f, 0.03600077f, 0.00759876f, 0.00102838f};

inline float gt_channel(__global const uchar4 *gt, int p, int c)
{
    uchar4 v = gt[p];
    return (c == 0 ? v.x : (c == 1 ? v.y : v.z)) * (1.f / 255.f);
}

/* horizontal blur of x, y, x^2, y^2, xy  -> tmp[5][3][HW] */
__kernel void ssim_h(int W, int H, __global const float *img, __global const uchar4 *gt, __global float *tmp)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    int hw = W * H, k;
    float s[5] = {0, 0, 0, 0, 0};
    if (x >= W || y >= H)
        return;
    for (k = -5; k <= 5; k++) {
        int xx = x + k;
        float w, a, b;
        if (xx < 0 || xx >= W)
            continue;
        w = GW[k + 5];
        a = img[c * hw + y * W + xx];
        b = gt_channel(gt, y * W + xx, c);
        s[0] += w * a;
        s[1] += w * b;
        s[2] += w * a * a;
        s[3] += w * b * b;
        s[4] += w * a * b;
    }
    for (k = 0; k < 5; k++)
        tmp[(k * 3 + c) * hw + y * W + x] = s[k];
}

/* vertical blur, SSIM and its partial derivatives -> efh[3][3][HW], ssim_map[3][HW] */
__kernel void ssim_v(int W, int H, __global const float *tmp, __global float *efh, __global float *ssim_map)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    int hw = W * H, k;
    float s[5] = {0, 0, 0, 0, 0};
    const float C1 = 0.0001f, C2 = 0.0009f;
    float mx, my, sxx, syy, sxy, A1, A2, B1, B2, f;
    int p = y * W + x;
    if (x >= W || y >= H)
        return;
    for (k = -5; k <= 5; k++) {
        int yy = y + k, q;
        float w;
        if (yy < 0 || yy >= H)
            continue;
        w = GW[k + 5];
        for (q = 0; q < 5; q++)
            s[q] += w * tmp[(q * 3 + c) * hw + yy * W + x];
    }
    mx = s[0];
    my = s[1];
    sxx = s[2] - mx * mx;
    syy = s[3] - my * my;
    sxy = s[4] - mx * my;
    A1 = 2.f * mx * my + C1;
    A2 = 2.f * sxy + C2;
    B1 = mx * mx + my * my + C1;
    B2 = sxx + syy + C2;
    f = (A1 * A2) / (B1 * B2);
    {
        float d_mx = (2.f * my * A2) / (B1 * B2) - f * 2.f * mx / B1;
        float d_sxx = -f / B2;
        float d_sxy = 2.f * A1 / (B1 * B2);
        efh[(0 * 3 + c) * hw + p] = d_mx - 2.f * mx * d_sxx - my * d_sxy;
        efh[(1 * 3 + c) * hw + p] = d_sxx;
        efh[(2 * 3 + c) * hw + p] = d_sxy;
    }
    ssim_map[c * hw + p] = f;
}

/* horizontal blur of the E, F, H maps -> tmp[3][3][HW] */
__kernel void ssim_bh(int W, int H, __global const float *efh, __global float *tmp)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    int hw = W * H, k, q;
    float s[3] = {0, 0, 0};
    if (x >= W || y >= H)
        return;
    for (k = -5; k <= 5; k++) {
        int xx = x + k;
        if (xx < 0 || xx >= W)
            continue;
        for (q = 0; q < 3; q++)
            s[q] += GW[k + 5] * efh[(q * 3 + c) * hw + y * W + xx];
    }
    for (q = 0; q < 3; q++)
        tmp[(q * 3 + c) * hw + y * W + x] = s[q];
}

/* vertical blur + final gradient of the loss w.r.t. the rendered image */
__kernel void loss_grad(int W, int H, float lambda_ssim, __global const float *img, __global const uchar4 *gt,
                        __global const float *tmp, __global float *dL_dimg)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    int hw = W * H, k, q;
    int p = y * W + x;
    float s[3] = {0, 0, 0};
    float a, b, n_inv, g_l1, g_ssim;
    if (x >= W || y >= H)
        return;
    for (k = -5; k <= 5; k++) {
        int yy = y + k;
        if (yy < 0 || yy >= H)
            continue;
        for (q = 0; q < 3; q++)
            s[q] += GW[k + 5] * tmp[(q * 3 + c) * hw + yy * W + x];
    }
    a = img[c * hw + p];
    b = gt_channel(gt, p, c);
    n_inv = 1.f / (3.f * (float)hw);
    g_l1 = a > b ? 1.f : (a < b ? -1.f : 0.f);
    g_ssim = s[0] + 2.f * a * s[1] + b * s[2];
    dL_dimg[c * hw + p] = ((1.f - lambda_ssim) * g_l1 - lambda_ssim * g_ssim) * n_inv;
}

/* ===================================================================== */
/* backward                                                              */
/* ===================================================================== */

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void render_bwd(__global const int2 *ranges, __global const int *point_list, int W, int H,
                __global const float2 *mean2d, __global const float4 *conic_opac,
                __global const float *colors, float bg_r, float bg_g, float bg_b,
                __global const float *final_Ts, __global const int *n_contrib,
                __global const float *dL_dimg, __global float *dL_dmean2d, __global float *dL_dconic,
                __global float *dL_dopa, __global float *dL_dcolor)
{
    __local int l_id[BLOCK];
    __local float2 l_xy[BLOCK];
    __local float4 l_co[BLOCK];
    __local float3 l_col[BLOCK];
    int tiles_x = (W + TILE - 1) / TILE;
    int tile = get_group_id(1) * tiles_x + get_group_id(0);
    int px = get_group_id(0) * TILE + get_local_id(0);
    int py = get_group_id(1) * TILE + get_local_id(1);
    int tid = get_local_id(1) * TILE + get_local_id(0);
    int inside = px < W && py < H;
    int done = !inside;
    int2 range = ranges[tile];
    int todo = range.y - range.x;
    int rounds = (todo + BLOCK - 1) / BLOCK;
    int hw = W * H;
    int p = inside ? py * W + px : 0;
    float T_final = inside ? final_Ts[p] : 0.f;
    float T = T_final;
    int contributor = todo;
    int last_contributor = inside ? n_contrib[p] : 0;
    float3 dL_dpix = (float3)(0.f, 0.f, 0.f);
    float3 accum = (float3)(0.f, 0.f, 0.f);
    float3 last_color = (float3)(0.f, 0.f, 0.f);
    float last_alpha = 0.f;
    float bg_dot;
    float2 pf = (float2)((float)px, (float)py);
    int r, j;

    if (inside)
        dL_dpix = (float3)(dL_dimg[p], dL_dimg[hw + p], dL_dimg[2 * hw + p]);
    bg_dot = bg_r * dL_dpix.x + bg_g * dL_dpix.y + bg_b * dL_dpix.z;

    for (r = 0; r < rounds; r++, todo -= BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            int progress = r * BLOCK + tid;
            if (range.x + progress < range.y) {
                int id = point_list[range.y - progress - 1];
                l_id[tid] = id;
                l_xy[tid] = mean2d[id];
                l_co[tid] = conic_opac[id];
                l_col[tid] = (float3)(colors[3 * id], colors[3 * id + 1], colors[3 * id + 2]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (j = 0; !done && j < min(BLOCK, todo); j++) {
            float2 d;
            float4 co;
            float power, G, alpha, dL_dalpha, dL_dG, gdx, gdy;
            float3 col;
            int id;
            contributor--;
            if (contributor >= last_contributor)
                continue;
            d = l_xy[j] - pf;
            co = l_co[j];
            power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.f)
                continue;
            G = exp(power);
            alpha = min(0.99f, co.w * G);
            if (alpha < ALPHA_MIN)
                continue;
            T = T / (1.f - alpha);
            col = l_col[j];
            id = l_id[j];
            accum = last_alpha * last_color + (1.f - last_alpha) * accum;
            last_color = col;
            dL_dalpha = dot(col - accum, dL_dpix) * T;
            last_alpha = alpha;
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot;
            {
                float w = alpha * T;
                atomic_add_f(&dL_dcolor[3 * id], w * dL_dpix.x);
                atomic_add_f(&dL_dcolor[3 * id + 1], w * dL_dpix.y);
                atomic_add_f(&dL_dcolor[3 * id + 2], w * dL_dpix.z);
            }
            dL_dG = co.w * dL_dalpha;
            gdx = G * d.x;
            gdy = G * d.y;
            atomic_add_f(&dL_dmean2d[2 * id], dL_dG * (-gdx * co.x - gdy * co.y));
            atomic_add_f(&dL_dmean2d[2 * id + 1], dL_dG * (-gdy * co.z - gdx * co.y));
            atomic_add_f(&dL_dconic[3 * id], -0.5f * gdx * d.x * dL_dG);
            atomic_add_f(&dL_dconic[3 * id + 1], -gdx * d.y * dL_dG);
            atomic_add_f(&dL_dconic[3 * id + 2], -0.5f * gdy * d.y * dL_dG);
            atomic_add_f(&dL_dopa[id], G * dL_dalpha);
        }
    }
}

/* derivative of the rotation matrix (row major) w.r.t. one quaternion component */
inline float quat_grad_dot(const float *dR, float4 q, int comp)
{
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float D[9];
    int k;
    float s = 0.f;
    if (comp == 0) {
        D[0] = 0;        D[1] = -2 * z;   D[2] = 2 * y;
        D[3] = 2 * z;    D[4] = 0;        D[5] = -2 * x;
        D[6] = -2 * y;   D[7] = 2 * x;    D[8] = 0;
    } else if (comp == 1) {
        D[0] = 0;        D[1] = 2 * y;    D[2] = 2 * z;
        D[3] = 2 * y;    D[4] = -4 * x;   D[5] = -2 * w;
        D[6] = 2 * z;    D[7] = 2 * w;    D[8] = -4 * x;
    } else if (comp == 2) {
        D[0] = -4 * y;   D[1] = 2 * x;    D[2] = 2 * w;
        D[3] = 2 * x;    D[4] = 0;        D[5] = 2 * z;
        D[6] = -2 * w;   D[7] = 2 * z;    D[8] = -4 * y;
    } else {
        D[0] = -4 * z;   D[1] = -2 * w;   D[2] = 2 * x;
        D[3] = 2 * w;    D[4] = -4 * z;   D[5] = 2 * y;
        D[6] = 2 * x;    D[7] = 2 * y;    D[8] = 0;
    }
    for (k = 0; k < 9; k++)
        s += dR[k] * D[k];
    return s;
}

__kernel void preprocess_bwd(int N, __global const float *means, __global const float *log_scales,
                             __global const float *rots, __global const float *opac,
                             __global const float *sh, int sh_deg, __constant float *cam,
                             __global const int *radii, __global const float *dL_dmean2d,
                             __global const float *dL_dconic, __global const float *dL_dopa,
                             __global const float *dL_dcolor, __global float *g_means,
                             __global float *g_scales, __global float *g_rots, __global float *g_opac,
                             __global float *g_sh)
{
    int i = get_global_id(0);
    geom g;
    float4 qn;
    float qlen;
    __global float *gsh;
    __constant float *W = cam + CAM_R;
    int k, r;
    if (i >= N)
        return;
    gsh = g_sh + (size_t)i * SH_STRIDE;
    if (radii[i] <= 0 || !compute_geom(i, means, log_scales, rots, cam, &g, &qn, &qlen)) {
        for (k = 0; k < 3; k++) {
            g_means[3 * i + k] = 0.f;
            g_scales[3 * i + k] = 0.f;
        }
        for (k = 0; k < 4; k++)
            g_rots[4 * i + k] = 0.f;
        g_opac[i] = 0.f;
        for (k = 0; k < SH_STRIDE; k++)
            gsh[k] = 0.f;
        return;
    }

    /* --- opacity ------------------------------------------------------ */
    {
        float sig = 1.f / (1.f + exp(-opac[i]));
        g_opac[i] = dL_dopa[i] * sig * (1.f - sig);
    }

    /* --- colour / SH (direction gradient is ignored) -------------------- */
    {
        float3 m = (float3)(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
        float3 cp = (float3)(cam[CAM_POS], cam[CAM_POS + 1], cam[CAM_POS + 2]);
        float3 dir = normalize(m - cp);
        float basis[16];
        int n = sh_basis(sh_deg, dir, basis);
        __global const float *f = sh + (size_t)i * SH_STRIDE;
        float3 raw = (float3)(0.5f, 0.5f, 0.5f);
        float3 dc = (float3)(dL_dcolor[3 * i], dL_dcolor[3 * i + 1], dL_dcolor[3 * i + 2]);
        for (k = 0; k < n; k++)
            raw += basis[k] * (float3)(f[3 * k], f[3 * k + 1], f[3 * k + 2]);
        if (raw.x < 0.f) dc.x = 0.f;
        if (raw.y < 0.f) dc.y = 0.f;
        if (raw.z < 0.f) dc.z = 0.f;
        for (k = 0; k < 16; k++) {
            float bk = k < n ? basis[k] : 0.f;
            gsh[3 * k] = bk * dc.x;
            gsh[3 * k + 1] = bk * dc.y;
            gsh[3 * k + 2] = bk * dc.z;
        }
    }

    {
        float fx = cam[CAM_FX], fy = cam[CAM_FY];
        float z = g.tc.z, z2 = z * z, z3 = z2 * z;
        float a = g.a, b = g.b, c = g.c;
        float det = a * c - b * b;
        float gA = dL_dconic[3 * i], gB = dL_dconic[3 * i + 1], gC = dL_dconic[3 * i + 2];
        float ga = 0.f, gb = 0.f, gc = 0.f;
        float Gs[4];
        float dSig[9];
        float dT[6];
        float3 dt;
        float gu = dL_dmean2d[2 * i], gv = dL_dmean2d[2 * i + 1];

        if (det != 0.f) {
            float inv = 1.f / (det * det);
            ga = inv * (-c * c * gA + b * c * gB - b * b * gC);
            gb = inv * (2.f * b * c * gA - (a * c + b * b) * gB + 2.f * a * b * gC);
            gc = inv * (-b * b * gA + a * b * gB - a * a * gC);
        }
        Gs[0] = ga;
        Gs[1] = 0.5f * gb;
        Gs[2] = 0.5f * gb;
        Gs[3] = gc;

        /* dL/dSigma = T^T Gs T */
        for (r = 0; r < 3; r++)
            for (k = 0; k < 3; k++) {
                float t0r = g.T[r], t1r = g.T[3 + r], t0k = g.T[k], t1k = g.T[3 + k];
                dSig[r * 3 + k] = Gs[0] * t0r * t0k + Gs[1] * t0r * t1k + Gs[2] * t1r * t0k + Gs[3] * t1r * t1k;
            }

        /* dL/dT = 2 Gs T Sigma */
        {
            float GT[6];
            for (k = 0; k < 3; k++) {
                GT[k] = Gs[0] * g.T[k] + Gs[1] * g.T[3 + k];
                GT[3 + k] = Gs[2] * g.T[k] + Gs[3] * g.T[3 + k];
            }
            for (r = 0; r < 2; r++)
                for (k = 0; k < 3; k++)
                    dT[r * 3 + k] = 2.f * (GT[r * 3] * g.Sig[k] + GT[r * 3 + 1] * g.Sig[3 + k] + GT[r * 3 + 2] * g.Sig[6 + k]);
        }

        /* T = J W  ->  dL/dJ */
        {
            float gJ00 = dT[0] * W[0] + dT[1] * W[1] + dT[2] * W[2];
            float gJ02 = dT[0] * W[6] + dT[1] * W[7] + dT[2] * W[8];
            float gJ11 = dT[3] * W[3] + dT[4] * W[4] + dT[5] * W[5];
            float gJ12 = dT[3] * W[6] + dT[4] * W[7] + dT[5] * W[8];
            dt.x = g.clamp_x ? 0.f : gJ02 * (-fx / z2);
            dt.y = g.clamp_y ? 0.f : gJ12 * (-fy / z2);
            dt.z = gJ00 * (-fx / z2) + gJ02 * (2.f * fx * g.txc / z3) + gJ11 * (-fy / z2) +
                   gJ12 * (2.f * fy * g.tyc / z3);
        }

        /* projected mean */
        dt.x += gu * fx / z;
        dt.y += gv * fy / z;
        dt.z += -(gu * fx * g.tc.x + gv * fy * g.tc.y) / z2;

        g_means[3 * i] = W[0] * dt.x + W[3] * dt.y + W[6] * dt.z;
        g_means[3 * i + 1] = W[1] * dt.x + W[4] * dt.y + W[7] * dt.z;
        g_means[3 * i + 2] = W[2] * dt.x + W[5] * dt.y + W[8] * dt.z;

        /* Sigma = M M^T, M = Rq diag(s):  dL/dM = 2 dSig M */
        {
            float M[9], dM[9], dR[9];
            float sv[3] = {g.s.x, g.s.y, g.s.z};
            float4 gq;
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    M[r * 3 + k] = g.Rq[r * 3 + k] * sv[k];
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    dM[r * 3 + k] = 2.f * (dSig[r * 3] * M[k] + dSig[r * 3 + 1] * M[3 + k] + dSig[r * 3 + 2] * M[6 + k]);
            for (k = 0; k < 3; k++) {
                float ds = dM[k] * g.Rq[k] + dM[3 + k] * g.Rq[3 + k] + dM[6 + k] * g.Rq[6 + k];
                g_scales[3 * i + k] = ds * sv[k];
            }
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    dR[r * 3 + k] = dM[r * 3 + k] * sv[k];
            gq.x = quat_grad_dot(dR, qn, 0);
            gq.y = quat_grad_dot(dR, qn, 1);
            gq.z = quat_grad_dot(dR, qn, 2);
            gq.w = quat_grad_dot(dR, qn, 3);
            /* through the normalisation q / |q| */
            gq = (gq - qn * dot(qn, gq)) / qlen;
            g_rots[4 * i] = gq.x;
            g_rots[4 * i + 1] = gq.y;
            g_rots[4 * i + 2] = gq.z;
            g_rots[4 * i + 3] = gq.w;
        }
    }
}

/* ===================================================================== */
/* optimiser + densification statistics                                 */
/* ===================================================================== */

/* Elements whose index modulo `group` is below `first` use lr1, others lr2
 * (used for the SH buffer: DC coefficients vs. higher bands). */
__kernel void adam(int n, __global float *param, __global const float *grad, __global float *m,
                   __global float *v, float lr1, float lr2, int group, int first, float b1, float b2,
                   float eps, float bc1, float bc2)
{
    int i = get_global_id(0);
    float gi, mi, vi, lr;
    if (i >= n)
        return;
    gi = grad[i];
    mi = b1 * m[i] + (1.f - b1) * gi;
    vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    lr = (i % group) < first ? lr1 : lr2;
    param[i] -= lr * (mi / bc1) / (sqrt(vi / bc2) + eps);
}

__kernel void accum_stats(int N, __global const int *radii, __global const float *dL_dmean2d, float sx,
                          float sy, __global float *accum, __global float *denom, __global float *max_radii)
{
    int i = get_global_id(0);
    float gx, gy;
    if (i >= N || radii[i] <= 0)
        return;
    gx = dL_dmean2d[2 * i] * sx;
    gy = dL_dmean2d[2 * i + 1] * sy;
    accum[i] += sqrt(gx * gx + gy * gy);
    denom[i] += 1.f;
    max_radii[i] = max(max_radii[i], (float)radii[i]);
}
