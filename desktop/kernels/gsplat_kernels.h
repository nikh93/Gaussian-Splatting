/*
 * gsplat_kernels.h - GPU kernels for differentiable 3D Gaussian Splatting.
 *
 * Written once in a C subset that compiles both as CUDA C (gsplat.cu, via NVRTC
 * or nvcc) and as OpenCL C (host prepends prelude_opencl.h). The dialect
 * differences are hidden behind the macros of prelude_cuda.h / prelude_opencl.h.
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
 * Buffer layout (structure of arrays, N gaussians):
 *   means [3N], log_scales [3N], rots [4N] (w,x,y,z), opacity logits [N],
 *   sh [48N] (16 coefficients x RGB, coefficient major),
 *   mean2d [2N], conic_opac [4N] (A, B, C, opacity), colors [3N],
 *   images planar float [3][H][W], ground truth RGBA8 [H][W][4].
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

/* camera buffer layout */
#define CAM_R 0    /* 9 floats, world->camera rotation, row major */
#define CAM_T 9    /* 3 floats */
#define CAM_POS 12 /* 3 floats, camera centre */
#define CAM_FX 15
#define CAM_FY 16
#define CAM_CX 17
#define CAM_CY 18
#define CAM_W 19
#define CAM_H 20

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f
CONST_DATA float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                             -1.0925484305920792f, 0.5462742152960396f};
CONST_DATA float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                             0.3731763325901154f,  -0.4570457994644658f, 1.445305721320277f,
                             -0.5900435899266435f};

/* SH basis for a unit direction; returns the number of coefficients written. */
DEVICE int sh_basis(int deg, float x, float y, float z, float *b)
{
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

/* Evaluates the SH basis for the direction camera -> gaussian i. */
DEVICE int view_basis(int i, GLOBAL const float *means, CONSTANT float *cam, int sh_deg, float *basis)
{
    float dx = means[3 * i] - cam[CAM_POS];
    float dy = means[3 * i + 1] - cam[CAM_POS + 1];
    float dz = means[3 * i + 2] - cam[CAM_POS + 2];
    float l = F_SQRT(dx * dx + dy * dy + dz * dz);
    float inv = l > 0.f ? 1.f / l : 0.f;
    return sh_basis(sh_deg, dx * inv, dy * inv, dz * inv, basis);
}

/* Rotation matrix (row major) from a normalised quaternion q = (w,x,y,z). */
DEVICE void quat_to_rot(const float *q, float *R)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
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
    float tc[3];        /* camera space centre */
    float txc, tyc;     /* clamped x, y used for the Jacobian */
    int clamp_x, clamp_y;
    float q[4];         /* normalised quaternion */
    float qlen;         /* length before normalisation */
    float Rq[9];        /* rotation from quaternion */
    float s[3];         /* activated scales */
    float Sig[9];       /* 3D covariance (full symmetric) */
    float T[6];         /* J * W (2x3) */
    float a, b, c;      /* 2D covariance incl. low pass */
} geom;

DEVICE int compute_geom(int i, GLOBAL const float *means, GLOBAL const float *log_scales,
                        GLOBAL const float *rots, CONSTANT float *cam, geom *g)
{
    float mx = means[3 * i], my = means[3 * i + 1], mz = means[3 * i + 2];
    float fx = cam[CAM_FX], fy = cam[CAM_FY];
    float limx = 1.3f * 0.5f * cam[CAM_W] / fx;
    float limy = 1.3f * 0.5f * cam[CAM_H] / fy;
    float M[9], TS[6];
    float J00, J02, J11, J12;
    int r, k;

    for (r = 0; r < 3; r++)
        g->tc[r] = cam[CAM_R + 3 * r] * mx + cam[CAM_R + 3 * r + 1] * my + cam[CAM_R + 3 * r + 2] * mz + cam[CAM_T + r];
    if (g->tc[2] <= 0.2f)
        return 0;

    {
        float xz = g->tc[0] / g->tc[2], yz = g->tc[1] / g->tc[2];
        g->clamp_x = xz < -limx || xz > limx;
        g->clamp_y = yz < -limy || yz > limy;
        g->txc = F_MIN(F_MAX(xz, -limx), limx) * g->tc[2];
        g->tyc = F_MIN(F_MAX(yz, -limy), limy) * g->tc[2];
    }

    for (k = 0; k < 4; k++)
        g->q[k] = rots[4 * i + k];
    g->qlen = F_SQRT(g->q[0] * g->q[0] + g->q[1] * g->q[1] + g->q[2] * g->q[2] + g->q[3] * g->q[3]);
    if (g->qlen < 1e-12f) {
        g->q[0] = 1.f;
        g->q[1] = g->q[2] = g->q[3] = 0.f;
        g->qlen = 1.f;
    } else {
        for (k = 0; k < 4; k++)
            g->q[k] /= g->qlen;
    }
    quat_to_rot(g->q, g->Rq);
    for (k = 0; k < 3; k++)
        g->s[k] = F_EXP(log_scales[3 * i + k]);

    /* Sigma = M M^T, M = Rq diag(s) */
    for (r = 0; r < 3; r++)
        for (k = 0; k < 3; k++)
            M[r * 3 + k] = g->Rq[r * 3 + k] * g->s[k];
    for (r = 0; r < 3; r++)
        for (k = 0; k < 3; k++)
            g->Sig[r * 3 + k] = M[r * 3] * M[k * 3] + M[r * 3 + 1] * M[k * 3 + 1] + M[r * 3 + 2] * M[k * 3 + 2];

    {
        float z = g->tc[2], z2 = z * z;
        J00 = fx / z;
        J02 = -fx * g->txc / z2;
        J11 = fy / z;
        J12 = -fy * g->tyc / z2;
    }
    for (k = 0; k < 3; k++) {
        g->T[k] = J00 * cam[CAM_R + k] + J02 * cam[CAM_R + 6 + k];
        g->T[3 + k] = J11 * cam[CAM_R + 3 + k] + J12 * cam[CAM_R + 6 + k];
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

KERNEL void preprocess(int N, GLOBAL const float *means, GLOBAL const float *log_scales,
                       GLOBAL const float *rots, GLOBAL const float *opac, GLOBAL const float *sh,
                       int sh_deg, CONSTANT float *cam, GLOBAL float *mean2d, GLOBAL float *depths,
                       GLOBAL int *radii, GLOBAL float *conic_opac, GLOBAL float *colors)
{
    int i = GID_X;
    geom g;
    float det, mid, lambda1, radius, u, v;
    int gx, gy;
    if (i >= N)
        return;
    radii[i] = 0;
    if (!compute_geom(i, means, log_scales, rots, cam, &g))
        return;
    det = g.a * g.c - g.b * g.b;
    if (det <= 0.f)
        return;
    mid = 0.5f * (g.a + g.c);
    lambda1 = mid + F_SQRT(F_MAX(0.1f, mid * mid - det));
    radius = F_CEIL(3.f * F_SQRT(lambda1));
#ifdef GS_FULL_TILES
    radius = 4096.f; /* testing only: every gaussian touches every tile */
#endif
    u = cam[CAM_FX] * g.tc[0] / g.tc[2] + cam[CAM_CX];
    v = cam[CAM_FY] * g.tc[1] / g.tc[2] + cam[CAM_CY];

    gx = ((int)cam[CAM_W] + TILE - 1) / TILE;
    gy = ((int)cam[CAM_H] + TILE - 1) / TILE;
    {
        int x0 = I_MIN(gx, I_MAX(0, (int)((u - radius) / TILE)));
        int y0 = I_MIN(gy, I_MAX(0, (int)((v - radius) / TILE)));
        int x1 = I_MIN(gx, I_MAX(0, (int)((u + radius + TILE - 1) / TILE)));
        int y1 = I_MIN(gy, I_MAX(0, (int)((v + radius + TILE - 1) / TILE)));
        if ((x1 - x0) * (y1 - y0) == 0)
            return;
    }

    {
        float basis[16];
        int n = view_basis(i, means, cam, sh_deg, basis), k, c;
        GLOBAL const float *f = sh + (size_t)i * SH_STRIDE;
        for (c = 0; c < 3; c++) {
            float col = 0.5f;
            for (k = 0; k < n; k++)
                col += basis[k] * f[3 * k + c];
            colors[3 * i + c] = F_MAX(col, 0.f);
        }
    }

    mean2d[2 * i] = u;
    mean2d[2 * i + 1] = v;
    depths[i] = g.tc[2];
    radii[i] = (int)radius;
    conic_opac[4 * i] = g.c / det;
    conic_opac[4 * i + 1] = -g.b / det;
    conic_opac[4 * i + 2] = g.a / det;
    conic_opac[4 * i + 3] = 1.f / (1.f + F_EXP(-opac[i]));
}

TILE_KERNEL void render_fwd(GLOBAL const int *ranges, GLOBAL const int *point_list, int W, int H,
                            GLOBAL const float *mean2d, GLOBAL const float *conic_opac,
                            GLOBAL const float *colors, float bg_r, float bg_g, float bg_b,
                            GLOBAL float *out_img, GLOBAL float *final_T, GLOBAL int *n_contrib)
{
    SHARED int l_id[BLOCK];
    SHARED float l_xy[2 * BLOCK];
    SHARED float l_co[4 * BLOCK];
    SHARED int l_done[1];
    int tiles_x = (W + TILE - 1) / TILE;
    int tile = GRP_Y * tiles_x + GRP_X;
    int px = GRP_X * TILE + LID_X;
    int py = GRP_Y * TILE + LID_Y;
    int tid = LID_Y * TILE + LID_X;
    int inside = px < W && py < H;
    int done = !inside;
    int r0 = ranges[2 * tile], r1 = ranges[2 * tile + 1];
    int todo = r1 - r0;
    int rounds = (todo + BLOCK - 1) / BLOCK;
    float T = 1.f, C0 = 0.f, C1 = 0.f, C2 = 0.f;
    int contributor = 0, last_contributor = 0;
    float pfx = (float)px, pfy = (float)py;
    int r, j;

    for (r = 0; r < rounds; r++, todo -= BLOCK) {
        /* stop when every pixel of the tile is saturated */
        SYNC();
        if (tid == 0)
            l_done[0] = 0;
        SYNC();
        if (done)
            SHARED_INC(&l_done[0]);
        SYNC();
        if (l_done[0] == BLOCK)
            break;
        {
            int progress = r0 + r * BLOCK + tid;
            if (progress < r1) {
                int id = point_list[progress];
                l_id[tid] = id;
                l_xy[2 * tid] = mean2d[2 * id];
                l_xy[2 * tid + 1] = mean2d[2 * id + 1];
                l_co[4 * tid] = conic_opac[4 * id];
                l_co[4 * tid + 1] = conic_opac[4 * id + 1];
                l_co[4 * tid + 2] = conic_opac[4 * id + 2];
                l_co[4 * tid + 3] = conic_opac[4 * id + 3];
            }
        }
        SYNC();
        for (j = 0; !done && j < I_MIN(BLOCK, todo); j++) {
            float dx = l_xy[2 * j] - pfx, dy = l_xy[2 * j + 1] - pfy;
            float power = -0.5f * (l_co[4 * j] * dx * dx + l_co[4 * j + 2] * dy * dy) - l_co[4 * j + 1] * dx * dy;
            float alpha, test_T, w;
            int id;
            contributor++;
            if (power > 0.f)
                continue;
            alpha = F_MIN(0.99f, l_co[4 * j + 3] * F_EXP(power));
            if (alpha < ALPHA_MIN)
                continue;
            test_T = T * (1.f - alpha);
            if (test_T < 0.0001f) {
                done = 1;
                continue;
            }
            id = l_id[j];
            w = alpha * T;
            C0 += colors[3 * id] * w;
            C1 += colors[3 * id + 1] * w;
            C2 += colors[3 * id + 2] * w;
            T = test_T;
            last_contributor = contributor;
        }
    }
    if (inside) {
        int p = py * W + px;
        int hw = W * H;
        final_T[p] = T;
        n_contrib[p] = last_contributor;
        out_img[p] = C0 + T * bg_r;
        out_img[hw + p] = C1 + T * bg_g;
        out_img[2 * hw + p] = C2 + T * bg_b;
    }
}

/* ===================================================================== */
/* loss: 0.8 * L1 + 0.2 * (1 - SSIM), 11x11 gaussian window, sigma 1.5    */
/* ===================================================================== */

CONST_DATA float GW[11] = {0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f, 0.21300553f, 0.26601172f,
                           0.21300553f, 0.10936069f, 0.03600077f, 0.00759876f, 0.00102838f};

#define GT(gt, p, c) ((float)(gt)[(p) * 4 + (c)] * (1.f / 255.f))

/* horizontal blur of x, y, x^2, y^2, xy  -> tmp[5][3][HW] */
KERNEL void ssim_h(int W, int H, GLOBAL const float *img, GLOBAL const unsigned char *gt, GLOBAL float *tmp)
{
    int x = GID_X, y = GID_Y, c = GID_Z;
    int hw = W * H, k;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f;
    if (x >= W || y >= H || c >= 3)
        return;
    for (k = -5; k <= 5; k++) {
        int xx = x + k;
        float w, a, b;
        if (xx < 0 || xx >= W)
            continue;
        w = GW[k + 5];
        a = img[c * hw + y * W + xx];
        b = GT(gt, y * W + xx, c);
        s0 += w * a;
        s1 += w * b;
        s2 += w * a * a;
        s3 += w * b * b;
        s4 += w * a * b;
    }
    tmp[(0 * 3 + c) * hw + y * W + x] = s0;
    tmp[(1 * 3 + c) * hw + y * W + x] = s1;
    tmp[(2 * 3 + c) * hw + y * W + x] = s2;
    tmp[(3 * 3 + c) * hw + y * W + x] = s3;
    tmp[(4 * 3 + c) * hw + y * W + x] = s4;
}

/* vertical blur, SSIM and its partial derivatives -> efh[3][3][HW], ssim_map[3][HW] */
KERNEL void ssim_v(int W, int H, GLOBAL const float *tmp, GLOBAL float *efh, GLOBAL float *ssim_map)
{
    int x = GID_X, y = GID_Y, c = GID_Z;
    int hw = W * H, k, q;
    float s[5];
    const float C1 = 0.0001f, C2 = 0.0009f;
    float mx, my, sxx, syy, sxy, A1, A2, B1, B2, f;
    int p = y * W + x;
    if (x >= W || y >= H || c >= 3)
        return;
    for (q = 0; q < 5; q++)
        s[q] = 0.f;
    for (k = -5; k <= 5; k++) {
        int yy = y + k;
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
KERNEL void ssim_bh(int W, int H, GLOBAL const float *efh, GLOBAL float *tmp)
{
    int x = GID_X, y = GID_Y, c = GID_Z;
    int hw = W * H, k, q;
    float s[3];
    if (x >= W || y >= H || c >= 3)
        return;
    s[0] = s[1] = s[2] = 0.f;
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
KERNEL void loss_grad(int W, int H, float lambda_ssim, GLOBAL const float *img, GLOBAL const unsigned char *gt,
                      GLOBAL const float *tmp, GLOBAL float *dL_dimg)
{
    int x = GID_X, y = GID_Y, c = GID_Z;
    int hw = W * H, k, q;
    int p = y * W + x;
    float s[3];
    float a, b, n_inv, g_l1, g_ssim;
    if (x >= W || y >= H || c >= 3)
        return;
    s[0] = s[1] = s[2] = 0.f;
    for (k = -5; k <= 5; k++) {
        int yy = y + k;
        if (yy < 0 || yy >= H)
            continue;
        for (q = 0; q < 3; q++)
            s[q] += GW[k + 5] * tmp[(q * 3 + c) * hw + yy * W + x];
    }
    a = img[c * hw + p];
    b = GT(gt, p, c);
    n_inv = 1.f / (3.f * (float)hw);
    g_l1 = a > b ? 1.f : (a < b ? -1.f : 0.f);
    g_ssim = s[0] + 2.f * a * s[1] + b * s[2];
    dL_dimg[c * hw + p] = ((1.f - lambda_ssim) * g_l1 - lambda_ssim * g_ssim) * n_inv;
}

/* ===================================================================== */
/* backward                                                              */
/* ===================================================================== */

TILE_KERNEL void render_bwd(GLOBAL const int *ranges, GLOBAL const int *point_list, int W, int H,
                            GLOBAL const float *mean2d, GLOBAL const float *conic_opac,
                            GLOBAL const float *colors, float bg_r, float bg_g, float bg_b,
                            GLOBAL const float *final_Ts, GLOBAL const int *n_contrib,
                            GLOBAL const float *dL_dimg, GLOBAL float *dL_dmean2d, GLOBAL float *dL_dconic,
                            GLOBAL float *dL_dopa, GLOBAL float *dL_dcolor)
{
    SHARED int l_id[BLOCK];
    SHARED float l_xy[2 * BLOCK];
    SHARED float l_co[4 * BLOCK];
    SHARED float l_col[3 * BLOCK];
    int tiles_x = (W + TILE - 1) / TILE;
    int tile = GRP_Y * tiles_x + GRP_X;
    int px = GRP_X * TILE + LID_X;
    int py = GRP_Y * TILE + LID_Y;
    int tid = LID_Y * TILE + LID_X;
    int inside = px < W && py < H;
    int done = !inside;
    int r0 = ranges[2 * tile], r1 = ranges[2 * tile + 1];
    int todo = r1 - r0;
    int rounds = (todo + BLOCK - 1) / BLOCK;
    int hw = W * H;
    int p = inside ? py * W + px : 0;
    float T_final = inside ? final_Ts[p] : 0.f;
    float T = T_final;
    int contributor = todo;
    int last_contributor = inside ? n_contrib[p] : 0;
    float dp0 = 0.f, dp1 = 0.f, dp2 = 0.f;    /* dL / dpixel */
    float ac0 = 0.f, ac1 = 0.f, ac2 = 0.f;    /* colour accumulated behind the current splat */
    float lc0 = 0.f, lc1 = 0.f, lc2 = 0.f;    /* last colour */
    float last_alpha = 0.f;
    float bg_dot;
    float pfx = (float)px, pfy = (float)py;
    int r, j;

    if (inside) {
        dp0 = dL_dimg[p];
        dp1 = dL_dimg[hw + p];
        dp2 = dL_dimg[2 * hw + p];
    }
    bg_dot = bg_r * dp0 + bg_g * dp1 + bg_b * dp2;

    for (r = 0; r < rounds; r++, todo -= BLOCK) {
        SYNC();
        {
            int progress = r * BLOCK + tid;
            if (r0 + progress < r1) {
                int id = point_list[r1 - progress - 1];
                l_id[tid] = id;
                l_xy[2 * tid] = mean2d[2 * id];
                l_xy[2 * tid + 1] = mean2d[2 * id + 1];
                l_co[4 * tid] = conic_opac[4 * id];
                l_co[4 * tid + 1] = conic_opac[4 * id + 1];
                l_co[4 * tid + 2] = conic_opac[4 * id + 2];
                l_co[4 * tid + 3] = conic_opac[4 * id + 3];
                l_col[3 * tid] = colors[3 * id];
                l_col[3 * tid + 1] = colors[3 * id + 1];
                l_col[3 * tid + 2] = colors[3 * id + 2];
            }
        }
        SYNC();
        for (j = 0; !done && j < I_MIN(BLOCK, todo); j++) {
            float dx, dy, cA, cB, cC, op, power, G, alpha, dL_dalpha, dL_dG, gdx, gdy, w;
            float c0, c1, c2;
            int id;
            contributor--;
            if (contributor >= last_contributor)
                continue;
            dx = l_xy[2 * j] - pfx;
            dy = l_xy[2 * j + 1] - pfy;
            cA = l_co[4 * j];
            cB = l_co[4 * j + 1];
            cC = l_co[4 * j + 2];
            op = l_co[4 * j + 3];
            power = -0.5f * (cA * dx * dx + cC * dy * dy) - cB * dx * dy;
            if (power > 0.f)
                continue;
            G = F_EXP(power);
            alpha = F_MIN(0.99f, op * G);
            if (alpha < ALPHA_MIN)
                continue;
            T = T / (1.f - alpha);
            c0 = l_col[3 * j];
            c1 = l_col[3 * j + 1];
            c2 = l_col[3 * j + 2];
            id = l_id[j];
            ac0 = last_alpha * lc0 + (1.f - last_alpha) * ac0;
            ac1 = last_alpha * lc1 + (1.f - last_alpha) * ac1;
            ac2 = last_alpha * lc2 + (1.f - last_alpha) * ac2;
            lc0 = c0;
            lc1 = c1;
            lc2 = c2;
            dL_dalpha = ((c0 - ac0) * dp0 + (c1 - ac1) * dp1 + (c2 - ac2) * dp2) * T;
            last_alpha = alpha;
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot;
            w = alpha * T;
            ATOMIC_ADD(&dL_dcolor[3 * id], w * dp0);
            ATOMIC_ADD(&dL_dcolor[3 * id + 1], w * dp1);
            ATOMIC_ADD(&dL_dcolor[3 * id + 2], w * dp2);
            dL_dG = op * dL_dalpha;
            gdx = G * dx;
            gdy = G * dy;
            ATOMIC_ADD(&dL_dmean2d[2 * id], dL_dG * (-gdx * cA - gdy * cB));
            ATOMIC_ADD(&dL_dmean2d[2 * id + 1], dL_dG * (-gdy * cC - gdx * cB));
            ATOMIC_ADD(&dL_dconic[3 * id], -0.5f * gdx * dx * dL_dG);
            ATOMIC_ADD(&dL_dconic[3 * id + 1], -gdx * dy * dL_dG);
            ATOMIC_ADD(&dL_dconic[3 * id + 2], -0.5f * gdy * dy * dL_dG);
            ATOMIC_ADD(&dL_dopa[id], G * dL_dalpha);
        }
    }
}

/* derivative of the rotation matrix (row major) w.r.t. one quaternion component, dotted with dR */
DEVICE float quat_grad_dot(const float *dR, const float *q, int comp)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
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

KERNEL void preprocess_bwd(int N, GLOBAL const float *means, GLOBAL const float *log_scales,
                           GLOBAL const float *rots, GLOBAL const float *opac, GLOBAL const float *sh,
                           int sh_deg, CONSTANT float *cam, GLOBAL const int *radii,
                           GLOBAL const float *dL_dmean2d, GLOBAL const float *dL_dconic,
                           GLOBAL const float *dL_dopa, GLOBAL const float *dL_dcolor, GLOBAL float *g_means,
                           GLOBAL float *g_scales, GLOBAL float *g_rots, GLOBAL float *g_opac,
                           GLOBAL float *g_sh)
{
    int i = GID_X;
    geom g;
    GLOBAL float *gsh;
    int k, r;
    if (i >= N)
        return;
    gsh = g_sh + (size_t)i * SH_STRIDE;
    if (radii[i] <= 0 || !compute_geom(i, means, log_scales, rots, cam, &g)) {
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
        float sig = 1.f / (1.f + F_EXP(-opac[i]));
        g_opac[i] = dL_dopa[i] * sig * (1.f - sig);
    }

    /* --- colour / SH (the view direction gradient is ignored) ---------- */
    {
        float basis[16];
        int n = view_basis(i, means, cam, sh_deg, basis), c;
        GLOBAL const float *f = sh + (size_t)i * SH_STRIDE;
        for (c = 0; c < 3; c++) {
            float raw = 0.5f, dc = dL_dcolor[3 * i + c];
            for (k = 0; k < n; k++)
                raw += basis[k] * f[3 * k + c];
            if (raw < 0.f)
                dc = 0.f; /* colour was clamped */
            for (k = 0; k < 16; k++)
                gsh[3 * k + c] = k < n ? basis[k] * dc : 0.f;
        }
    }

    {
        float fx = cam[CAM_FX], fy = cam[CAM_FY];
        float z = g.tc[2], z2 = z * z, z3 = z2 * z;
        float a = g.a, b = g.b, c = g.c;
        float det = a * c - b * b;
        float gA = dL_dconic[3 * i], gB = dL_dconic[3 * i + 1], gC = dL_dconic[3 * i + 2];
        float ga = 0.f, gb = 0.f, gc = 0.f;
        float Gs[4], dSig[9], dT[6], dt[3];
        float gu = dL_dmean2d[2 * i], gv = dL_dmean2d[2 * i + 1];

        /* conic = inverse of the 2D covariance [[a b][b c]] */
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

        /* T = J W  ->  dL/dJ  ->  dL/d(camera space centre) */
        {
            float gJ00 = dT[0] * cam[CAM_R + 0] + dT[1] * cam[CAM_R + 1] + dT[2] * cam[CAM_R + 2];
            float gJ02 = dT[0] * cam[CAM_R + 6] + dT[1] * cam[CAM_R + 7] + dT[2] * cam[CAM_R + 8];
            float gJ11 = dT[3] * cam[CAM_R + 3] + dT[4] * cam[CAM_R + 4] + dT[5] * cam[CAM_R + 5];
            float gJ12 = dT[3] * cam[CAM_R + 6] + dT[4] * cam[CAM_R + 7] + dT[5] * cam[CAM_R + 8];
            dt[0] = g.clamp_x ? 0.f : gJ02 * (-fx / z2);
            dt[1] = g.clamp_y ? 0.f : gJ12 * (-fy / z2);
            dt[2] = gJ00 * (-fx / z2) + gJ02 * (2.f * fx * g.txc / z3) + gJ11 * (-fy / z2) +
                    gJ12 * (2.f * fy * g.tyc / z3);
        }

        /* projected mean */
        dt[0] += gu * fx / z;
        dt[1] += gv * fy / z;
        dt[2] += -(gu * fx * g.tc[0] + gv * fy * g.tc[1]) / z2;

        for (k = 0; k < 3; k++)
            g_means[3 * i + k] = cam[CAM_R + k] * dt[0] + cam[CAM_R + 3 + k] * dt[1] + cam[CAM_R + 6 + k] * dt[2];

        /* Sigma = M M^T, M = Rq diag(s):  dL/dM = 2 dSig M */
        {
            float M[9], dM[9], dR[9], gq[4];
            float proj;
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    M[r * 3 + k] = g.Rq[r * 3 + k] * g.s[k];
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    dM[r * 3 + k] = 2.f * (dSig[r * 3] * M[k] + dSig[r * 3 + 1] * M[3 + k] + dSig[r * 3 + 2] * M[6 + k]);
            for (k = 0; k < 3; k++) {
                float ds = dM[k] * g.Rq[k] + dM[3 + k] * g.Rq[3 + k] + dM[6 + k] * g.Rq[6 + k];
                g_scales[3 * i + k] = ds * g.s[k];
            }
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    dR[r * 3 + k] = dM[r * 3 + k] * g.s[k];
            for (k = 0; k < 4; k++)
                gq[k] = quat_grad_dot(dR, g.q, k);
            /* through the normalisation q / |q| */
            proj = g.q[0] * gq[0] + g.q[1] * gq[1] + g.q[2] * gq[2] + g.q[3] * gq[3];
            for (k = 0; k < 4; k++)
                g_rots[4 * i + k] = (gq[k] - g.q[k] * proj) / g.qlen;
        }
    }
}

/* ===================================================================== */
/* optimiser + densification statistics                                 */
/* ===================================================================== */

/* Elements whose index modulo `group` is below `first` use lr1, others lr2
 * (used for the SH buffer: DC coefficients vs. higher bands). */
KERNEL void adam(int n, GLOBAL float *param, GLOBAL const float *grad, GLOBAL float *m, GLOBAL float *v,
                 float lr1, float lr2, int group, int first, float b1, float b2, float eps, float bc1, float bc2)
{
    int i = GID_X;
    float gi, mi, vi, lr;
    if (i >= n)
        return;
    gi = grad[i];
    mi = b1 * m[i] + (1.f - b1) * gi;
    vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    lr = (i % group) < first ? lr1 : lr2;
    param[i] -= lr * (mi / bc1) / (F_SQRT(vi / bc2) + eps);
}

KERNEL void accum_stats(int N, GLOBAL const int *radii, GLOBAL const float *dL_dmean2d, float sx, float sy,
                        GLOBAL float *accum, GLOBAL float *denom, GLOBAL float *max_radii)
{
    int i = GID_X;
    float gx, gy;
    if (i >= N || radii[i] <= 0)
        return;
    gx = dL_dmean2d[2 * i] * sx;
    gy = dL_dmean2d[2 * i + 1] * sy;
    accum[i] += F_SQRT(gx * gx + gy * gy);
    denom[i] += 1.f;
    max_radii[i] = F_MAX(max_radii[i], (float)radii[i]);
}
