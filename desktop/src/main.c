/*
 * main.c - gsplat-server: Gaussian Splatting training server (C11 + OpenCL).
 *
 *   gsplat-server [--port 8080] [--data ./gsplat-data]      run the server for the app
 *   gsplat-server --train <capture dir> --out <dir>           train one capture offline
 *   gsplat-server --list-devices | --selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cl_util.h"
#include "dataset.h"
#include "platform.h"
#include "server.h"
#include "trainer.h"

#ifndef GS_KERNEL_DIR
#define GS_KERNEL_DIR ""
#endif

static void usage(void)
{
    printf("gsplat-server - Gaussian Splatting training on the GPU (OpenCL)\n\n"
           "Server mode (default):\n"
           "  --bind ADDR          address to listen on (default 0.0.0.0; use your Tailscale IP to\n"
           "                       accept connections from the tailnet only)\n"
           "  --port N             TCP port (default 8080)\n"
           "  --data DIR           data directory for jobs (default ./gsplat-data)\n"
           "  --token SECRET       require this token from clients (optional)\n"
           "\n"
           "Offline mode:\n"
           "  --train DIR          train the capture in DIR (transforms.json + images)\n"
           "  --out DIR            output directory (default DIR/output)\n"
           "  --preview DIR        write photo/rendering comparisons (PPM) after training\n"
           "\n"
           "Training options:\n"
           "  --iterations N       optimisation steps (default 30000)\n"
           "  --max-dim N          downscale images so the longest side is <= N (default 1600)\n"
           "  --sh-degree N        spherical harmonics degree 0..3 (default 3)\n"
           "  --max-gaussians N    upper bound for densification (default 3000000)\n"
           "  --white-background   train against a white instead of black background\n"
           "\n"
           "OpenCL:\n"
           "  --device N           OpenCL device index (see --list-devices, default 0 = first GPU)\n"
           "  --kernels FILE       path to gsplat.cl\n"
           "  --list-devices       list OpenCL devices and exit\n"
           "  --selftest           check the GPU gradients against finite differences\n");
}

static int find_kernels(const char *explicit_path, char *out, size_t len)
{
    char exe[1024];
    const char *candidates[6];
    char c0[1100], c1[1100], c2[1100];
    int i;
    if (explicit_path) {
        snprintf(out, len, "%s", explicit_path);
        return gs_file_exists(out) ? 0 : -1;
    }
    gs_exe_dir(exe, sizeof(exe));
    snprintf(c0, sizeof(c0), "%skernels/gsplat.cl", exe);
    snprintf(c1, sizeof(c1), "%s../kernels/gsplat.cl", exe);
    snprintf(c2, sizeof(c2), "%s/gsplat.cl", GS_KERNEL_DIR);
    candidates[0] = c0;
    candidates[1] = c1;
    candidates[2] = c2;
    candidates[3] = "kernels/gsplat.cl";
    candidates[4] = "../kernels/gsplat.cl";
    candidates[5] = NULL;
    for (i = 0; candidates[i]; i++) {
        if (gs_file_exists(candidates[i])) {
            snprintf(out, len, "%s", candidates[i]);
            return 0;
        }
    }
    return -1;
}

static void offline_progress(void *user, int iter, int total, float loss, float psnr, int n)
{
    (void)user;
    (void)iter;
    (void)total;
    (void)loss;
    (void)psnr;
    (void)n;
}

int main(int argc, char **argv)
{
    gs_server_config cfg;
    gs_train_params params;
    gs_cl cl;
    const char *train_dir = NULL, *out_dir = NULL, *kernel_arg = NULL;
    char kernel_path[1100], err[512];
    int device = 0, i, list = 0, selftest = 0;

    setvbuf(stdout, NULL, _IOLBF, 0);
    gs_train_params_default(&params);
    memset(&cfg, 0, sizeof(cfg));
    cfg.bind_addr = "0.0.0.0";
    cfg.port = 8080;
    cfg.data_dir = "gsplat-data";

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define NEEDV()                                                   \
    if (!v) {                                                     \
        fprintf(stderr, "missing value for %s\n", a);             \
        return 2;                                                 \
    }                                                             \
    i++;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage();
            return 0;
        } else if (!strcmp(a, "--bind")) {
            NEEDV();
            cfg.bind_addr = v;
        } else if (!strcmp(a, "--port")) {
            NEEDV();
            cfg.port = atoi(v);
        } else if (!strcmp(a, "--data")) {
            NEEDV();
            cfg.data_dir = v;
        } else if (!strcmp(a, "--token")) {
            NEEDV();
            cfg.token = v;
        } else if (!strcmp(a, "--train")) {
            NEEDV();
            train_dir = v;
        } else if (!strcmp(a, "--out")) {
            NEEDV();
            out_dir = v;
        } else if (!strcmp(a, "--iterations")) {
            NEEDV();
            params.iterations = atoi(v);
        } else if (!strcmp(a, "--max-dim")) {
            NEEDV();
            params.max_dim = atoi(v);
        } else if (!strcmp(a, "--sh-degree")) {
            NEEDV();
            params.sh_degree = atoi(v) < 0 ? 0 : (atoi(v) > 3 ? 3 : atoi(v));
        } else if (!strcmp(a, "--max-gaussians")) {
            NEEDV();
            params.max_gaussians = atoi(v);
        } else if (!strcmp(a, "--preview")) {
            NEEDV();
            params.preview_dir = v;
        } else if (!strcmp(a, "--white-background")) {
            params.white_background = 1;
        } else if (!strcmp(a, "--device")) {
            NEEDV();
            device = atoi(v);
        } else if (!strcmp(a, "--kernels")) {
            NEEDV();
            kernel_arg = v;
        } else if (!strcmp(a, "--list-devices")) {
            list = 1;
        } else if (!strcmp(a, "--selftest")) {
            selftest = 1;
        } else {
            fprintf(stderr, "unknown option %s (see --help)\n", a);
            return 2;
        }
#undef NEEDV
    }
    if (params.iterations < 1)
        params.iterations = 1;
    if (params.iterations < 2 * params.densify_until)
        params.densify_until = params.iterations / 2;

    if (list) {
        printf("OpenCL devices:\n");
        gs_cl_list_devices();
        return 0;
    }
    if (find_kernels(kernel_arg, kernel_path, sizeof(kernel_path)) != 0) {
        fprintf(stderr, "cannot find kernels/gsplat.cl (use --kernels)\n");
        return 1;
    }
    /* the self test needs a smooth loss: no tile culling, no alpha cut-off */
    if (gs_cl_init(&cl, device, kernel_path, selftest ? "-DGS_FULL_TILES" : NULL, err, sizeof(err)) != 0) {
        fprintf(stderr, "OpenCL initialisation failed: %s\n", err);
        fprintf(stderr, "Available devices:\n");
        gs_cl_list_devices();
        return 1;
    }
    printf("[gpu] %s (%s, %llu MB)\n", cl.device_name, cl.platform_name, (unsigned long long)(cl.global_mem >> 20));

    if (selftest) {
        int r = gs_selftest(&cl);
        gs_cl_release(&cl);
        return r;
    }

    if (train_dir) {
        gs_dataset ds;
        gs_train_control ctl;
        char out_buf[1100];
        int r;
        if (!out_dir) {
            snprintf(out_buf, sizeof(out_buf), "%s/output", train_dir);
            out_dir = out_buf;
        }
        gs_mkdirs(out_dir);
        if (gs_dataset_load(&ds, train_dir, params.max_dim, err, sizeof(err)) != 0) {
            fprintf(stderr, "cannot load capture: %s\n", err);
            gs_cl_release(&cl);
            return 1;
        }
        printf("[data] %d views (%dx%d), %d points\n", ds.num_cams, ds.cams[0].width, ds.cams[0].height,
               ds.num_points);
        memset(&ctl, 0, sizeof(ctl));
        ctl.progress = offline_progress;
        r = gs_train(&cl, &ds, &params, out_dir, &ctl, err, sizeof(err));
        if (r != 0)
            fprintf(stderr, "training failed: %s\n", err);
        else
            printf("model written to %s/model.ply and %s/model.splat\n", out_dir, out_dir);
        gs_dataset_free(&ds);
        gs_cl_release(&cl);
        return r == 0 ? 0 : 1;
    }

    if (gs_net_init() != 0) {
        fprintf(stderr, "network initialisation failed\n");
        return 1;
    }
    cfg.defaults = params;
    cfg.cl = &cl;
    printf("[server] point the app to http://<tailscale-ip>:%d\n", cfg.port);
    return gs_server_run(&cfg) == 0 ? 0 : 1;
}
