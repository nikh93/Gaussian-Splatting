/*
 * gpu.c - backend selection and shared helpers.
 */
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

const char *gs_backend_name(gs_backend b)
{
    switch (b) {
    case GS_BACKEND_CUDA: return "CUDA";
    case GS_BACKEND_OPENCL: return "OpenCL";
    default: return "auto";
    }
}

int gs_gpu_init(gs_gpu *g, gs_backend backend, int device, const char *kernel_dir, int test_build, char *err,
                size_t err_len)
{
    char cuda_err[512] = "", cl_err[512] = "";
    memset(g, 0, sizeof(*g));
#ifdef GS_WITH_CUDA
    if (backend == GS_BACKEND_AUTO || backend == GS_BACKEND_CUDA) {
        if (gs_cuda_init(g, device, kernel_dir, test_build, cuda_err, sizeof(cuda_err)) == 0)
            return 0;
        memset(g, 0, sizeof(*g));
        if (backend == GS_BACKEND_AUTO)
            fprintf(stderr, "[gpu] CUDA not available (%s), trying OpenCL\n", cuda_err);
    }
#else
    if (backend == GS_BACKEND_CUDA)
        snprintf(cuda_err, sizeof(cuda_err), "this build has no CUDA support");
#endif
#ifdef GS_WITH_OPENCL
    if (backend == GS_BACKEND_AUTO || backend == GS_BACKEND_OPENCL) {
        if (gs_opencl_init(g, device, kernel_dir, test_build, cl_err, sizeof(cl_err)) == 0)
            return 0;
        memset(g, 0, sizeof(*g));
    }
#else
    if (backend == GS_BACKEND_OPENCL)
        snprintf(cl_err, sizeof(cl_err), "this build has no OpenCL support");
#endif
    snprintf(err, err_len, "%s%s%s%s%s", cuda_err[0] ? "CUDA: " : "", cuda_err, cuda_err[0] && cl_err[0] ? "; " : "",
             cl_err[0] ? "OpenCL: " : "", cl_err);
    if (!err[0])
        snprintf(err, err_len, "no GPU backend available");
    return -1;
}

void gs_gpu_release(gs_gpu *g)
{
    if (g->ops)
        g->ops->release(g);
    memset(g, 0, sizeof(*g));
}

void gs_gpu_list_devices(void)
{
#ifdef GS_WITH_CUDA
    printf("CUDA devices:\n");
    gs_cuda_list_devices();
#endif
#ifdef GS_WITH_OPENCL
    printf("OpenCL devices:\n");
    gs_opencl_list_devices();
#endif
}

/* prelude + gsplat_kernels.h as one NUL terminated string */
char *gs_gpu_load_source(const char *kernel_dir, const char *prelude, size_t *len)
{
    char p1[1100], p2[1100];
    size_t l1, l2;
    char *a, *b, *out;
    snprintf(p1, sizeof(p1), "%s/%s", kernel_dir, prelude);
    snprintf(p2, sizeof(p2), "%s/gsplat_kernels.h", kernel_dir);
    a = gs_read_file(p1, &l1);
    b = gs_read_file(p2, &l2);
    if (!a || !b) {
        free(a);
        free(b);
        return NULL;
    }
    {
        static const char sep[] = "\n#line 1 \"gsplat_kernels.h\"\n";
        size_t ls = sizeof(sep) - 1;
        out = (char *)malloc(l1 + ls + l2 + 1);
        memcpy(out, a, l1);
        memcpy(out + l1, sep, ls);
        memcpy(out + l1 + ls, b, l2);
        out[l1 + ls + l2] = 0;
        if (len)
            *len = l1 + ls + l2;
    }
    free(a);
    free(b);
    return out;
}
