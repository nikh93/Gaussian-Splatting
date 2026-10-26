/*
 * gpu_opencl.c - OpenCL 1.2 backend (AMD / Intel GPUs, CPU via PoCL for testing).
 */
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu.h"

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
} cl_impl;

static const char *cl_errstr(cl_int e)
{
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    default: return "CL_ERROR";
    }
}

#define IMPL(g) ((cl_impl *)(g)->impl)

static int fail(gs_gpu *g, const char *what, cl_int e)
{
    snprintf(g->error, sizeof(g->error), "%s: %s (%d)", what, cl_errstr(e), (int)e);
    fprintf(stderr, "[opencl] %s\n", g->error);
    return -1;
}

static void ocl_release(gs_gpu *g)
{
    cl_impl *c = IMPL(g);
    if (!c)
        return;
    if (c->program)
        clReleaseProgram(c->program);
    if (c->queue)
        clReleaseCommandQueue(c->queue);
    if (c->ctx)
        clReleaseContext(c->ctx);
    free(c);
    g->impl = NULL;
}

static int ocl_bind_thread(gs_gpu *g)
{
    (void)g;
    return 0;
}

static int ocl_alloc(gs_gpu *g, gs_buf *b, size_t bytes)
{
    cl_int e;
    b->u.ptr = clCreateBuffer(IMPL(g)->ctx, CL_MEM_READ_WRITE, bytes ? bytes : 4, NULL, &e);
    b->bytes = bytes;
    if (e != CL_SUCCESS) {
        b->u.ptr = NULL;
        return fail(g, "clCreateBuffer", e);
    }
    return 0;
}

static void ocl_free(gs_gpu *g, gs_buf *b)
{
    (void)g;
    if (b->u.ptr)
        clReleaseMemObject((cl_mem)b->u.ptr);
    b->u.ptr = NULL;
    b->bytes = 0;
}

static int ocl_write(gs_gpu *g, gs_buf *b, size_t off, size_t bytes, const void *src)
{
    cl_int e;
    if (!bytes)
        return 0;
    e = clEnqueueWriteBuffer(IMPL(g)->queue, (cl_mem)b->u.ptr, CL_TRUE, off, bytes, src, 0, NULL, NULL);
    return e == CL_SUCCESS ? 0 : fail(g, "clEnqueueWriteBuffer", e);
}

static int ocl_read(gs_gpu *g, const gs_buf *b, size_t off, size_t bytes, void *dst)
{
    cl_int e;
    if (!bytes)
        return 0;
    e = clEnqueueReadBuffer(IMPL(g)->queue, (cl_mem)b->u.ptr, CL_TRUE, off, bytes, dst, 0, NULL, NULL);
    return e == CL_SUCCESS ? 0 : fail(g, "clEnqueueReadBuffer", e);
}

static int ocl_zero(gs_gpu *g, gs_buf *b, size_t bytes)
{
    cl_int e;
    float zero = 0.f;
    if (!bytes)
        return 0;
    e = clEnqueueFillBuffer(IMPL(g)->queue, (cl_mem)b->u.ptr, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
    return e == CL_SUCCESS ? 0 : fail(g, "clEnqueueFillBuffer", e);
}

static gs_kernel ocl_kernel(gs_gpu *g, const char *name)
{
    cl_int e;
    cl_kernel k = clCreateKernel(IMPL(g)->program, name, &e);
    if (e != CL_SUCCESS) {
        fail(g, name, e);
        return NULL;
    }
    return (gs_kernel)k;
}

static void ocl_kernel_release(gs_gpu *g, gs_kernel k)
{
    (void)g;
    if (k)
        clReleaseKernel((cl_kernel)k);
}

static int ocl_launch(gs_gpu *g, gs_kernel k, int dims, const size_t global[3], const size_t local[3], int nargs,
                      const gs_arg *args)
{
    size_t gl[3];
    int i;
    cl_int e;
    for (i = 0; i < nargs; i++) {
        e = clSetKernelArg((cl_kernel)k, (cl_uint)i, args[i].size, args[i].ptr);
        if (e != CL_SUCCESS)
            return fail(g, "clSetKernelArg", e);
    }
    for (i = 0; i < dims; i++) {
        if (global[i] == 0)
            return 0;
        gl[i] = (global[i] + local[i] - 1) / local[i] * local[i];
    }
    e = clEnqueueNDRangeKernel(IMPL(g)->queue, (cl_kernel)k, (cl_uint)dims, NULL, gl, local, 0, NULL, NULL);
    return e == CL_SUCCESS ? 0 : fail(g, "clEnqueueNDRangeKernel", e);
}

static int ocl_finish(gs_gpu *g)
{
    cl_int e = clFinish(IMPL(g)->queue);
    return e == CL_SUCCESS ? 0 : fail(g, "clFinish", e);
}

static const gs_gpu_ops ocl_ops = {ocl_release, ocl_bind_thread, ocl_alloc,  ocl_free,
                                   ocl_write,   ocl_read,        ocl_zero,   ocl_kernel,
                                   ocl_kernel_release, ocl_launch, ocl_finish};

/* ---------------------------------------------------------------------- */

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_device_type type;
} dev_entry;

/* GPUs first, then accelerators, then CPUs */
static int collect_devices(dev_entry *out, int max)
{
    cl_platform_id plats[16];
    cl_uint np = 0, p;
    int n = 0, pass;
    if (clGetPlatformIDs(16, plats, &np) != CL_SUCCESS)
        return 0;
    for (pass = 0; pass < 3; pass++) {
        cl_device_type want = pass == 0 ? CL_DEVICE_TYPE_GPU
                                        : (pass == 1 ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU);
        for (p = 0; p < np; p++) {
            cl_device_id devs[16];
            cl_uint nd = 0, d;
            if (clGetDeviceIDs(plats[p], want, 16, devs, &nd) != CL_SUCCESS)
                continue;
            for (d = 0; d < nd && n < max; d++) {
                out[n].platform = plats[p];
                out[n].device = devs[d];
                out[n].type = want;
                n++;
            }
        }
    }
    return n;
}

void gs_opencl_list_devices(void)
{
    dev_entry devs[64];
    int n = collect_devices(devs, 64), i;
    if (n == 0) {
        printf("  (none)\n");
        return;
    }
    for (i = 0; i < n; i++) {
        char name[256], pname[256];
        cl_ulong mem = 0;
        clGetDeviceInfo(devs[i].device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        clGetDeviceInfo(devs[i].device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        clGetPlatformInfo(devs[i].platform, CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        printf("  [%d] %s (%s, %s, %.1f GB)\n", i, name, pname,
               devs[i].type == CL_DEVICE_TYPE_GPU ? "GPU" : (devs[i].type == CL_DEVICE_TYPE_CPU ? "CPU" : "ACC"),
               (double)mem / (1024.0 * 1024.0 * 1024.0));
    }
}

int gs_opencl_init(gs_gpu *g, int device, const char *kernel_dir, int test_build, char *err, size_t err_len)
{
    dev_entry devs[64];
    int n = collect_devices(devs, 64);
    cl_impl *c;
    cl_int e;
    char *src;
    size_t src_len;
    const char *opts = test_build ? "-cl-std=CL1.2 -cl-mad-enable -DTILE=16 -DGS_FULL_TILES"
                                  : "-cl-std=CL1.2 -cl-mad-enable -DTILE=16";

    if (n == 0) {
        snprintf(err, err_len, "no OpenCL device found");
        return -1;
    }
    if (device < 0)
        device = 0;
    if (device >= n) {
        snprintf(err, err_len, "OpenCL device %d does not exist (%d available)", device, n);
        return -1;
    }
    c = (cl_impl *)calloc(1, sizeof(cl_impl));
    g->impl = c;
    g->ops = &ocl_ops;
    g->backend = GS_BACKEND_OPENCL;
    c->platform = devs[device].platform;
    c->device = devs[device].device;
    clGetDeviceInfo(c->device, CL_DEVICE_NAME, sizeof(g->device_name), g->device_name, NULL);
    clGetPlatformInfo(c->platform, CL_PLATFORM_NAME, sizeof(g->platform_name), g->platform_name, NULL);
    {
        cl_ulong mem = 0;
        clGetDeviceInfo(c->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        g->global_mem = mem;
    }
    c->ctx = clCreateContext(NULL, 1, &c->device, NULL, NULL, &e);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateContext: %s", cl_errstr(e));
        ocl_release(g);
        return -1;
    }
    c->queue = clCreateCommandQueue(c->ctx, c->device, 0, &e);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateCommandQueue: %s", cl_errstr(e));
        ocl_release(g);
        return -1;
    }
    src = gs_gpu_load_source(kernel_dir, "prelude_opencl.h", &src_len);
    if (!src) {
        snprintf(err, err_len, "cannot read kernels from %s", kernel_dir);
        ocl_release(g);
        return -1;
    }
    c->program = clCreateProgramWithSource(c->ctx, 1, (const char **)&src, &src_len, &e);
    free(src);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateProgramWithSource: %s", cl_errstr(e));
        ocl_release(g);
        return -1;
    }
    e = clBuildProgram(c->program, 1, &c->device, opts, NULL, NULL);
    if (e != CL_SUCCESS) {
        char *log;
        size_t log_len = 0;
        clGetProgramBuildInfo(c->program, c->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        log = (char *)malloc(log_len + 1);
        clGetProgramBuildInfo(c->program, c->device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        fprintf(stderr, "OpenCL build log:\n%s\n", log);
        free(log);
        snprintf(err, err_len, "kernel build failed: %s", cl_errstr(e));
        ocl_release(g);
        return -1;
    }
    return 0;
}
