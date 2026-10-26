/*
 * cl_util.c - OpenCL context / program helpers.
 */
#include "cl_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

const char *gs_cl_errstr(cl_int e)
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
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    default: return "CL_ERROR";
    }
}

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_device_type type;
} dev_entry;

/* Collects devices ordered GPUs first, then accelerators, then CPUs. */
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

void gs_cl_list_devices(void)
{
    dev_entry devs[64];
    int n = collect_devices(devs, 64), i;
    if (n == 0) {
        printf("No OpenCL devices found. Install your GPU vendor's OpenCL driver.\n");
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

int gs_cl_init(gs_cl *cl, int device_index, const char *kernel_path, const char *extra_opts, char *err,
               size_t err_len)
{
    dev_entry devs[64];
    int n = collect_devices(devs, 64);
    cl_int e;
    char *src;
    size_t src_len;
    char opts[512];
    snprintf(opts, sizeof(opts), "-cl-std=CL1.2 -cl-mad-enable -DTILE=16 %s", extra_opts ? extra_opts : "");

    memset(cl, 0, sizeof(*cl));
    if (n == 0) {
        snprintf(err, err_len, "no OpenCL device found");
        return -1;
    }
    if (device_index < 0)
        device_index = 0;
    if (device_index >= n) {
        snprintf(err, err_len, "OpenCL device %d does not exist (%d available)", device_index, n);
        return -1;
    }
    cl->platform = devs[device_index].platform;
    cl->device = devs[device_index].device;
    clGetDeviceInfo(cl->device, CL_DEVICE_NAME, sizeof(cl->device_name), cl->device_name, NULL);
    clGetPlatformInfo(cl->platform, CL_PLATFORM_NAME, sizeof(cl->platform_name), cl->platform_name, NULL);
    clGetDeviceInfo(cl->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl->global_mem), &cl->global_mem, NULL);

    cl->ctx = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &e);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateContext: %s", gs_cl_errstr(e));
        return -1;
    }
    cl->queue = clCreateCommandQueue(cl->ctx, cl->device, 0, &e);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateCommandQueue: %s", gs_cl_errstr(e));
        gs_cl_release(cl);
        return -1;
    }
    src = gs_read_file(kernel_path, &src_len);
    if (!src) {
        snprintf(err, err_len, "cannot read kernel file %s", kernel_path);
        gs_cl_release(cl);
        return -1;
    }
    cl->program = clCreateProgramWithSource(cl->ctx, 1, (const char **)&src, &src_len, &e);
    free(src);
    if (e != CL_SUCCESS) {
        snprintf(err, err_len, "clCreateProgramWithSource: %s", gs_cl_errstr(e));
        gs_cl_release(cl);
        return -1;
    }
    e = clBuildProgram(cl->program, 1, &cl->device, opts, NULL, NULL);
    if (e != CL_SUCCESS) {
        char *log;
        size_t log_len = 0;
        clGetProgramBuildInfo(cl->program, cl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        log = (char *)malloc(log_len + 1);
        clGetProgramBuildInfo(cl->program, cl->device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        fprintf(stderr, "OpenCL build log:\n%s\n", log);
        free(log);
        snprintf(err, err_len, "kernel build failed: %s", gs_cl_errstr(e));
        gs_cl_release(cl);
        return -1;
    }
    return 0;
}

void gs_cl_release(gs_cl *cl)
{
    if (cl->program)
        clReleaseProgram(cl->program);
    if (cl->queue)
        clReleaseCommandQueue(cl->queue);
    if (cl->ctx)
        clReleaseContext(cl->ctx);
    memset(cl, 0, sizeof(*cl));
}
