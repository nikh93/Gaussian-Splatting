/*
 * cl_util.h - OpenCL context / program helpers.
 */
#ifndef GS_CL_UTIL_H
#define GS_CL_UTIL_H

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif
#include <stddef.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    char device_name[256];
    char platform_name[256];
    cl_ulong global_mem;
} gs_cl;

/* Lists all OpenCL devices to stdout. */
void gs_cl_list_devices(void);

/* Opens device number `device_index` (over all platforms, GPUs first) and builds the
 * kernel file with optional extra compiler options. Returns 0 on success. */
int gs_cl_init(gs_cl *cl, int device_index, const char *kernel_path, const char *extra_opts, char *err,
               size_t err_len);
void gs_cl_release(gs_cl *cl);

const char *gs_cl_errstr(cl_int e);

#endif
