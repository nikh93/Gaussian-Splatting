/*
 * gpu.h - minimal compute abstraction with two backends:
 *   CUDA   (NVIDIA, default): driver API loaded at runtime, kernels compiled with
 *          NVRTC for the installed GPU (or a PTX file built by nvcc)
 *   OpenCL (fallback for other GPUs / CPU testing)
 * Both run the same kernel source (kernels/gsplat_kernels.h).
 */
#ifndef GS_GPU_H
#define GS_GPU_H

#include <stddef.h>

typedef enum { GS_BACKEND_AUTO = 0, GS_BACKEND_CUDA, GS_BACKEND_OPENCL } gs_backend;

typedef struct {
    size_t bytes;
    union {
        void *ptr;              /* OpenCL: cl_mem */
        unsigned long long dev; /* CUDA: CUdeviceptr */
    } u;
} gs_buf;

typedef void *gs_kernel;

typedef struct {
    const void *ptr;
    size_t size;
} gs_arg;

typedef struct gs_gpu gs_gpu;

typedef struct {
    void (*release)(gs_gpu *g);
    int (*bind_thread)(gs_gpu *g);
    int (*alloc)(gs_gpu *g, gs_buf *b, size_t bytes);
    void (*free)(gs_gpu *g, gs_buf *b);
    int (*write)(gs_gpu *g, gs_buf *b, size_t offset, size_t bytes, const void *src);
    int (*read)(gs_gpu *g, const gs_buf *b, size_t offset, size_t bytes, void *dst);
    int (*zero)(gs_gpu *g, gs_buf *b, size_t bytes);
    gs_kernel (*kernel)(gs_gpu *g, const char *name);
    void (*kernel_release)(gs_gpu *g, gs_kernel k);
    /* global = total number of threads per dimension, local = block size */
    int (*launch)(gs_gpu *g, gs_kernel k, int dims, const size_t global[3], const size_t local[3], int nargs,
                  const gs_arg *args);
    int (*finish)(gs_gpu *g);
} gs_gpu_ops;

struct gs_gpu {
    gs_backend backend;
    char device_name[256];
    char platform_name[256];
    unsigned long long global_mem;
    char error[512]; /* last error message */
    const gs_gpu_ops *ops;
    void *impl;
};

/* Opens a device and builds the kernels found in kernel_dir. test_build=1 compiles the
 * smooth-loss variant used by the gradient self test. Returns 0 on success. */
int gs_gpu_init(gs_gpu *g, gs_backend backend, int device, const char *kernel_dir, int test_build, char *err,
                size_t err_len);
void gs_gpu_release(gs_gpu *g);
void gs_gpu_list_devices(void);
const char *gs_backend_name(gs_backend b);

/* backend entry points (gpu_cuda.c / gpu_opencl.c) */
int gs_cuda_init(gs_gpu *g, int device, const char *kernel_dir, int test_build, char *err, size_t err_len);
void gs_cuda_list_devices(void);
int gs_opencl_init(gs_gpu *g, int device, const char *kernel_dir, int test_build, char *err, size_t err_len);
void gs_opencl_list_devices(void);

/* helpers shared by the backends */
char *gs_gpu_load_source(const char *kernel_dir, const char *prelude, size_t *len);

static inline gs_arg gs_arg_buf(const gs_gpu *g, const gs_buf *b)
{
    gs_arg a;
    if (g->backend == GS_BACKEND_CUDA) {
        a.ptr = &b->u.dev;
        a.size = sizeof(b->u.dev);
    } else {
        a.ptr = &b->u.ptr;
        a.size = sizeof(b->u.ptr);
    }
    return a;
}

#endif
