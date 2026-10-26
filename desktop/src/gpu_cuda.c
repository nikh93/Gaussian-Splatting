/*
 * gpu_cuda.c - CUDA backend (NVIDIA GPUs, e.g. RTX 3080).
 *
 * Pure C: the CUDA driver API (libcuda.so / nvcuda.dll, part of the NVIDIA
 * driver) is loaded at runtime, so building needs no CUDA toolkit. The kernels
 * (kernels/gsplat_kernels.h) are compiled when the server starts:
 *   1. with NVRTC (libnvrtc / nvrtc64_*.dll from the CUDA toolkit) for exactly the
 *      compute capability of the installed GPU, or
 *   2. from kernels/gsplat.ptx (resp. gsplat_test.ptx) built by nvcc at build time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu.h"
#include "platform.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  define CUDAAPI __stdcall
#else
#  include <dlfcn.h>
#  define CUDAAPI
#endif

/* ---- minimal driver API declarations (ABI stable since CUDA 4) ----------- */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_JIT_INFO_LOG_BUFFER 3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6

typedef struct {
    CUresult(CUDAAPI *Init)(unsigned int);
    CUresult(CUDAAPI *DriverGetVersion)(int *);
    CUresult(CUDAAPI *DeviceGetCount)(int *);
    CUresult(CUDAAPI *DeviceGet)(CUdevice *, int);
    CUresult(CUDAAPI *DeviceGetName)(char *, int, CUdevice);
    CUresult(CUDAAPI *DeviceTotalMem)(size_t *, CUdevice);
    CUresult(CUDAAPI *DeviceGetAttribute)(int *, int, CUdevice);
    CUresult(CUDAAPI *DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult(CUDAAPI *DevicePrimaryCtxRelease)(CUdevice);
    CUresult(CUDAAPI *CtxSetCurrent)(CUcontext);
    CUresult(CUDAAPI *CtxSynchronize)(void);
    CUresult(CUDAAPI *ModuleLoadDataEx)(CUmodule *, const void *, unsigned int, int *, void **);
    CUresult(CUDAAPI *ModuleUnload)(CUmodule);
    CUresult(CUDAAPI *ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult(CUDAAPI *MemAlloc)(CUdeviceptr *, size_t);
    CUresult(CUDAAPI *MemFree)(CUdeviceptr);
    CUresult(CUDAAPI *MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult(CUDAAPI *MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult(CUDAAPI *MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult(CUDAAPI *LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                    unsigned, CUstream, void **, void **);
    CUresult(CUDAAPI *GetErrorString)(CUresult, const char **);
} cuda_api;

/* ---- NVRTC ----------------------------------------------------------------- */
typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;
typedef struct {
    nvrtcResult (*Version)(int *, int *);
    nvrtcResult (*CreateProgram)(nvrtcProgram *, const char *, const char *, int, const char *const *,
                                 const char *const *);
    nvrtcResult (*CompileProgram)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*GetPTXSize)(nvrtcProgram, size_t *);
    nvrtcResult (*GetPTX)(nvrtcProgram, char *);
    nvrtcResult (*GetProgramLogSize)(nvrtcProgram, size_t *);
    nvrtcResult (*GetProgramLog)(nvrtcProgram, char *);
    nvrtcResult (*DestroyProgram)(nvrtcProgram *);
    const char *(*GetErrorString)(nvrtcResult);
} nvrtc_api;

typedef struct {
    void *lib;
    cuda_api cu;
    CUdevice dev;
    CUcontext ctx;
    CUmodule module;
} cuda_impl;

#define IMPL(g) ((cuda_impl *)(g)->impl)

/* ---------------------------------------------------------------------- */
/* dynamic loading                                                         */
/* ---------------------------------------------------------------------- */

static void *lib_open(const char *name)
{
#ifdef _WIN32
    /* full paths: let the DLL find its siblings (nvrtc-builtins) */
    if (strchr(name, '\\') || strchr(name, '/'))
        return (void *)LoadLibraryExA(name, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *lib_sym(void *lib, const char *name)
{
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void lib_close(void *lib)
{
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

#define LOAD(api, field, sym)                                                   \
    do {                                                                        \
        *(void **)&(api)->field = lib_sym(lib, sym);                            \
        if (!(api)->field) {                                                    \
            missing = sym;                                                      \
            goto missing_symbol;                                                \
        }                                                                       \
    } while (0)

static void *load_cuda(cuda_api *api, char *err, size_t err_len)
{
#ifdef _WIN32
    static const char *names[] = {"nvcuda.dll", NULL};
#else
    static const char *names[] = {"libcuda.so.1", "libcuda.so", NULL};
#endif
    const char *missing = NULL;
    void *lib = NULL;
    int i;
    for (i = 0; names[i] && !lib; i++)
        lib = lib_open(names[i]);
    if (!lib) {
        snprintf(err, err_len, "NVIDIA driver library (%s) not found", names[0]);
        return NULL;
    }
    LOAD(api, Init, "cuInit");
    LOAD(api, DriverGetVersion, "cuDriverGetVersion");
    LOAD(api, DeviceGetCount, "cuDeviceGetCount");
    LOAD(api, DeviceGet, "cuDeviceGet");
    LOAD(api, DeviceGetName, "cuDeviceGetName");
    LOAD(api, DeviceTotalMem, "cuDeviceTotalMem_v2");
    LOAD(api, DeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD(api, DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    *(void **)&api->DevicePrimaryCtxRelease = lib_sym(lib, "cuDevicePrimaryCtxRelease_v2");
    if (!api->DevicePrimaryCtxRelease)
        LOAD(api, DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease");
    LOAD(api, CtxSetCurrent, "cuCtxSetCurrent");
    LOAD(api, CtxSynchronize, "cuCtxSynchronize");
    LOAD(api, ModuleLoadDataEx, "cuModuleLoadDataEx");
    LOAD(api, ModuleUnload, "cuModuleUnload");
    LOAD(api, ModuleGetFunction, "cuModuleGetFunction");
    LOAD(api, MemAlloc, "cuMemAlloc_v2");
    LOAD(api, MemFree, "cuMemFree_v2");
    LOAD(api, MemcpyHtoD, "cuMemcpyHtoD_v2");
    LOAD(api, MemcpyDtoH, "cuMemcpyDtoH_v2");
    LOAD(api, MemsetD8, "cuMemsetD8_v2");
    LOAD(api, LaunchKernel, "cuLaunchKernel");
    LOAD(api, GetErrorString, "cuGetErrorString");
    return lib;
missing_symbol:
    snprintf(err, err_len, "NVIDIA driver too old (missing %s)", missing);
    lib_close(lib);
    return NULL;
}

static void *load_nvrtc(nvrtc_api *api)
{
    const char *missing = NULL;
    void *lib = NULL;
    const char *env = getenv("GS_NVRTC");
    char path[1100];
    int i;
#ifdef _WIN32
    static const char *names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll",
                                  "nvrtc64_111_0.dll", NULL};
    const char *cuda_path = getenv("CUDA_PATH");
#else
    static const char *names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2",
                                  "/usr/local/cuda/lib64/libnvrtc.so", NULL};
#endif
    if (env)
        lib = lib_open(env);
    for (i = 0; names[i] && !lib; i++) {
        lib = lib_open(names[i]);
#ifdef _WIN32
        if (!lib && cuda_path) {
            snprintf(path, sizeof(path), "%s\\bin\\%s", cuda_path, names[i]);
            lib = lib_open(path);
            if (!lib) {
                snprintf(path, sizeof(path), "%s\\bin\\x64\\%s", cuda_path, names[i]);
                lib = lib_open(path);
            }
        }
#endif
    }
    (void)path;
    if (!lib)
        return NULL;
    LOAD(api, Version, "nvrtcVersion");
    LOAD(api, CreateProgram, "nvrtcCreateProgram");
    LOAD(api, CompileProgram, "nvrtcCompileProgram");
    LOAD(api, GetPTXSize, "nvrtcGetPTXSize");
    LOAD(api, GetPTX, "nvrtcGetPTX");
    LOAD(api, GetProgramLogSize, "nvrtcGetProgramLogSize");
    LOAD(api, GetProgramLog, "nvrtcGetProgramLog");
    LOAD(api, DestroyProgram, "nvrtcDestroyProgram");
    LOAD(api, GetErrorString, "nvrtcGetErrorString");
    return lib;
missing_symbol:
    fprintf(stderr, "[cuda] NVRTC library lacks %s, ignoring it\n", missing);
    lib_close(lib);
    return NULL;
}

/* Compiles the kernel source to PTX for compute capability `cc` (e.g. 86).
 * Returns malloc'd PTX or NULL. */
char *gs_nvrtc_compile(const char *src, int cc, int test_build, char *log_out, size_t log_len)
{
    nvrtc_api api;
    void *lib = load_nvrtc(&api);
    /* try the device architecture first, then older ones a given NVRTC knows */
    int archs[8], na = 0, a;
    char *ptx = NULL;
    if (log_out && log_len)
        log_out[0] = 0;
    if (!lib) {
        if (log_out)
            snprintf(log_out, log_len, "NVRTC library not found");
        return NULL;
    }
    archs[na++] = cc;
    if (cc > 90) archs[na++] = 90;
    if (cc > 86) archs[na++] = 86;
    if (cc > 80) archs[na++] = 80;
    if (cc > 75) archs[na++] = 75;
    {
        int major = 0, minor = 0;
        api.Version(&major, &minor);
        printf("[cuda] compiling kernels with NVRTC %d.%d\n", major, minor);
    }
    for (a = 0; a < na && !ptx; a++) {
        nvrtcProgram prog;
        char arch[64];
        const char *opts[5];
        int nopt = 0;
        nvrtcResult r;
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d", archs[a]);
        opts[nopt++] = arch;
        opts[nopt++] = "-DTILE=16";
        opts[nopt++] = "-DGS_PRELUDE_INCLUDED";
        if (test_build)
            opts[nopt++] = "-DGS_FULL_TILES";
        if (api.CreateProgram(&prog, src, "gsplat.cu", 0, NULL, NULL) != 0)
            break;
        r = api.CompileProgram(prog, nopt, opts);
        if (r == 0) {
            size_t n = 0;
            api.GetPTXSize(prog, &n);
            ptx = (char *)malloc(n + 1);
            api.GetPTX(prog, ptx);
            ptx[n] = 0;
            printf("[cuda] kernels compiled for compute_%d\n", archs[a]);
        } else {
            size_t n = 0;
            api.GetProgramLogSize(prog, &n);
            if (log_out && log_len && n > 1) {
                char *l = (char *)malloc(n + 1);
                api.GetProgramLog(prog, l);
                l[n] = 0;
                snprintf(log_out, log_len, "%s: %s", api.GetErrorString(r), l);
                free(l);
            } else if (log_out) {
                snprintf(log_out, log_len, "%s", api.GetErrorString(r));
            }
        }
        api.DestroyProgram(&prog);
    }
    lib_close(lib);
    return ptx;
}

/* ---------------------------------------------------------------------- */
/* ops                                                                     */
/* ---------------------------------------------------------------------- */

static int check(gs_gpu *g, CUresult r, const char *what)
{
    const char *s = NULL;
    if (r == CUDA_SUCCESS)
        return 0;
    if (IMPL(g) && IMPL(g)->cu.GetErrorString)
        IMPL(g)->cu.GetErrorString(r, &s);
    snprintf(g->error, sizeof(g->error), "%s: %s (%d)", what, s ? s : "CUDA error", r);
    fprintf(stderr, "[cuda] %s\n", g->error);
    return -1;
}

static void cu_release(gs_gpu *g)
{
    cuda_impl *c = IMPL(g);
    if (!c)
        return;
    if (c->ctx) {
        c->cu.CtxSetCurrent(c->ctx);
        if (c->module)
            c->cu.ModuleUnload(c->module);
        c->cu.DevicePrimaryCtxRelease(c->dev);
    }
    if (c->lib)
        lib_close(c->lib);
    free(c);
    g->impl = NULL;
}

static int cu_bind_thread(gs_gpu *g)
{
    return check(g, IMPL(g)->cu.CtxSetCurrent(IMPL(g)->ctx), "cuCtxSetCurrent");
}

static int cu_alloc(gs_gpu *g, gs_buf *b, size_t bytes)
{
    CUdeviceptr p = 0;
    b->bytes = bytes;
    b->u.dev = 0;
    if (check(g, IMPL(g)->cu.MemAlloc(&p, bytes ? bytes : 4), "cuMemAlloc") != 0)
        return -1;
    b->u.dev = p;
    return 0;
}

static void cu_free(gs_gpu *g, gs_buf *b)
{
    if (b->u.dev)
        IMPL(g)->cu.MemFree((CUdeviceptr)b->u.dev);
    b->u.dev = 0;
    b->bytes = 0;
}

static int cu_write(gs_gpu *g, gs_buf *b, size_t off, size_t bytes, const void *src)
{
    if (!bytes)
        return 0;
    return check(g, IMPL(g)->cu.MemcpyHtoD((CUdeviceptr)b->u.dev + off, src, bytes), "cuMemcpyHtoD");
}

static int cu_read(gs_gpu *g, const gs_buf *b, size_t off, size_t bytes, void *dst)
{
    if (!bytes)
        return 0;
    return check(g, IMPL(g)->cu.MemcpyDtoH(dst, (CUdeviceptr)b->u.dev + off, bytes), "cuMemcpyDtoH");
}

static int cu_zero(gs_gpu *g, gs_buf *b, size_t bytes)
{
    if (!bytes)
        return 0;
    return check(g, IMPL(g)->cu.MemsetD8((CUdeviceptr)b->u.dev, 0, bytes), "cuMemsetD8");
}

static gs_kernel cu_kernel(gs_gpu *g, const char *name)
{
    CUfunction f = NULL;
    if (check(g, IMPL(g)->cu.ModuleGetFunction(&f, IMPL(g)->module, name), name) != 0)
        return NULL;
    return (gs_kernel)f;
}

static void cu_kernel_release(gs_gpu *g, gs_kernel k)
{
    (void)g;
    (void)k; /* functions belong to the module */
}

static int cu_launch(gs_gpu *g, gs_kernel k, int dims, const size_t global[3], const size_t local[3], int nargs,
                     const gs_arg *args)
{
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    void *params[32];
    int i;
    if (nargs > 32)
        return -1;
    for (i = 0; i < dims; i++) {
        if (global[i] == 0)
            return 0;
        block[i] = (unsigned)local[i];
        grid[i] = (unsigned)((global[i] + local[i] - 1) / local[i]);
    }
    for (i = 0; i < nargs; i++)
        params[i] = (void *)args[i].ptr;
    return check(g,
                 IMPL(g)->cu.LaunchKernel((CUfunction)k, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                                          NULL, params, NULL),
                 "cuLaunchKernel");
}

static int cu_finish(gs_gpu *g)
{
    return check(g, IMPL(g)->cu.CtxSynchronize(), "cuCtxSynchronize");
}

static const gs_gpu_ops cu_ops = {cu_release, cu_bind_thread, cu_alloc, cu_free,  cu_write,         cu_read,
                                  cu_zero,    cu_kernel,      cu_kernel_release,  cu_launch,        cu_finish};

/* ---------------------------------------------------------------------- */

void gs_cuda_list_devices(void)
{
    cuda_api api;
    char err[256];
    void *lib = load_cuda(&api, err, sizeof(err));
    int n = 0, i;
    if (!lib) {
        printf("  (none: %s)\n", err);
        return;
    }
    if (api.Init(0) != CUDA_SUCCESS || api.DeviceGetCount(&n) != CUDA_SUCCESS || n == 0) {
        printf("  (none)\n");
        lib_close(lib);
        return;
    }
    for (i = 0; i < n; i++) {
        CUdevice d;
        char name[256] = "?";
        size_t mem = 0;
        int ma = 0, mi = 0;
        api.DeviceGet(&d, i);
        api.DeviceGetName(name, sizeof(name), d);
        api.DeviceTotalMem(&mem, d);
        api.DeviceGetAttribute(&ma, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
        api.DeviceGetAttribute(&mi, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
        printf("  [%d] %s (sm_%d%d, %.1f GB)\n", i, name, ma, mi, (double)mem / (1024.0 * 1024.0 * 1024.0));
    }
    lib_close(lib);
}

int gs_cuda_init(gs_gpu *g, int device, const char *kernel_dir, int test_build, char *err, size_t err_len)
{
    cuda_impl *c = (cuda_impl *)calloc(1, sizeof(cuda_impl));
    int n = 0, ma = 0, mi = 0, drv = 0;
    size_t mem = 0;
    char *ptx = NULL;
    char log[4096];

    g->impl = c;
    g->ops = &cu_ops;
    g->backend = GS_BACKEND_CUDA;
    c->lib = load_cuda(&c->cu, err, err_len);
    if (!c->lib)
        goto fail;
    if (c->cu.Init(0) != CUDA_SUCCESS || c->cu.DeviceGetCount(&n) != CUDA_SUCCESS || n == 0) {
        snprintf(err, err_len, "no CUDA device found");
        goto fail;
    }
    if (device < 0)
        device = 0;
    if (device >= n) {
        snprintf(err, err_len, "CUDA device %d does not exist (%d available)", device, n);
        goto fail;
    }
    c->cu.DeviceGet(&c->dev, device);
    c->cu.DeviceGetName(g->device_name, sizeof(g->device_name), c->dev);
    c->cu.DeviceTotalMem(&mem, c->dev);
    c->cu.DeviceGetAttribute(&ma, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, c->dev);
    c->cu.DeviceGetAttribute(&mi, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, c->dev);
    c->cu.DriverGetVersion(&drv);
    g->global_mem = mem;
    snprintf(g->platform_name, sizeof(g->platform_name), "CUDA %d.%d driver, sm_%d%d", drv / 1000,
             (drv % 1000) / 10, ma, mi);
    if (c->cu.DevicePrimaryCtxRetain(&c->ctx, c->dev) != CUDA_SUCCESS ||
        c->cu.CtxSetCurrent(c->ctx) != CUDA_SUCCESS) {
        snprintf(err, err_len, "cannot create CUDA context");
        c->ctx = NULL;
        goto fail;
    }

    /* kernels: NVRTC for this GPU, else prebuilt PTX */
    {
        char *src = gs_gpu_load_source(kernel_dir, "prelude_cuda.h", NULL);
        if (src) {
            ptx = gs_nvrtc_compile(src, ma * 10 + mi, test_build, log, sizeof(log));
            free(src);
            if (!ptx)
                fprintf(stderr, "[cuda] runtime compilation unavailable: %s\n", log);
        }
        if (!ptx) {
            char p[1100];
            snprintf(p, sizeof(p), "%s/%s", kernel_dir, test_build ? "gsplat_test.ptx" : "gsplat.ptx");
            ptx = gs_read_file(p, NULL);
            if (ptx)
                printf("[cuda] using prebuilt %s\n", p);
            else {
                snprintf(err, err_len,
                         "no kernels: install the CUDA toolkit (NVRTC) or build gsplat.ptx with nvcc (%s)", log);
                goto fail;
            }
        }
    }
    {
        char jit_log[8192];
        int opts[2] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
        void *vals[2];
        CUresult r;
        jit_log[0] = 0;
        vals[0] = jit_log;
        vals[1] = (void *)(size_t)sizeof(jit_log);
        r = c->cu.ModuleLoadDataEx(&c->module, ptx, 2, opts, vals);
        free(ptx);
        ptx = NULL;
        if (r != CUDA_SUCCESS) {
            const char *s = NULL;
            c->cu.GetErrorString(r, &s);
            fprintf(stderr, "[cuda] PTX JIT log:\n%s\n", jit_log);
            snprintf(err, err_len, "cuModuleLoadData: %s%s", s ? s : "error",
                     r == 222 ? " - please update the NVIDIA driver" : ""); /* CUDA_ERROR_UNSUPPORTED_PTX_VERSION */
            c->module = NULL;
            goto fail;
        }
    }
    return 0;
fail:
    free(ptx);
    cu_release(g);
    g->ops = NULL;
    return -1;
}
