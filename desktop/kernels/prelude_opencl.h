/*
 * prelude_opencl.h - maps the kernel dialect of gsplat_kernels.h to OpenCL C 1.2.
 * The host prepends this file to gsplat_kernels.h before building the program.
 */
#define KERNEL __kernel
#define TILE_KERNEL __kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
#define DEVICE inline
#define GLOBAL __global
#define CONSTANT __constant
#define CONST_DATA __constant
#define SHARED __local
#define SYNC() barrier(CLK_LOCAL_MEM_FENCE)

#define GID_X ((int)get_global_id(0))
#define GID_Y ((int)get_global_id(1))
#define GID_Z ((int)get_global_id(2))
#define LID_X ((int)get_local_id(0))
#define LID_Y ((int)get_local_id(1))
#define GRP_X ((int)get_group_id(0))
#define GRP_Y ((int)get_group_id(1))

#define F_EXP exp
#define F_SQRT sqrt
#define F_CEIL ceil
#define F_MIN fmin
#define F_MAX fmax
#define I_MIN min
#define I_MAX max

#define SHARED_INC(p) atomic_inc(p)
#define ATOMIC_ADD(p, v) gs_atomic_add_f((p), (v))

/* OpenCL 1.2 has no float atomics: compare-and-swap loop */
inline void gs_atomic_add_f(volatile __global float *addr, float val)
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
