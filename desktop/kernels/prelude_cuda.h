/*
 * prelude_cuda.h - maps the kernel dialect of gsplat_kernels.h to CUDA C.
 * Must not include system headers: the source is also compiled with NVRTC.
 */
#define KERNEL extern "C" __global__
#define TILE_KERNEL extern "C" __global__ __launch_bounds__(TILE * TILE)
#define DEVICE __device__ __forceinline__
#define GLOBAL
#define CONSTANT const
#define CONST_DATA __constant__
#define SHARED __shared__
#define SYNC() __syncthreads()

#define GID_X ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define GID_Y ((int)(blockIdx.y * blockDim.y + threadIdx.y))
#define GID_Z ((int)(blockIdx.z * blockDim.z + threadIdx.z))
#define LID_X ((int)threadIdx.x)
#define LID_Y ((int)threadIdx.y)
#define GRP_X ((int)blockIdx.x)
#define GRP_Y ((int)blockIdx.y)

#define F_EXP expf
#define F_SQRT sqrtf
#define F_CEIL ceilf
#define F_MIN fminf
#define F_MAX fmaxf
#define I_MIN min
#define I_MAX max

#define SHARED_INC(p) atomicAdd((p), 1)
#define ATOMIC_ADD(p, v) atomicAdd((p), (v)) /* native float atomics */

typedef unsigned long long gs_size_t;
#define size_t gs_size_t
