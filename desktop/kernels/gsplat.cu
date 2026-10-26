/*
 * gsplat.cu - CUDA build of the Gaussian Splatting kernels.
 *
 * gsplat-server compiles this file at runtime with NVRTC for the installed GPU
 * (e.g. sm_86 for an RTX 3080). The prelude and kernel files are concatenated
 * by the host in that case; with nvcc (build time PTX) the includes below are used.
 */
#ifndef GS_PRELUDE_INCLUDED
#include "prelude_cuda.h"
#include "gsplat_kernels.h"
#endif
