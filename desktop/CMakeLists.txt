cmake_minimum_required(VERSION 3.13)
project(gsplat_server C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA backend: the NVIDIA driver API is loaded at runtime (libcuda / nvcuda.dll),
# so building it needs no CUDA toolkit. Kernels are compiled at runtime with NVRTC;
# if nvcc is found, a PTX fallback is built as well.
option(GS_WITH_CUDA "CUDA backend (NVIDIA GPUs)" ON)
# OpenCL backend: AMD/Intel GPUs or CPU (PoCL) for testing; used if the SDK is found.
option(GS_WITH_OPENCL "OpenCL backend" ON)
set(GS_CUDA_ARCH "86" CACHE STRING "compute capability for the prebuilt PTX (86 = RTX 30xx)")

find_package(Threads REQUIRED)

set(SOURCES
  src/main.c src/server.c src/http.c src/json.c src/platform.c
  src/dataset.c src/model.c src/trainer.c src/gpu.c)
if(GS_WITH_CUDA)
  list(APPEND SOURCES src/gpu_cuda.c)
endif()
if(GS_WITH_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    list(APPEND SOURCES src/gpu_opencl.c)
  else()
    message(STATUS "OpenCL SDK not found - building without the OpenCL backend")
    set(GS_WITH_OPENCL OFF)
  endif()
endif()
if(NOT GS_WITH_CUDA AND NOT GS_WITH_OPENCL)
  message(FATAL_ERROR "enable at least one GPU backend (GS_WITH_CUDA / GS_WITH_OPENCL)")
endif()

add_executable(gsplat-server ${SOURCES})
target_link_libraries(gsplat-server PRIVATE Threads::Threads)
if(GS_WITH_CUDA)
  target_compile_definitions(gsplat-server PRIVATE GS_WITH_CUDA)
  target_link_libraries(gsplat-server PRIVATE ${CMAKE_DL_LIBS})
endif()
if(GS_WITH_OPENCL)
  target_compile_definitions(gsplat-server PRIVATE GS_WITH_OPENCL)
  target_link_libraries(gsplat-server PRIVATE OpenCL::OpenCL)
endif()
if(WIN32)
  target_link_libraries(gsplat-server PRIVATE ws2_32)
else()
  target_link_libraries(gsplat-server PRIVATE m)
endif()

if(MSVC)
  target_compile_options(gsplat-server PRIVATE /W3)
  target_compile_definitions(gsplat-server PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(gsplat-server PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
target_compile_definitions(gsplat-server PRIVATE GS_KERNEL_DIR="${CMAKE_INSTALL_PREFIX}/share/gsplat-server")

# kernel sources are needed at runtime: copy them next to the executable
set(KERNEL_FILES kernels/gsplat_kernels.h kernels/prelude_cuda.h kernels/prelude_opencl.h kernels/gsplat.cu)
foreach(f ${KERNEL_FILES})
  get_filename_component(name ${f} NAME)
  add_custom_command(TARGET gsplat-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:gsplat-server>/kernels
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${f}
            $<TARGET_FILE_DIR:gsplat-server>/kernels/${name})
endforeach()

# optional prebuilt PTX (used when NVRTC is not installed)
if(GS_WITH_CUDA)
  find_program(GS_NVCC nvcc HINTS $ENV{CUDA_PATH}/bin /usr/local/cuda/bin)
  if(GS_NVCC)
    set(PTX_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)
    add_custom_command(OUTPUT ${PTX_DIR}/gsplat.ptx ${PTX_DIR}/gsplat_test.ptx
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PTX_DIR}
      COMMAND ${GS_NVCC} -ptx -O3 -arch=compute_${GS_CUDA_ARCH} -DTILE=16
              ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gsplat.cu -o ${PTX_DIR}/gsplat.ptx
      COMMAND ${GS_NVCC} -ptx -O3 -arch=compute_${GS_CUDA_ARCH} -DTILE=16 -DGS_FULL_TILES
              ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gsplat.cu -o ${PTX_DIR}/gsplat_test.ptx
      DEPENDS ${KERNEL_FILES}
      COMMENT "nvcc: building PTX for compute_${GS_CUDA_ARCH}")
    add_custom_target(gsplat_ptx ALL DEPENDS ${PTX_DIR}/gsplat.ptx ${PTX_DIR}/gsplat_test.ptx)
    add_dependencies(gsplat-server gsplat_ptx)
    add_custom_command(TARGET gsplat-server POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PTX_DIR}/gsplat.ptx ${PTX_DIR}/gsplat_test.ptx
              $<TARGET_FILE_DIR:gsplat-server>/kernels/)
    install(FILES ${PTX_DIR}/gsplat.ptx ${PTX_DIR}/gsplat_test.ptx DESTINATION share/gsplat-server)
  else()
    message(STATUS "nvcc not found - kernels will be compiled at runtime with NVRTC")
  endif()
endif()

install(TARGETS gsplat-server RUNTIME DESTINATION bin)
install(FILES ${KERNEL_FILES} DESTINATION share/gsplat-server)
