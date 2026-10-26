cmake_minimum_required(VERSION 3.13)
project(gsplat_server C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

add_executable(gsplat-server
  src/main.c
  src/server.c
  src/http.c
  src/json.c
  src/platform.c
  src/dataset.c
  src/model.c
  src/cl_util.c
  src/trainer.c
)

target_link_libraries(gsplat-server PRIVATE OpenCL::OpenCL Threads::Threads)
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

# kernels are loaded at runtime: copy them next to the executable
add_custom_command(TARGET gsplat-server POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:gsplat-server>/kernels
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gsplat.cl
          $<TARGET_FILE_DIR:gsplat-server>/kernels/gsplat.cl)
target_compile_definitions(gsplat-server PRIVATE GS_KERNEL_DIR="${CMAKE_INSTALL_PREFIX}/share/gsplat-server")

install(TARGETS gsplat-server RUNTIME DESTINATION bin)
install(FILES kernels/gsplat.cl DESTINATION share/gsplat-server)
