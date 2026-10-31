# JNI bridge of the Cardboard SDK's Java API (com.google.cardboard.sdk.*), built the
# same way as Google's "paperscope" sample: java_api/src/main/jni/cardboard_jni.cc
# linked against the SDK library libGfxPluginCardboard.so from the SDK's AAR.
# Prepare the SDK first with android/cardboard/build_cardboard_sdk.sh.
cmake_minimum_required(VERSION 3.22.1)
project(gsplat_cardboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CARDBOARD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cardboard)
set(CARDBOARD_SRC ${CARDBOARD_DIR}/src)
set(CARDBOARD_LIB ${CARDBOARD_DIR}/lib/${ANDROID_ABI}/libGfxPluginCardboard.so)

if(NOT EXISTS ${CARDBOARD_LIB} OR NOT EXISTS ${CARDBOARD_SRC}/java_api/src/main/jni/cardboard_jni.cc)
    message(FATAL_ERROR "Cardboard SDK not prepared: run android/cardboard/build_cardboard_sdk.sh")
endif()

add_library(GfxPluginCardboard SHARED IMPORTED)
set_target_properties(GfxPluginCardboard PROPERTIES IMPORTED_LOCATION ${CARDBOARD_LIB})

find_library(LOG_LIB log)
find_library(ANDROID_LIB android)
find_library(GLESv2_LIB GLESv2)

# the library name is fixed by com.google.cardboard.sdk.Initialize (System.loadLibrary)
add_library(cardboard_sdk_jni SHARED
    ${CARDBOARD_SRC}/java_api/src/main/jni/cardboard_jni.cc
    ${CARDBOARD_SRC}/sdk/jni_utils/android/jni_utils.cc
)
target_include_directories(cardboard_sdk_jni PRIVATE ${CARDBOARD_SRC}/sdk)
target_link_libraries(cardboard_sdk_jni PRIVATE GfxPluginCardboard ${ANDROID_LIB} ${LOG_LIB} ${GLESv2_LIB})
target_link_options(cardboard_sdk_jni PRIVATE "-Wl,-z,max-page-size=16384")
