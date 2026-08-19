cmake_minimum_required(VERSION 3.18)
project(ai-infra-cuda LANGUAGES CXX CUDA)

# ============================================================
# Encoding fix: force MSVC to treat all source files as UTF-8
# This is THE fundamental fix for Chinese Windows (code page 936)
# ============================================================
if(MSVC)
    add_compile_options("/utf-8")
endif()

# CUDA architecture - adjust to your GPU
# RTX 30 series = 80, RTX 40 series = 89, RTX 50 series = 100
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4060 = sm_89 (Ada Lovelace)

# ---- Week 3: native_gemm ----
add_executable(native_gemm tiling/native_gemm.cu)
