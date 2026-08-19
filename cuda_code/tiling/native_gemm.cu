#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 32

// ============================================================
// Kernel 1: Naive GEMM (baseline)
// ============================================================
__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ============================================================
// Kernel 2: Shared Memory Tiled GEMM (Block-level tiling only)
// ============================================================
__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // load A tile
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // load B tile
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // compute on shared memory
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================
// Kernel 3: Warp-level + Thread-level Tiled GEMM
// Block Tile (64x64 in Shared Memory)
//   -> Thread Tile (8x8 per thread, in Registers)
//
// Key improvement over Kernel 2:
//   - Inner loop: Shared Memory -> Register -> compute (1 cycle vs 20 cycles)
//   - Each thread computes 8x8=64 C elements (not just 1)
//   - float4 for global->shared cooperative loads
// ============================================================
#define BLOCK_TILE 64
#define THREAD_TILE 8

__global__ void sgemm_warp_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // ---- Shared Memory (Block-level tile) ----
    // BLOCK_TILE=64, two tiles = 64*64*4*2 = 32 KB, fits in 48 KB shared memory
    __shared__ float As[BLOCK_TILE][BLOCK_TILE];
    __shared__ float Bs[BLOCK_TILE][BLOCK_TILE];

    // ---- Thread position within the 8x8 thread block ----
    int tx = threadIdx.x;  // 0..7
    int ty = threadIdx.y;  // 0..7

    // ---- Global position of this thread's C fragment (THREAD_TILE x THREAD_TILE) ----
    int c_row = blockIdx.y * BLOCK_TILE + ty * THREAD_TILE;
    int c_col = blockIdx.x * BLOCK_TILE + tx * THREAD_TILE;

    // ---- Registers: Thread-level C fragment (8x8 = 64 accumulators) ----
    float c_frag[THREAD_TILE][THREAD_TILE] = {0.0f};

    // ---- Registers: A and B fragments loaded from Shared Memory ----
    float a_reg[THREAD_TILE];
    float b_reg[THREAD_TILE];

    int num_block_tiles = (K + BLOCK_TILE - 1) / BLOCK_TILE;
    int total_threads = blockDim.x * blockDim.y;  // 8 * 8 = 64

    // ---- Outer loop: traverse K dimension in BLOCK_TILE steps ----
    for (int bk = 0; bk < num_block_tiles; bk++) {

        // ============================================================
        // (1) Cooperative load: Global Memory -> Shared Memory
        // 64 threads load 64x64 = 4096 elements.
        // Each thread loads 4096/64 = 64 elements = 16 float4 loads
        // ============================================================
        for (int offset = 0; offset < BLOCK_TILE * BLOCK_TILE; offset += total_threads) {
            int idx = offset + ty * blockDim.x + tx;
            int smem_row = idx / BLOCK_TILE;
            int smem_col = idx % BLOCK_TILE;

            // Load A tile
            int g_row_A = blockIdx.y * BLOCK_TILE + smem_row;
            int g_col_A = bk * BLOCK_TILE + smem_col;
            As[smem_row][smem_col] = (g_row_A < M && g_col_A < K)
                ? A[g_row_A * K + g_col_A] : 0.0f;

            // Load B tile
            int g_row_B = bk * BLOCK_TILE + smem_row;
            int g_col_B = blockIdx.x * BLOCK_TILE + smem_col;
            Bs[smem_row][smem_col] = (g_row_B < K && g_col_B < N)
                ? B[g_row_B * N + g_col_B] : 0.0f;
        }

        __syncthreads();

        // ============================================================
        // (2) Inner loop: Shared Memory -> Register -> Compute
        // Each iteration:
        //   - Load THREAD_TILE elements from As into a_reg (register, ~20 cyc each)
        //   - Load THREAD_TILE elements from Bs into b_reg (register, ~20 cyc each)
        //   - Multiply all pairs: 64 FMAs in registers (~1 cyc each)
        // Reuse ratio: 64 FMAs / 16 shared memory loads = 4x
        // ============================================================
        for (int k = 0; k < BLOCK_TILE; k++) {

            // Load A fragment: THREAD_TILE consecutive elements from As
            // As[ty*THREAD_TILE .. ty*THREAD_TILE+THREAD_TILE-1][k]
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty * THREAD_TILE + i][k];
            }

            // Load B fragment: THREAD_TILE consecutive elements from Bs
            // Bs[k][tx*THREAD_TILE .. tx*THREAD_TILE+THREAD_TILE-1]
            // These are in the same row -> consecutive in memory -> bank-conflict-free
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                b_reg[j] = Bs[k][tx * THREAD_TILE + j];
            }

            // Register x Register -> Register (outer product)
            // All data in registers, ~1 cycle per FMA
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    c_frag[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // ============================================================
    // (3) Write back: Register -> Global Memory
    // ============================================================
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = c_row + i;
            int col = c_col + j;
            if (row < M && col < N) {
                C[row * N + col] = c_frag[i][j];
            }
        }
    }
}

// ============================================================
// CPU reference (for correctness check)
// ============================================================
void sgemm_cpu(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

typedef void (*gemm_kernel_t)(const float*, const float*, float*, int, int, int);

// ============================================================
// Timing helper
// ============================================================
float time_kernel(gemm_kernel_t kernel, dim3 grid, dim3 block,
                  float* d_A, float* d_B, float* d_C,
                  int M, int N, int K, int warmup, int repeat) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // timed runs
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

// ============================================================
// main
// ============================================================
int main() {
    int M = 2048, N = 2048, K = 2048;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // allocate GPU memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_warp;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C_naive, bytes_C);
    cudaMalloc(&d_C_tiled, bytes_C);
    cudaMalloc(&d_C_warp, bytes_C);

    // generate test data (A=1.0, B=2.0 for easy verification)
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // ---- launch configs ----
    dim3 threads_naive(16, 16);
    dim3 blocks_naive(
        (N + 15) / 16,
        (M + 15) / 16
    );

    dim3 threads_tiled(TILE, TILE);            // 32x32 = 1024 threads
    dim3 blocks_tiled(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // Warp-tiled: 8x8=64 threads per block, each computes 8x8 C elements
    // Covers 64x64 C per block
    dim3 threads_warp(THREAD_TILE, THREAD_TILE);  // 8x8 = 64 threads
    dim3 blocks_warp(
        (N + BLOCK_TILE - 1) / BLOCK_TILE,
        (M + BLOCK_TILE - 1) / BLOCK_TILE
    );

    // ---- run and time ----
    printf("Matrix: M=%d N=%d K=%d\n\n", M, N, K);

    float ms_naive = time_kernel(
        sgemm_naive, blocks_naive, threads_naive,
        d_A, d_B, d_C_naive, M, N, K, 3, 10
    );
    printf("Naive GEMM:          %8.3f ms\n", ms_naive);

    float ms_tiled = time_kernel(
        sgemm_tiled, blocks_tiled, threads_tiled,
        d_A, d_B, d_C_tiled, M, N, K, 3, 10
    );
    printf("Tiled GEMM:          %8.3f ms  (speedup: %.2fx)\n",
           ms_tiled, ms_naive / ms_tiled);

    float ms_warp = time_kernel(
        sgemm_warp_tiled, blocks_warp, threads_warp,
        d_A, d_B, d_C_warp, M, N, K, 3, 10
    );
    printf("Warp-Tiled GEMM:     %8.3f ms  (speedup: %.2fx)\n",
           ms_warp, ms_naive / ms_warp);

    // ---- verify correctness ----
    float *h_C_naive = (float*)malloc(bytes_C);
    float *h_C_tiled = (float*)malloc(bytes_C);
    float *h_C_warp  = (float*)malloc(bytes_C);
    float *h_C_ref   = (float*)malloc(bytes_C);

    cudaMemcpy(h_C_naive, d_C_naive, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled, d_C_tiled, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_warp,  d_C_warp,  bytes_C, cudaMemcpyDeviceToHost);

    // ---- CPU reference (also timed) ----
    printf("Running CPU reference...\n");
    cudaEvent_t cpu_start, cpu_stop;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);
    cudaEventRecord(cpu_start);
    sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);
    cudaEventRecord(cpu_stop);
    cudaEventSynchronize(cpu_stop);
    float ms_cpu;
    cudaEventElapsedTime(&ms_cpu, cpu_start, cpu_stop);
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_stop);

    // ---- Compare each GPU kernel vs CPU reference ----
    int errors_naive = 0, errors_tiled = 0, errors_warp = 0;
    float max_err_naive = 0.0f, max_err_tiled = 0.0f, max_err_warp = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float e_n = fabsf(h_C_naive[i] - h_C_ref[i]);
        float e_t = fabsf(h_C_tiled[i] - h_C_ref[i]);
        float e_w = fabsf(h_C_warp[i]  - h_C_ref[i]);
        if (e_n > 1e-2f) errors_naive++;
        if (e_t > 1e-2f) errors_tiled++;
        if (e_w > 1e-2f) errors_warp++;
        if (e_n > max_err_naive) max_err_naive = e_n;
        if (e_t > max_err_tiled) max_err_tiled = e_t;
        if (e_w > max_err_warp) max_err_warp = e_w;
    }

    printf("\n--- Correctness (vs CPU reference) ---\n");
    printf("Naive vs CPU:      %d errors / %d  (max diff: %.2e)\n",
           errors_naive, M * N, max_err_naive);
    printf("Tiled vs CPU:      %d errors / %d  (max diff: %.2e)\n",
           errors_tiled, M * N, max_err_tiled);
    printf("Warp-tiled vs CPU: %d errors / %d  (max diff: %.2e)\n",
           errors_warp,  M * N, max_err_warp);

    // ---- GFLOPS ----
    float gflops = 2.0f * M * N * K / 1e9;
    printf("\n--- Performance ---\n");
    printf("%-20s %10s %10s %10s\n", "Kernel", "Time(ms)", "GFLOPS", "Speedup");
    printf("%-20s %10.3f %10.1f %10s\n",
           "CPU (reference)", ms_cpu, gflops / (ms_cpu / 1000.0f), "1.00x");
    printf("%-20s %10.3f %10.1f %9.2fx\n",
           "GPU Naive", ms_naive, gflops / (ms_naive / 1000.0f),
           ms_cpu / ms_naive);
    printf("%-20s %10.3f %10.1f %9.2fx\n",
           "GPU Tiled", ms_tiled, gflops / (ms_tiled / 1000.0f),
           ms_cpu / ms_tiled);
    printf("%-20s %10.3f %10.1f %9.2fx\n",
           "GPU Warp-Tiled", ms_warp, gflops / (ms_warp / 1000.0f),
           ms_cpu / ms_warp);

    // ---- cleanup ----
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C_naive); cudaFree(d_C_tiled); cudaFree(d_C_warp);
    free(h_A); free(h_B);
    free(h_C_naive); free(h_C_tiled); free(h_C_warp); free(h_C_ref);

    return 0;
}
