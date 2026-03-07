// Matrix Multiplication in CUDA
// Every line is commented. Read top to bottom.

#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free
#include <time.h>    // clock for timing

// ─────────────────────────────────────────────────────────────────────────────
// KEY CONCEPT:
//
// Sequential matmul has 3 nested loops:
//   for i:
//     for j:
//       for k:
//         C[i][j] += A[i][k] * B[k][j]
//
// In CUDA we kill the outer 2 loops.
// Instead we launch N*N threads — one per output element.
// Each thread handles exactly one C[row][col] independently.
// They all run at the same time.
// ─────────────────────────────────────────────────────────────────────────────

#define N 512          // Matrix size: N x N
#define BLOCK_SIZE 16  // Each block has 16x16 = 256 threads


// ─── THE GPU KERNEL ───────────────────────────────────────────────────────────
// __global__ means: this function runs ON THE GPU, called FROM THE CPU
// Every thread runs this same function simultaneously
// But each thread has a unique (row, col) so they work on different elements

__global__ void matmul(float *A, float *B, float *C) {

    // Each thread figures out WHICH element it is responsible for.
    // blockIdx  = which block  this thread is in  (in the grid)
    // threadIdx = which thread this is            (inside its block)
    // blockDim  = how big each block is           (16x16 in our case)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety check: the grid might be slightly larger than the matrix
    // so we ignore threads that fall outside the matrix bounds
    if (row < N && col < N) {

        float sum = 0.0f;

        // This is the only loop left — the inner k loop
        // This thread computes the dot product of row `row` of A
        // with column `col` of B
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
            //       ^               ^
            //       row-th row of A  col-th col of B
        }

        // Write the result into C
        // This thread owns this one element and no other thread touches it
        C[row * N + col] = sum;
    }
}


// ─── HELPER: fill a matrix with random values ─────────────────────────────────
void fill_random(float *M) {
    for (int i = 0; i < N * N; i++)
        M[i] = (float)(rand() % 10);
}


// ─── MAIN ─────────────────────────────────────────────────────────────────────
int main() {

    printf("Matrix size: %d x %d\n", N, N);
    printf("Block size:  %d x %d = %d threads per block\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*BLOCK_SIZE);

    // ── Step 1: Allocate memory on the CPU (called "host") ───────────────────
    // GPU and CPU have SEPARATE memory. We need matrices in both places.
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);  // h_ prefix = host (CPU)
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);  // result will land here

    srand(42);
    fill_random(h_A);
    fill_random(h_B);

    printf("\nStep 1: Matrices created on CPU RAM\n");


    // ── Step 2: Allocate memory on the GPU (called "device") ─────────────────
    float *d_A, *d_B, *d_C;  // d_ prefix = device (GPU)

    cudaMalloc(&d_A, bytes);  // reserve GPU memory
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    printf("Step 2: Memory reserved on GPU VRAM\n");


    // ── Step 3: Copy input matrices from CPU → GPU ────────────────────────────
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    printf("Step 3: Matrices copied CPU -> GPU\n");


    // ── Step 4: Define the thread layout ─────────────────────────────────────
    // blockDim:  how many threads per block  → 16 x 16
    // gridDim:   how many blocks in the grid → enough to cover N x N
    //
    // Example for N=512, BLOCK_SIZE=16:
    //   gridDim = (32, 32) → 1024 blocks total
    //   each block has 256 threads
    //   total threads = 1024 * 256 = 262144 = 512*512 ✓

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // columns
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // rows

    printf("Step 4: Thread layout defined\n");
    printf("        Grid:    %d x %d blocks\n",  gridDim.x,  gridDim.y);
    printf("        Block:   %d x %d threads\n", blockDim.x, blockDim.y);
    printf("        Total:   %d threads launched\n", gridDim.x * gridDim.y * BLOCK_SIZE * BLOCK_SIZE);


    // ── Step 5: Launch the kernel ─────────────────────────────────────────────
    // Syntax: kernel<<<gridDim, blockDim>>>(arguments)
    // This tells the GPU: spawn all those threads and run matmul()

    clock_t t0 = clock();

    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();  // CPU waits here until ALL GPU threads finish

    clock_t t1 = clock();
    double ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    printf("Step 5: Kernel done in %.3f ms\n", ms);


    // ── Step 6: Copy result from GPU → CPU ───────────────────────────────────
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Step 6: Result copied GPU -> CPU\n");
    printf("\nC[0][0] = %.1f  (sanity check, just to confirm it ran)\n", h_C[0]);


    // ── Step 7: Free all memory ───────────────────────────────────────────────
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Step 7: Memory freed. Done.\n");

    return 0;
}
