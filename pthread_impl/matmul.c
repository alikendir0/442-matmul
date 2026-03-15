#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float *A, *B, *C;
int N = 512;          // default matrix size (512x512)
int num_threads = 4;  // default number of threads

typedef struct { int thread_id; } thread_arg_t;

// Pthreads Kernel (The main function that will run in parallel)
void* matmul_kernel(void* arg) {
    thread_arg_t *data = (thread_arg_t *)arg;

    // Row-wise block workload distribution: 
    // Each thread performs contiguous memory reads to maximize cache efficiency.
    int start = (data->thread_id) * (N / num_threads);
    
    // The last thread processes any remaining rows if N is not perfectly divisible by num_threads.
    int end = (data->thread_id == num_threads - 1) ? N : (data->thread_id + 1) * (N / num_threads);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum; // Local accumulation : Aim is minimize writes to main memory and prevent bandwidth bottlenecks.
        }
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);

    size_t size = N * N * sizeof(float);
    A = malloc(size); B = malloc(size); C = malloc(size);
    
    // Memory allocation and initialization are excluded from the timing.
    // As per Amdahl's Law, we isolate the parallel kernel's execution time 
    // to accurately measure the speedup, independent of the serial I/O overhead.
    for(int i=0; i<N*N; i++) { 
        A[i] = 1.0f; 
        B[i] = 1.0f; 
    }

    double total_time = 0.0;
    int num_runs = 3; // 3 run average for more reliable timing results.

    // Averages over 3 runs to mitigate execution noise caused by OS background tasks 
    // and context switching overhead.
    for (int run = 0; run < num_runs; run++) {
        pthread_t threads[num_threads];
        thread_arg_t args[num_threads];

        struct timespec start, end;

        // --- TIMER START ---
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (num_threads == 1) {
            // Single-thread: run kernel directly without pthread overhead.
            // This gives a fair serial baseline — just pure computation time,
            // analogous to how CUDA excludes context initialization.
            args[0].thread_id = 0;
            matmul_kernel(&args[0]);
        } else {
            for (int i = 0; i < num_threads; i++) {
                args[i].thread_id = i;
                pthread_create(&threads[i], NULL, matmul_kernel, &args[i]);
            }

            // Barrier synchronization : The main thread waits for all worker threads to complete their execution.
            for (int i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
            }
        }

        // --- TIMER END ---
        clock_gettime(CLOCK_MONOTONIC, &end);

        double current_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += current_time;
    }

    double average_time = total_time / num_runs;
    
    // Console I/O is strictly excluded from the performance timer in order to prevent latency overhead.
    printf("N: %d, Threads: %d, Avg Time (3 runs): %.9f s\n", N, num_threads, average_time);

    free(A); free(B); free(C);
    return 0;
}