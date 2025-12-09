#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>

#define N 1024

// Kernel function definitions (must be defined outside of main)
__global__ void initKernel(int *arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        arr[idx] = idx;
    }
}

__global__ void reverseCopyKernel(int *src, int *dst, int n)
{
    int indx = threadIdx.x;
    if (indx < n)
    {
        dst[n - 1 - indx] = src[indx];
    }
}

__global__ void reverseCopyKernelN(int *src, int *dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[n - 1 - idx] = src[idx];
    }
}

__global__ void reverseCopySharedKernel(int *src, int *dst, int n)
{
    __shared__ int sharedMem[256]; // Fixed size for shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n)
    {
        sharedMem[tid] = src[idx];
    }
    __syncthreads();

    if (idx < n)
    {
        dst[n - 1 - idx] = sharedMem[blockDim.x - 1 - tid];
    }
}

int main()
{
    //=====Aufgabe 1: CUDA Memory Operations Demo=====//
    std::cout << "CUDA Memory Operations Demo" << std::endl;
    std::cout << "============================\n"
              << std::endl;

    // Check for CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA devices found! Cannot execute CUDA operations.\n");
        return 1;
    }

    int num_elements = N;
    int num_bytes = num_elements * sizeof(int);

    // Host array
    int *h_a = (int *)malloc(num_bytes);

    // Initialize host array with values 0 to N-1
    printf("Initializing host array h_a with values 0 to %d\n", N - 1);
    for (int i = 0; i < num_elements; i++)
    {
        h_a[i] = i;
    }
    printf("First 10 values in h_a: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n\n");

    // Step 1: Allocate device memory d_a and d_b
    int *d_a, *d_b;
    printf("Step 1: Allocating device memory d_a and d_b (%d bytes each)\n", num_bytes);
    cudaMalloc((void **)&d_a, num_bytes);
    cudaMalloc((void **)&d_b, num_bytes);
    printf("Device memory allocated successfully\n\n");

    // Step 2: Copy from host h_a to device d_a
    printf("Step 2: Copying h_a to d_a (Host -> Device)\n");
    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    printf("Copy completed\n\n");

    // Step 3: Copy from device d_a to device d_b
    printf("Step 3: Copying d_a to d_b (Device -> Device)\n");
    cudaMemcpy(d_b, d_a, num_bytes, cudaMemcpyDeviceToDevice);
    printf("Copy completed\n\n");

    // Clear h_a to verify the copy back works
    printf("Clearing h_a to verify next step\n");
    for (int i = 0; i < num_elements; i++)
    {
        h_a[i] = -1;
    }
    printf("First 10 values in h_a after clearing: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n\n");

    // Step 4: Copy from device d_b to host h_a
    printf("Step 4: Copying d_b to h_a (Device -> Host)\n");
    cudaMemcpy(h_a, d_b, num_bytes, cudaMemcpyDeviceToHost);
    printf("Copy completed\n\n");

    // Verify the result
    printf("Verifying data after all copies:\n");
    printf("First 10 values in h_a: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n");
    printf("Last 10 values in h_a: ");
    for (int i = num_elements - 10; i < num_elements; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n\n");

    // Verify all values are correct
    bool success = true;
    for (int i = 0; i < num_elements; i++)
    {
        if (h_a[i] != i)
        {
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("✓ SUCCESS: All %d elements copied correctly through the chain:\n", num_elements);
        printf("  h_a -> d_a -> d_b -> h_a\n\n");
    }
    else
    {
        printf("✗ ERROR: Data mismatch detected!\n\n");
    }

    // Step 5: Free device memory
    printf("Step 5: Freeing device memory d_a and d_b\n");
    cudaFree(d_a);
    cudaFree(d_b);
    printf("Device memory freed\n\n");

    // Free host memory
    free(h_a);
    printf("Freed host memory for CUDA Memory Operations Demo\n\n");

    printf("Aufgabe 1 completed successfully!\n\n");

    //=====Aufgabe 2: Kernel Funktion=====//
    std::cout << "CUDA Kernel Function Demo" << std::endl;
    std::cout << "=========================\n"
              << std::endl;

    // 1. Allokiere Speicher h_a (host) und d_a (device)
    int *h_array = (int *)malloc(num_bytes);
    int *d_array;
    cudaMalloc((void **)&d_array, num_bytes);
    printf("Allocated host and device memory for kernel demo\n\n");

    // 2. Bestimme Dimensionierung des Gitters und rufe Kernel-Funktion auf
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // 3. Kernel-Funktion: schreibe das Ergebnis in das Feld d_a
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, num_elements);
    cudaDeviceSynchronize();
    printf("Kernel execution completed\n\n");

    // 4. Kopiere das Ergebnis von d_a nach h_a
    cudaMemcpy(h_array, d_array, num_bytes, cudaMemcpyDeviceToHost);
    printf("Copy from device to host completed\n\n");

    // 5. Vergleiche, ob das Ergebnis korrekt ist
    success = true;
    for (int i = 0; i < num_elements; i++)
    {
        if (h_array[i] != i)
        {
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("✓ SUCCESS: Kernel initialized all %d elements correctly.\n\n", num_elements);
    }
    else
    {
        printf("✗ ERROR: Kernel initialization failed!\n\n");
    }

    // Free device and host memory for kernel demo
    cudaFree(d_array);
    free(h_array);
    printf("Freed device and host memory for kernel demo\n\n");
    printf("Aufgabe 2 completed successfully!\n\n");

    //=====Aufgabe 3: Speicherbereich umgekehrt Kopieren, 1 Block=====//
    std::cout << "CUDA Reverse Copy Demo (1 Block)" << std::endl;
    std::cout << "==============================\n"
              << std::endl;

    // 1. Allokiere Speicher h_a (host) und d_a, d_b (device) mit jeweils 256 Elementen
    int num_elements_rev = 256;
    int num_bytes_rev = num_elements_rev * sizeof(int);
    int *h_rev = (int *)malloc(num_bytes_rev);
    int *d_a_rev, *d_b_rev;
    cudaMalloc((void **)&d_a_rev, num_bytes_rev);
    cudaMalloc((void **)&d_b_rev, num_bytes_rev);

    // Initialize host array
    for (int i = 0; i < num_elements_rev; i++)
    {
        h_rev[i] = i;
    }
    printf("Allocated host and device memory for reverse copy demo\n\n");

    // 2. Definiere Berechnungsgitter mit einem Block und 256 Threads
    int threads = 256;
    int blocks = 1;
    printf("Using %d block of %d threads for reverse copy kernel\n", blocks, threads);

    // 3. Kopiere Speicher von h_a nach d_a
    cudaMemcpy(d_a_rev, h_rev, num_bytes_rev, cudaMemcpyHostToDevice);
    printf("Copied data from host to device d_a_rev\n\n");

    // 4. Kernel-Funktion: Kopiere Inhalt von d_a nach d_b in umgekehrter Reihenfolge
    reverseCopyKernel<<<blocks, threads>>>(d_a_rev, d_b_rev, num_elements_rev);
    cudaDeviceSynchronize();
    printf("Reverse copy kernel execution completed\n\n");

    // 5. Kopiere d_b zurück nach h_a
    cudaMemcpy(h_rev, d_b_rev, num_bytes_rev, cudaMemcpyDeviceToHost);
    printf("Copied data from device d_b_rev back to host\n\n");

    // Verify
    printf("First 10 values (should be reversed): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_rev[i]);
    }
    printf("\n\n");

    cudaFree(d_a_rev);
    cudaFree(d_b_rev);
    free(h_rev);
    printf("Freed device and host memory for reverse copy demo\n\n");
    printf("Aufgabe 3 completed successfully!\n\n");

    //=====Aufgabe 4: Speicherbereich umgekehrt Kopieren, N Blöcke=====//
    std::cout << "CUDA Reverse Copy Demo (N Blocks)" << std::endl;
    std::cout << "==============================\n"
              << std::endl;

    // 1. Allokiere Speicher h_a (host) und d_a, d_b (device) mit jeweils 1024 Elementen
    int *h_rev_n = (int *)malloc(num_bytes);
    int *d_a_rev_n, *d_b_rev_n;
    cudaMalloc((void **)&d_a_rev_n, num_bytes);
    cudaMalloc((void **)&d_b_rev_n, num_bytes);

    // Initialize host array
    for (int i = 0; i < N; i++)
    {
        h_rev_n[i] = i;
    }
    printf("Allocated host and device memory for reverse copy N blocks demo\n\n");

    // 2. Definiere Blockgröße (z.B. 256 Threads pro Block)
    int threadsPerBlockN = 256;
    int numBlocksN = (N + threadsPerBlockN - 1) / threadsPerBlockN;
    printf("Using %d blocks of %d threads for reverse copy N blocks kernel\n", numBlocksN, threadsPerBlockN);

    // 3. Kopiere Speicher von h_a nach d_a
    cudaMemcpy(d_a_rev_n, h_rev_n, num_bytes, cudaMemcpyHostToDevice);
    printf("Copied data from host to device d_a_rev_n\n\n");

    // 4. Kernel-Funktion: Kopiere Inhalt von d_a nach d_b in umgekehrter Reihenfolge
    reverseCopyKernelN<<<numBlocksN, threadsPerBlockN>>>(d_a_rev_n, d_b_rev_n, N);
    cudaDeviceSynchronize();
    printf("Reverse copy N blocks kernel execution completed\n\n");

    // 5. Kopiere d_b zurück nach h_a
    cudaMemcpy(h_rev_n, d_b_rev_n, num_bytes, cudaMemcpyDeviceToHost);
    printf("Copied data from device d_b_rev_n back to host\n\n");

    // Verify
    printf("First 10 values (should be reversed): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_rev_n[i]);
    }
    printf("\n");
    printf("Last 10 values: ");
    for (int i = N - 10; i < N; i++)
    {
        printf("%d ", h_rev_n[i]);
    }
    printf("\n\n");

    cudaFree(d_a_rev_n);
    cudaFree(d_b_rev_n);
    free(h_rev_n);
    printf("Freed device and host memory for reverse copy N blocks demo\n\n");
    printf("Aufgabe 4 completed successfully!\n\n");

    //=====Aufgabe 5: Speicherbereich umgekehrt Kopieren mit Shared Memory=====//
    std::cout << "CUDA Reverse Copy with Shared Memory Demo" << std::endl;
    std::cout << "=========================================\n"
              << std::endl;

    // 1. Allokiere Speicher h_a (host) und d_a, d_b (device) mit jeweils 1024 Elementen
    int *h_rev_shared = (int *)malloc(num_bytes);
    int *d_a_rev_shared, *d_b_rev_shared;
    cudaMalloc((void **)&d_a_rev_shared, num_bytes);
    cudaMalloc((void **)&d_b_rev_shared, num_bytes);

    // Initialize host array
    for (int i = 0; i < N; i++)
    {
        h_rev_shared[i] = i;
    }
    printf("Allocated host and device memory for reverse copy with shared memory demo\n\n");

    // 2. Definiere Blockgröße (256 Threads pro Block for shared memory)
    int threadsPerBlockShared = 256;
    int numBlocksShared = (N + threadsPerBlockShared - 1) / threadsPerBlockShared;
    printf("Using %d blocks of %d threads for reverse copy with shared memory kernel\n", numBlocksShared, threadsPerBlockShared);

    // 3. Kopiere Speicher von h_a nach d_a
    cudaMemcpy(d_a_rev_shared, h_rev_shared, num_bytes, cudaMemcpyHostToDevice);
    printf("Copied data from host to device d_a_rev_shared\n\n");

    // 4. Kernel-Funktion: Kopiere Inhalt von d_a nach d_b in umgekehrter Reihenfolge mit Shared Memory
    reverseCopySharedKernel<<<numBlocksShared, threadsPerBlockShared>>>(d_a_rev_shared, d_b_rev_shared, N);
    cudaDeviceSynchronize();
    printf("Reverse copy with shared memory kernel execution completed\n\n");

    // 5. Kopiere d_b zurück nach h_a
    cudaMemcpy(h_rev_shared, d_b_rev_shared, num_bytes, cudaMemcpyDeviceToHost);
    printf("Copied data from device d_b_rev_shared back to host\n\n");

    // Verify
    printf("First 10 values (should be reversed): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_rev_shared[i]);
    }
    printf("\n");
    printf("Last 10 values: ");
    for (int i = N - 10; i < N; i++)
    {
        printf("%d ", h_rev_shared[i]);
    }
    printf("\n\n");

    cudaFree(d_a_rev_shared);
    cudaFree(d_b_rev_shared);
    free(h_rev_shared);
    printf("Freed device and host memory for reverse copy with shared memory demo\n\n");
    printf("Aufgabe 5 completed successfully!\n\n");

    printf("========================================\n");
    printf("All CUDA exercises completed successfully!\n");
    printf("========================================\n");

    return 0;
}
