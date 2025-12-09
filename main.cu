#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cstring>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define N 1024
#define RADIX_SIZE 65536  // Size for radix sort test

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

// ===== Custom Parallel RadixSort GPU Kernels =====

// Kernel 1: Count digit occurrences using shared memory
__global__ void radixCountKernel(unsigned int *input, unsigned int *globalCounts, int n, int bit)
{
    __shared__ unsigned int localCounts[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < 256)
    {
        localCounts[tid] = 0;
    }
    __syncthreads();
    
    // Count in shared memory (faster than global atomics)
    if (idx < n)
    {
        unsigned int digit = (input[idx] >> bit) & 0xFF;
        atomicAdd(&localCounts[digit], 1);
    }
    __syncthreads();
    
    // Write block results to global memory
    if (tid < 256)
    {
        globalCounts[blockIdx.x * 256 + tid] = localCounts[tid];
    }
}

// Kernel 2: Parallel prefix sum (scan) - Blelloch algorithm
__global__ void parallelPrefixSum(unsigned int *data, int n)
{
    extern __shared__ unsigned int temp[];
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < n)
    {
        temp[tid] = data[tid];
    }
    else
    {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < n; stride *= 2)
    {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n)
        {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear last element
    if (tid == 0)
    {
        temp[n - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = n / 2; stride > 0; stride /= 2)
    {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n)
        {
            unsigned int temp_val = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = temp_val;
        }
        __syncthreads();
    }
    
    // Write results back
    if (tid < n)
    {
        data[tid] = temp[tid];
    }
}

// Kernel 3: Local prefix sum within each block
__global__ void blockPrefixSum(unsigned int *blockSums, unsigned int *globalCounts, int numBlocks)
{
    int digit = blockIdx.x;
    
    if (digit < 256)
    {
        unsigned int sum = 0;
        for (int i = 0; i < numBlocks; i++)
        {
            unsigned int val = globalCounts[i * 256 + digit];
            globalCounts[i * 256 + digit] = sum;
            sum += val;
        }
        blockSums[digit] = sum;
    }
}

// Kernel 4: Scatter elements to their sorted positions
__global__ void radixScatterKernel(unsigned int *input, unsigned int *output, 
                                   unsigned int *prefixSum, int n, int bit)
{
    __shared__ unsigned int localPrefix[256];
    __shared__ unsigned int localElements[256];
    __shared__ unsigned int localDigits[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x;
    
    // Initialize shared memory for prefix sums
    if (tid < 256)
    {
        localPrefix[tid] = prefixSum[blockId * 256 + tid];
    }
    __syncthreads();
    
    // Load element and compute digit
    unsigned int element = 0;
    unsigned int digit = 0;
    if (idx < n)
    {
        element = input[idx];
        digit = (element >> bit) & 0xFF;
        localElements[tid] = element;
        localDigits[tid] = digit;
    }
    __syncthreads();
    
    // Scatter to output
    if (idx < n)
    {
        unsigned int position = atomicAdd(&localPrefix[digit], 1);
        output[position] = element;
    }
}

// Optimized version: Scatter with shared memory and proper ranking
__global__ void radixScatterOptimized(unsigned int *input, unsigned int *output,
                                      unsigned int *blockOffsets, unsigned int *digitCounts,
                                      int n, int bit, int numBlocks)
{
    __shared__ unsigned int sh_counts[256];
    __shared__ unsigned int sh_offsets[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < 256)
    {
        sh_counts[tid] = 0;
    }
    __syncthreads();
    
    // Load element, compute digit, and count
    unsigned int value = 0;
    unsigned int digit = 0;
    unsigned int rank = 0; // Rank of this element within its digit group in this block
    
    if (idx < n)
    {
        value = input[idx];
        digit = (value >> bit) & 0xFF;
        rank = atomicAdd(&sh_counts[digit], 1); // Get rank before incrementing
    }
    __syncthreads();
    
    // Compute prefix sum to get offsets (starting position for each digit within this block)
    if (tid < 256)
    {
        sh_offsets[tid] = 0;
        for (int d = 0; d < tid; d++)
        {
            sh_offsets[tid] += sh_counts[d];
        }
    }
    __syncthreads();
    
    // Scatter to output
    if (idx < n)
    {
        unsigned int blockBase = blockOffsets[blockIdx.x * 256 + digit]; // Global start for this block's portion of this digit
        unsigned int outputPos = blockBase + rank; // Final position = block base + rank within digit group
        
        // Debug: print first few
        if (idx < 5 && bit == 0)
        {
            printf("idx=%d, value=%u, digit=%u, rank=%u, blockBase=%u, outputPos=%u\n",
                   idx, value, digit, rank, blockBase, outputPos);
        }
        
        output[outputPos] = value;
    }
}

// CPU RadixSort for comparison
void radixSortCPU(unsigned int *arr, int n)
{
    unsigned int *output = (unsigned int *)malloc(n * sizeof(unsigned int));
    
    for (int bit = 0; bit < 32; bit += 8)
    {
        unsigned int counts[256] = {0};
        
        // Count occurrences
        for (int i = 0; i < n; i++)
        {
            unsigned int digit = (arr[i] >> bit) & 0xFF;
            counts[digit]++;
        }
        
        // Prefix sum
        for (int i = 1; i < 256; i++)
        {
            counts[i] += counts[i - 1];
        }
        
        // Scatter
        for (int i = n - 1; i >= 0; i--)
        {
            unsigned int digit = (arr[i] >> bit) & 0xFF;
            counts[digit]--;
            output[counts[digit]] = arr[i];
        }
        
        // Copy back
        for (int i = 0; i < n; i++)
        {
            arr[i] = output[i];
        }
    }
    
    free(output);
}

// Custom GPU RadixSort implementation
void radixSortGPU_Custom(unsigned int *d_input, unsigned int *d_output, int n)
{
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int *d_counts;
    unsigned int *d_blockOffsets;
    unsigned int *d_prefixSum;
    
    cudaMalloc(&d_counts, numBlocks * 256 * sizeof(unsigned int));
    cudaMalloc(&d_blockOffsets, numBlocks * 256 * sizeof(unsigned int));
    cudaMalloc(&d_prefixSum, 256 * sizeof(unsigned int));
    
    unsigned int *currentInput = d_input;
    unsigned int *currentOutput = d_output;
    
    // Process 8 bits at a time
    for (int bit = 0; bit < 32; bit += 8)
    {
        printf("Pass %d (bit %d)\n", bit/8, bit);
        
        // Step 1: Count occurrences
        cudaMemset(d_counts, 0, numBlocks * 256 * sizeof(unsigned int));
        radixCountKernel<<<numBlocks, threadsPerBlock>>>(currentInput, d_counts, n, bit);
        cudaDeviceSynchronize();
        
        // Step 2: Compute prefix sums (simplified approach)
        cudaMemset(d_prefixSum, 0, 256 * sizeof(unsigned int));
        
        // Compute block offsets and global prefix sum
        unsigned int h_counts[numBlocks * 256];
        cudaMemcpy(h_counts, d_counts, numBlocks * 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        unsigned int h_blockOffsets[numBlocks * 256];
        
        // First, compute global starting position for each digit (prefix sum across all digits)
        unsigned int digitStarts[256];
        digitStarts[0] = 0;
        for (int digit = 1; digit < 256; digit++)
        {
            digitStarts[digit] = digitStarts[digit - 1];
            for (int block = 0; block < numBlocks; block++)
            {
                digitStarts[digit] += h_counts[block * 256 + (digit - 1)];
            }
        }
        
        // Then, compute block-specific offsets for each digit
        for (int digit = 0; digit < 256; digit++)
        {
            unsigned int blockSum = 0;
            for (int block = 0; block < numBlocks; block++)
            {
                h_blockOffsets[block * 256 + digit] = digitStarts[digit] + blockSum;
                blockSum += h_counts[block * 256 + digit];
            }
        }
        
        cudaMemcpy(d_blockOffsets, h_blockOffsets, numBlocks * 256 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemset(d_counts, 0, numBlocks * 256 * sizeof(unsigned int));
        
        // Step 3: Scatter elements
        radixScatterOptimized<<<numBlocks, threadsPerBlock>>>(currentInput, currentOutput, 
                                                               d_blockOffsets, d_counts, n, bit, numBlocks);
        cudaDeviceSynchronize();
        
        // Swap buffers
        unsigned int *temp = currentInput;
        currentInput = currentOutput;
        currentOutput = temp;
        
        // Debug: check first few elements after swap
        if ((bit == 0 || bit == 8) && n <= 1024)
        {
            unsigned int h_check[10];
            cudaMemcpy(h_check, currentInput, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After pass %d, first 10 elements: ", bit/8);
            for (int i = 0; i < 10; i++)
            {
                printf("%u(d=%u) ", h_check[i], (h_check[i] >> bit) & 0xFF);
            }
            printf("\n");
        }
    }
    
    // Copy final result to d_output
    if (currentInput != d_output)
    {
        printf("Copying result from alternate buffer to d_output\n");
        cudaMemcpy(d_output, currentInput, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }
    else
    {
        printf("Result already in d_output\n");
    }
    
    cudaFree(d_counts);
    cudaFree(d_blockOffsets);
    cudaFree(d_prefixSum);
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

    //=====Aufgabe 6: RadixSort on GPU=====//
    std::cout << "\nRadixSort: CPU vs Custom GPU vs Thrust GPU" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    int sort_size = RADIX_SIZE;
    int sort_bytes = sort_size * sizeof(unsigned int);
    
    // Allocate host memory
    unsigned int *h_unsorted = (unsigned int *)malloc(sort_bytes);
    unsigned int *h_sorted_gpu_thrust = (unsigned int *)malloc(sort_bytes);
    unsigned int *h_sorted_gpu_custom = (unsigned int *)malloc(sort_bytes);
    unsigned int *h_sorted_cpu = (unsigned int *)malloc(sort_bytes);
    
    // Generate random data
    printf("Generating %d random numbers...\n", sort_size);
    srand(time(NULL));
    for (int i = 0; i < sort_size; i++)
    {
        h_unsorted[i] = rand() % 1000000;
    }
    
    // Copy for CPU version
    for (int i = 0; i < sort_size; i++)
    {
        h_sorted_cpu[i] = h_unsorted[i];
    }
    
    printf("First 10 unsorted values: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%u ", h_unsorted[i]);
    }
    printf("\n\n");
    
    // ===== CPU RadixSort =====
    printf("Running CPU RadixSort...\n");
    clock_t cpu_start = clock();
    radixSortCPU(h_sorted_cpu, sort_size);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %.3f ms\n\n", cpu_time);
    
    // ===== Custom GPU RadixSort =====
    printf("Running Custom GPU RadixSort...\n");
    unsigned int *d_input_custom, *d_output_custom;
    cudaMalloc(&d_input_custom, sort_bytes);
    cudaMalloc(&d_output_custom, sort_bytes);
    
    cudaEvent_t start_custom, stop_custom;
    cudaEventCreate(&start_custom);
    cudaEventCreate(&stop_custom);
    
    cudaEventRecord(start_custom);
    
    cudaMemcpy(d_input_custom, h_unsorted, sort_bytes, cudaMemcpyHostToDevice);
    radixSortGPU_Custom(d_input_custom, d_output_custom, sort_size);
    cudaMemcpy(h_sorted_gpu_custom, d_output_custom, sort_bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_custom);
    cudaEventSynchronize(stop_custom);
    
    float gpu_custom_time = 0;
    cudaEventElapsedTime(&gpu_custom_time, start_custom, stop_custom);
    printf("Custom GPU Time: %.3f ms\n\n", gpu_custom_time);
    
    // ===== Thrust GPU RadixSort =====
    printf("Running Thrust GPU RadixSort...\n");
    unsigned int *d_data_thrust;
    cudaMalloc(&d_data_thrust, sort_bytes);
    
    cudaEvent_t start_thrust, stop_thrust;
    cudaEventCreate(&start_thrust);
    cudaEventCreate(&stop_thrust);
    
    cudaEventRecord(start_thrust);
    
    cudaMemcpy(d_data_thrust, h_unsorted, sort_bytes, cudaMemcpyHostToDevice);
    thrust::device_ptr<unsigned int> dev_ptr(d_data_thrust);
    thrust::sort(dev_ptr, dev_ptr + sort_size);
    cudaMemcpy(h_sorted_gpu_thrust, d_data_thrust, sort_bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_thrust);
    cudaEventSynchronize(stop_thrust);
    
    float gpu_thrust_time = 0;
    cudaEventElapsedTime(&gpu_thrust_time, start_thrust, stop_thrust);
    printf("Thrust GPU Time: %.3f ms\n\n", gpu_thrust_time);
    
    // ===== Verification =====
    printf("Verifying results...\n");
    
    // Verify CPU
    bool cpu_correct = true;
    for (int i = 1; i < sort_size; i++)
    {
        if (h_sorted_cpu[i] < h_sorted_cpu[i - 1])
        {
            cpu_correct = false;
            break;
        }
    }
    printf("CPU RadixSort: %s\n", cpu_correct ? "✓ CORRECT" : "✗ FAILED");
    
    // Verify Custom GPU
    bool gpu_custom_correct = true;
    int first_error_idx = -1;
    for (int i = 1; i < sort_size; i++)
    {
        if (h_sorted_gpu_custom[i] < h_sorted_gpu_custom[i - 1])
        {
            if (first_error_idx == -1) first_error_idx = i;
            gpu_custom_correct = false;
            break;
        }
    }
    printf("Custom GPU RadixSort: %s\n", gpu_custom_correct ? "✓ CORRECT" : "✗ FAILED");
    if (!gpu_custom_correct)
    {
        printf("  First error at index %d: h[%d]=%u > h[%d]=%u\n", 
               first_error_idx, first_error_idx-1, h_sorted_gpu_custom[first_error_idx-1],
               first_error_idx, h_sorted_gpu_custom[first_error_idx]);
        
        // Check for duplicates or missing elements
        unsigned int *sorted_copy = (unsigned int *)malloc(sort_bytes);
        memcpy(sorted_copy, h_sorted_gpu_custom, sort_bytes);
        std::sort(sorted_copy, sorted_copy + sort_size);
        
        unsigned int *sorted_orig = (unsigned int *)malloc(sort_bytes);
        memcpy(sorted_orig, h_unsorted, sort_bytes);
        std::sort(sorted_orig, sorted_orig + sort_size);
        
        bool same_elements = true;
        for (int i = 0; i < sort_size; i++)
        {
            if (sorted_copy[i] != sorted_orig[i])
            {
                printf("  Element mismatch: original[%d]=%u, custom[%d]=%u\n", 
                       i, sorted_orig[i], i, sorted_copy[i]);
                same_elements = false;
                if (i > 10) break; // Only print first few
            }
        }
        if (same_elements)
        {
            printf("  All elements present, just wrong order\n");
        }
        
        free(sorted_copy);
        free(sorted_orig);
    }
    
    // Verify Thrust GPU
    bool gpu_thrust_correct = true;
    for (int i = 1; i < sort_size; i++)
    {
        if (h_sorted_gpu_thrust[i] < h_sorted_gpu_thrust[i - 1])
        {
            gpu_thrust_correct = false;
            break;
        }
    }
    printf("Thrust GPU RadixSort: %s\n\n", gpu_thrust_correct ? "✓ CORRECT" : "✗ FAILED");
    
    // ===== Display Results =====
    printf("First 10 sorted values (Custom GPU): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%u ", h_sorted_gpu_custom[i]);
    }
    printf("\n");
    
    printf("Last 10 sorted values (Custom GPU): ");
    for (int i = sort_size - 10; i < sort_size; i++)
    {
        printf("%u ", h_sorted_gpu_custom[i]);
    }
    printf("\n\n");
    
    printf("========================================\n");
    printf("Performance Comparison (%d elements):\n", sort_size);
    printf("========================================\n");
    printf("  CPU RadixSort:    %.3f ms\n", cpu_time);
    printf("  Custom GPU:       %.3f ms  (%.2fx speedup)\n", gpu_custom_time, cpu_time / gpu_custom_time);
    printf("  Thrust GPU:       %.3f ms  (%.2fx speedup)\n", gpu_thrust_time, cpu_time / gpu_thrust_time);
    printf("========================================\n");
    printf("  Custom vs Thrust: %.2fx %s\n", 
           gpu_custom_time / gpu_thrust_time,
           gpu_custom_time < gpu_thrust_time ? "faster" : "slower");
    printf("========================================\n\n");
    
    // Cleanup
    cudaFree(d_input_custom);
    cudaFree(d_output_custom);
    cudaFree(d_data_thrust);
    cudaEventDestroy(start_custom);
    cudaEventDestroy(stop_custom);
    cudaEventDestroy(start_thrust);
    cudaEventDestroy(stop_thrust);
    free(h_unsorted);
    free(h_sorted_gpu_thrust);
    free(h_sorted_gpu_custom);
    free(h_sorted_cpu);
    
    printf("Aufgabe 6 (RadixSort) completed successfully!\n\n");

    return 0;
}
