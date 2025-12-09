# CUDA Programming Exercises

CUDA exercises demonstrating memory operations, kernel functions, and parallel computing on GPU.

## Project Structure

- `main.cu` - Main CUDA program with 5 exercises
- `CMakeLists.txt` - CMake build configuration
- `run_gpu.sh` - SLURM job script for HPC execution

## Exercises

1. **Memory Operations** - Copy operations between host and device
2. **Kernel Functions** - Basic GPU kernel execution
3. **Reverse Copy (1 Block)** - Array reversal with single block
4. **Reverse Copy (N Blocks)** - Array reversal with multiple blocks
5. **Shared Memory** - Optimized reversal using shared memory

## Building

```bash
cmake -B build -S .
cmake --build build
```

## Running on HPC

```bash
sbatch run_gpu.sh
```

## Requirements

- CUDA Toolkit 11.8+
- CMake 3.20+
- GPU with compute capability 7.5+

