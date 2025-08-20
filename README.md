# Micro Transformer with OpenMP

A high-performance C++23 implementation of a simplified Transformer encoder with OpenMP parallelization. Achieves **10.51x speedup** on 16 cores with perfect numerical accuracy.

## Performance Results

### Speedup Analysis (based on seq_length=256)

| Threads | Time (ms) | Speedup | Efficiency |
| ------- | --------- | ------- | ---------- |
| 1       | 1832.3    | 1.00x   | 100.0%     |
| 2       | 651.2     | 2.81x   | 140.6%     |
| 4       | 336.5     | 5.44x   | 136.1%     |
| 8       | 215.7     | 8.49x   | 106.1%     |
| 16      | 174.3     | 10.51x  | 65.7%      |

### Scalability Across Problem Sizes

| Seq Length | 1 Thread  | 16 Threads | Speedup |
| ---------- | --------- | ---------- | ------- |
| 64         | 413.7 ms  | 56.2 ms    | 7.36x   |
| 128        | 853.9 ms  | 89.5 ms    | 9.54x   |
| 256        | 1832.3 ms | 174.3 ms   | 10.51x  |


## Requirements

- **Compiler**: GCC 15.1.0+ with C++23 support
- **OpenMP**: Version 4.5+ for parallel programming
- **Windows**: MinGW-w64 toolchain

## Quick Start

### Method 1: Direct Compilation
```powershell
g++ -std=c++23 -fopenmp -O3 -march=native -I./include src/*.cpp -o MicroTransformer.exe
.\MicroTransformer.exe
```

### Method 2: CMake (Recommended)
```powershell
mkdir build && cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
mingw32-make -j
.\bin\MicroTransformerOpenMP.exe
```

## Features

- **Superlinear Speedup**: Achieves 2.81x speedup on 2 cores (140.6% efficiency)  
- **Multi-head self-attention** with parallel Q/K/V computation via sections
- **Feed-forward networks** with blocked matrix multiplication (64×64 blocks)
- **Layer normalization** with SIMD reductions for mean/variance computation
- **Smart parallelism control**: Conditional parallelization to avoid nested overhead
- **Cache-optimized design**: Block size tuned for L1 cache performance

## Project Structure

```
src/
├── matrix.cpp          # Matrix operations with blocked multiplication
├── attention.cpp       # Multi-head attention with parallel Q/K/V
├── layers.cpp          # Feed-forward and layer normalization  
├── encoder.cpp         # Transformer encoder layers
├── benchmark.cpp       # Performance measurement suite
└── main.cpp            # Main program and benchmark runner

include/                # Header files
CMakeLists.txt         # Build configuration
```

## Sample Output

```
================================================================
           Micro Transformer with OpenMP Parallelization
================================================================
C++ Standard: 202302
OpenMP version: 201511
Max threads available: 16

Testing sequence length: 256
  Serial: 1832.342 ms
  2 threads: 651.156 ms (speedup: 2.81x, correctness: PASS)
  4 threads: 336.488 ms (speedup: 5.44x, correctness: PASS)
  8 threads: 215.736 ms (speedup: 8.49x, correctness: PASS)
  16 threads: 174.317 ms (speedup: 10.51x, correctness: PASS)

Results saved to: benchmark_results_1755423478.csv
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
