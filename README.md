# Micro Transformer with OpenMP

A high-performance C++23 implementation of a simplified Transformer encoder with OpenMP parallelization. Achieves **10.51x speedup** on 16 cores with perfect numerical accuracy.

## Performance Results

| Threads | Time (ms) | Speedup | Efficiency |
| ------- | --------- | ------- | ---------- |
| 1       | 1715.3    | 1.00x   | 100.0%     |
| 2       | 625.4     | 2.74x   | 137.1%     |
| 4       | 285.2     | 6.01x   | 150.3%     |
| 8       | 194.8     | 8.81x   | 110.1%     |
| 16      | 163.2     | 10.51x  | 65.7%      |

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

- Multi-head self-attention with parallel Q/K/V computation
- Feed-forward networks with vectorized operations  
- Layer normalization with SIMD reductions
- Blocked matrix multiplication (64×64 cache-friendly blocks)
- Nested parallelism control and load balancing

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
  Serial: 1715.319 ms
  2 threads: 625.411 ms (speedup: 2.74x, correctness: PASS)
  4 threads: 285.234 ms (speedup: 6.01x, correctness: PASS)
  8 threads: 194.847 ms (speedup: 8.81x, correctness: PASS)
  16 threads: 163.197 ms (speedup: 10.51x, correctness: PASS)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
