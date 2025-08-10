# Micro Transformer with OpenMP - C++23 Implementation
[2025ss] Parallel Computing
## Project Overview
A parallel micro Transformer model implementation based on C++23 standard, with OpenMP parallelization optimizations.

## Requirements
- **Compiler**: GCC 15.1.0+ (with C++23 support)
- **Standard**: C++23 (202302L)
- **Parallel Library**: OpenMP
- **Build System**: Make or CMake

## Verified C++23 Support
✅ Compiler Version: GCC 15.1.0
✅ C++23 Standard: __cplusplus = 202302L
✅ Compile Options: -std=c++23

## Build Configuration

### Compiler Path
```bash
# Newly installed GCC 15.1.0 path
GCC_PATH="C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
```

### Basic Compilation Commands
```bash
# Compile C++23 program
g++ -std=c++23 -fopenmp -O3 -o program source.cpp

# Debug version
g++ -std=c++23 -fopenmp -g -Wall -Wextra -o program_debug source.cpp
```

## C++23 Features to Use
- `std::expected` - Error handling
- `std::format` - String formatting
- `constexpr` enhancements - Compile-time computation
- `std::ranges` - Data processing
- Module system - Code organization

## OpenMP Parallelization Targets
- Matrix multiplication parallelization
- Attention mechanism parallel computation
- Feed-forward network parallel processing
- Batch processing parallelization

## Next Steps
1. Set up project structure
2. Implement basic Transformer components
3. Add OpenMP parallelization
4. Performance testing and optimization
