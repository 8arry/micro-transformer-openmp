# Micro Transformer OpenMP Build Script
# Enhanced build system for C++23 + OpenMP project

param(
    [string]$Target = "build",
    [string]$Config = "Release",
    [int]$ThreadCount = 0
)

# Configuration
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

# GCC Configuration for WinLibs
$GCC_PATH = "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
$GCC_COMPILER = "$GCC_PATH\g++.exe"
$CMAKE_GENERATOR = "MinGW Makefiles"

# Directories
$PROJECT_ROOT = $PSScriptRoot
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
$SRC_DIR = Join-Path $PROJECT_ROOT "src"
$INCLUDE_DIR = Join-Path $PROJECT_ROOT "include"
$BIN_DIR = Join-Path $BUILD_DIR "bin"

function Write-Header {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "           Micro Transformer OpenMP Build System               " -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Target: $Target" -ForegroundColor Yellow
    Write-Host "Config: $Config" -ForegroundColor Yellow
    Write-Host "Compiler: $GCC_COMPILER" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Test-Environment {
    Write-Host "`nTesting build environment..." -ForegroundColor Blue
    
    # Test GCC compiler
    if (-not (Test-Path $GCC_COMPILER)) {
        throw "GCC compiler not found at: $GCC_COMPILER"
    }
    
    # Test C++23 support
    Write-Host "  Testing C++23 support..." -ForegroundColor Gray
    $cpp23_test = echo 'int main(){return 0;}' | & $GCC_COMPILER -std=c++23 -x c++ - -o nul 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "C++23 not supported: $cpp23_test"
    }
    
    # Test OpenMP support
    Write-Host "  Testing OpenMP support..." -ForegroundColor Gray
    $omp_test = echo '#include <omp.h>' + "`n" + 'int main(){return omp_get_max_threads();}' | & $GCC_COMPILER -std=c++23 -fopenmp -x c++ - -o nul 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "OpenMP not supported: $omp_test"
    }
    
    Write-Host "  Environment verification completed!" -ForegroundColor Green
}

function Build-Project {
    Write-Host "`nBuilding project..." -ForegroundColor Blue
    
    # Create and enter build directory
    if (Test-Path $BUILD_DIR) {
        Remove-Item $BUILD_DIR -Recurse -Force
    }
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    Push-Location $BUILD_DIR
    
    try {
        # Configure with CMake
        Write-Host "  Configuring with CMake..." -ForegroundColor Gray
        $env:PATH = "$GCC_PATH;$env:PATH"
        
        $cmake_cmd = @(
            "cmake"
            "-G", $CMAKE_GENERATOR
            "-DCMAKE_BUILD_TYPE=$Config"
            "-DCMAKE_CXX_COMPILER=$GCC_COMPILER"
            ".."
        )
        
        & $cmake_cmd[0] $cmake_cmd[1..($cmake_cmd.Length-1)]
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build
        Write-Host "  Building executable..." -ForegroundColor Gray
        cmake --build . --config $Config --parallel
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        Write-Host "  Build completed successfully!" -ForegroundColor Green
        
        # Show build output
        $exe_path = Join-Path $BIN_DIR "MicroTransformerOpenMP.exe"
        if (Test-Path $exe_path) {
            $file_info = Get-Item $exe_path
            Write-Host "  Executable: $exe_path" -ForegroundColor Yellow
            Write-Host "  Size: $([math]::Round($file_info.Length / 1024, 2)) KB" -ForegroundColor Yellow
            Write-Host "  Modified: $($file_info.LastWriteTime)" -ForegroundColor Yellow
        }
        
    } finally {
        Pop-Location
    }
}

function Test-Basic {
    Write-Host "`nRunning basic functionality test..." -ForegroundColor Blue
    
    $exe_path = Join-Path $BIN_DIR "MicroTransformerOpenMP.exe"
    if (-not (Test-Path $exe_path)) {
        throw "Executable not found. Build first."
    }
    
    # Set thread count if specified
    if ($ThreadCount -gt 0) {
        $env:OMP_NUM_THREADS = $ThreadCount
        Write-Host "  Using $ThreadCount OpenMP threads" -ForegroundColor Yellow
    }
    
    Write-Host "  Executing transformer test..." -ForegroundColor Gray
    & $exe_path
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Basic test completed successfully!" -ForegroundColor Green
    } else {
        throw "Test execution failed with exit code: $LASTEXITCODE"
    }
}

function Show-Help {
    Write-Host @"
Micro Transformer OpenMP Build System

Usage: .\build.ps1 [Target] [Config] [ThreadCount]

Targets:
  build       - Build the project (default)
  test        - Build and run basic test
  benchmark   - Build and run performance benchmark
  clean       - Clean build artifacts
  help        - Show this help message

Config:
  Release     - Optimized build (default)
  Debug       - Debug build with symbols

ThreadCount:
  0           - Use default thread count (default)
  1-16        - Force specific thread count for testing

Examples:
  .\build.ps1
  .\build.ps1 test Release 4
  .\build.ps1 benchmark
  .\build.ps1 clean

"@ -ForegroundColor Cyan
}

function Clean-Build {
    Write-Host "`nCleaning build artifacts..." -ForegroundColor Blue
    
    if (Test-Path $BUILD_DIR) {
        Remove-Item $BUILD_DIR -Recurse -Force
        Write-Host "  Build directory cleaned." -ForegroundColor Green
    }
    
    # Clean any temporary files
    Get-ChildItem -Path $PROJECT_ROOT -Filter "*.tmp" -Recurse | Remove-Item -Force
    Get-ChildItem -Path $PROJECT_ROOT -Filter "*.log" -Recurse | Remove-Item -Force
    Get-ChildItem -Path $PROJECT_ROOT -Filter "benchmark_results_*.csv" | Remove-Item -Force
    
    Write-Host "  Cleanup completed!" -ForegroundColor Green
}

# Main execution
try {
    Write-Header
    
    switch ($Target.ToLower()) {
        "build" {
            Test-Environment
            Build-Project
        }
        "test" {
            Test-Environment
            Build-Project
            Test-Basic
        }
        "benchmark" {
            Test-Environment
            Build-Project
            Test-Basic
        }
        "clean" {
            Clean-Build
        }
        "help" {
            Show-Help
        }
        default {
            Write-Warning "Unknown target: $Target"
            Show-Help
            exit 1
        }
    }
    
    Write-Host "`n================================================================" -ForegroundColor Cyan
    Write-Host "           Build process completed successfully!                " -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Cyan
    
} catch {
    Write-Host "`n================================================================" -ForegroundColor Red
    Write-Host "           Build process failed!                               " -ForegroundColor Red
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
