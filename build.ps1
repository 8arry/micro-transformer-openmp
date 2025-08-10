# PowerShell build script - build.ps1
param(
    [string]$Target = "help"
)

# Compiler path
$GCC_DIR = "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
$GCC = "$GCC_DIR\g++.exe"

# Set PATH to include GCC libraries
$env:PATH = "$GCC_DIR;$env:PATH"

# Compile options
$CXXFLAGS = "-std=c++23", "-fopenmp", "-Wall", "-Wextra", "-O3"
$CXXFLAGS_DEBUG = "-std=c++23", "-fopenmp", "-Wall", "-Wextra", "-g", "-DDEBUG"

switch ($Target) {
    "test-cpp23" {
        Write-Host "Testing C++23 support..." -ForegroundColor Green
        echo "" | & $GCC -std=c++23 -dM -E -x c++ - | Select-String "__cplusplus"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] C++23 support confirmed!" -ForegroundColor Green
        }
    }
    
    "test-openmp" {
        Write-Host "Testing OpenMP support..." -ForegroundColor Green
        $testCode = @"
#include <omp.h>
#include <iostream>
int main() {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        #pragma omp critical
        std::cout << "Hello from thread " << tid << std::endl;
    }
    return 0;
}
"@
        
        $testCode | & $GCC -std=c++23 -fopenmp -o test_openmp_temp.exe -x c++ -
        if ($LASTEXITCODE -eq 0) {
            & ".\test_openmp_temp.exe"
            Remove-Item "test_openmp_temp.exe" -ErrorAction SilentlyContinue
            Write-Host "[OK] OpenMP support confirmed!" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] OpenMP test failed!" -ForegroundColor Red
        }
    }
    
    "build" {
        Write-Host "Building micro-transformer..." -ForegroundColor Green
        if (!(Test-Path "build")) { New-Item -ItemType Directory -Name "build" }
        
        $sources = Get-ChildItem "src\*.cpp" -Name
        if ($sources.Count -eq 0) {
            Write-Host "[ERROR] No source files found in src/" -ForegroundColor Red
            return
        }
        
        Write-Host "Found source files: $($sources -join ', ')" -ForegroundColor Cyan
        
        $sourceFiles = $sources | ForEach-Object { "src\$_" }
        $output = "build\micro_transformer.exe"
        
        Write-Host "Compiling with C++23 and OpenMP..." -ForegroundColor Yellow
        & $GCC $CXXFLAGS -o $output $sourceFiles
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Build successful!" -ForegroundColor Green
            Write-Host "Executable: $output" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Build failed!" -ForegroundColor Red
        }
    }
    
    "clean" {
        Write-Host "Cleaning build files..." -ForegroundColor Green
        Remove-Item "build" -Recurse -ErrorAction SilentlyContinue
        Remove-Item "*.exe" -ErrorAction SilentlyContinue
        Write-Host "[OK] Clean completed!" -ForegroundColor Green
    }
    
    "help" {
        Write-Host "Available targets:" -ForegroundColor Cyan
        Write-Host "  test-cpp23   - Test C++23 support"
        Write-Host "  test-openmp  - Test OpenMP support"
        Write-Host "  build        - Build the project"
        Write-Host "  run          - Build and run the project"
        Write-Host "  clean        - Clean build files"
        Write-Host "  help         - Show this help"
        Write-Host ""
        Write-Host "Usage: .\build.ps1 <target>" -ForegroundColor Yellow
    }
    
    "run" {
        Write-Host "Building and running micro-transformer..." -ForegroundColor Green
        & $PSCommandPath build
        if ($LASTEXITCODE -eq 0 -and (Test-Path "build\micro_transformer.exe")) {
            Write-Host "`nRunning the program..." -ForegroundColor Green
            Write-Host "=" * 50 -ForegroundColor Gray
            & ".\build\micro_transformer.exe"
            Write-Host "=" * 50 -ForegroundColor Gray
        } else {
            Write-Host "[ERROR] Cannot run - build failed or executable not found" -ForegroundColor Red
        }
    }
    
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        & $PSCommandPath help
    }
}
