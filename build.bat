@echo off
REM ============================================================
REM  build.bat  —  One-click build for CUDA Kernel Library
REM  Uses MSYS2 MinGW-w64 GCC + CUDA nvcc (no MSVC needed)
REM ============================================================

echo.
echo ============================================================
echo   CUDA Kernel Library  —  Build Script
echo ============================================================

REM ── 1. Verify MSYS2 GCC ────────────────────────────────────
echo.
echo [1/5] Checking MSYS2 MinGW-w64 GCC ...
if not exist "C:\msys64\mingw64\bin\g++.exe" (
    echo [ERROR] MSYS2 MinGW-w64 GCC not found at C:\msys64\mingw64\bin\
    echo.
    echo Please install it by running this in PowerShell:
    echo   C:\msys64\usr\bin\pacman.exe --noconfirm -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
    pause
    exit /b 1
)
"C:\msys64\mingw64\bin\g++.exe" --version | findstr "g++"
echo        GCC OK

REM ── 2. Verify CUDA nvcc ─────────────────────────────────────
echo.
echo [2/5] Checking CUDA nvcc ...
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin

REM Auto-find CUDA if not at default path
if not exist "%CUDA_BIN%\nvcc.exe" (
    for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        set CUDA_BIN=%%D\bin
    )
)
if not exist "%CUDA_BIN%\nvcc.exe" (
    echo [ERROR] nvcc.exe not found! Install CUDA Toolkit from:
    echo   https://developer.nvidia.com/cuda-downloads
    echo Or: winget install -e --id Nvidia.CUDA
    pause
    exit /b 1
)
"%CUDA_BIN%\nvcc.exe" --version | findstr "release"
echo        nvcc OK

REM ── 3. Verify CMake ──────────────────────────────────────────
echo.
echo [3/5] Checking CMake ...
set CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe
if not exist "%CMAKE_EXE%" (
    set CMAKE_EXE=C:\msys64\mingw64\bin\cmake.exe
)
if not exist "%CMAKE_EXE%" (
    echo [ERROR] cmake.exe not found!
    echo Run: winget install -e --id Kitware.CMake
    pause
    exit /b 1
)
"%CMAKE_EXE%" --version | findstr "cmake version"
echo        CMake OK

REM ── 4. Configure with CMake ──────────────────────────────────
echo.
echo [4/5] Configuring build (MinGW Makefiles + CUDA arch 89) ...

REM Delete stale build cache if it exists (avoids old MSVC config)
if exist build\CMakeCache.txt (
    echo        Removing stale CMake cache...
    del /q build\CMakeCache.txt
    rd /s /q build\CMakeFiles 2>nul
)
if not exist build mkdir build

"%CMAKE_EXE%" -S . -B build ^
    -G "MinGW Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -DCMAKE_C_COMPILER="C:/msys64/mingw64/bin/gcc.exe" ^
    -DCMAKE_CXX_COMPILER="C:/msys64/mingw64/bin/g++.exe" ^
    -DCMAKE_CUDA_HOST_COMPILER="C:/msys64/mingw64/bin/g++.exe" ^
    -DCMAKE_CUDA_COMPILER="%CUDA_BIN:\=/%/nvcc.exe" ^
    -DCMAKE_MAKE_PROGRAM="C:/msys64/mingw64/bin/mingw32-make.exe"

if errorlevel 1 (
    echo.
    echo [ERROR] CMake configuration FAILED. Check errors above.
    pause
    exit /b 1
)

REM ── 5. Build ──────────────────────────────────────────────────
echo.
echo [5/5] Building all targets (this may take 2-5 minutes) ...
"C:\msys64\mingw64\bin\mingw32-make.exe" -C build -j 4

if errorlevel 1 (
    echo.
    echo [ERROR] Build FAILED. Check errors above.
    pause
    exit /b 1
)

REM ── Done ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo.
echo  Next steps — run from the 'build' folder:
echo.
echo   Correctness Tests:
echo     build\test_matmul.exe
echo     build\test_layernorm.exe
echo     build\test_softmax.exe
echo.
echo   Full Benchmark Suite:
echo     build\cuda_kernel_library.exe
echo.
echo   Standalone Benchmarks (with custom shapes):
echo     build\benchmark_matmul.exe   1024 1024 1024
echo     build\benchmark_layernorm.exe  128 4096
echo     build\benchmark_softmax.exe    128 4096
echo.
echo   Analyze Results:
echo     python scripts\analyze_results.py
echo ============================================================
echo.
pause
