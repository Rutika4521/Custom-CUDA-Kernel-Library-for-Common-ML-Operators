@echo off
setlocal

REM ---- Activate VS 2022 Build Tools ----
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo ERROR: VS environment activation failed. & exit /b 1 )

echo.
echo [OK] Compilers:
cl 2>&1 | findstr "Version"
nvcc --version | findstr "release"
"C:\Program Files\CMake\bin\cmake.exe" --version | findstr "cmake version"

REM ---- Navigate to project root ----
cd /d "%~dp0"

REM ---- Clean and create build dir ----
echo.
echo [Step 1] Cleaning build...
if exist build\CMakeCache.txt del /q build\CMakeCache.txt
if exist build\CMakeFiles rd /s /q build\CMakeFiles
if not exist build mkdir build
cd build

REM ---- CMake Configure ----
REM  Use relative paths only (avoids space-in-path quoting issues).
REM  nvcc is already on PATH from vcvarsall, so no need to specify COMPILER.
echo.
echo [Step 2] CMake configure...
"C:\Program Files\CMake\bin\cmake.exe" .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
if errorlevel 1 ( echo ERROR: CMake configure failed. & cd .. & exit /b 1 )
echo [OK] Configure done.

REM ---- Build ----
echo.
echo [Step 3] Building (this takes 5-10 minutes on first run)...
nmake /NOLOGO
if errorlevel 1 ( echo ERROR: Build failed. & cd .. & exit /b 1 )
echo [OK] Build done.

REM ---- Tests ----
echo.
echo [Step 4] Correctness tests...
echo --- MatMul ---
test_matmul.exe
echo --- LayerNorm ---
test_layernorm.exe
echo --- Softmax ---
test_softmax.exe

REM ---- Full Benchmark ----
echo.
echo [Step 5] Full benchmark suite...
cuda_kernel_library.exe

REM ---- Standalone Benchmarks ----
echo.
echo [Step 6] Standalone benchmarks...
benchmark_matmul.exe 1024 1024 1024
benchmark_layernorm.exe 128 4096
benchmark_softmax.exe 128 4096

REM ---- Analyze ----
cd ..
echo.
echo [Step 7] Analyzing results...
python scripts\analyze_results.py 2>nul || echo (Python not found, open results\benchmark_results.csv manually)

echo.
echo =====================================================
echo  ALL DONE. See results\benchmark_results.csv
echo =====================================================
