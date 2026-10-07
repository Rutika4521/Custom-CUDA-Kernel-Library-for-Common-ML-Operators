@echo off
setlocal
cd /d "%~dp0"
where cl >nul 2>&1
if errorlevel 1 call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Open an x64 Native Tools Command Prompt or install VS 2022 C++ Build Tools.
    exit /b 1
)
set "TASK_ARCH=120-real"
if not "%~1"=="" set "TASK_ARCH=%~1"
set "TASK_FRESH="
set "TASK_CACHE="
if exist build\CMakeCache.txt for /f "tokens=1,* delims==" %%A in ('findstr /B "CMAKE_CACHEFILE_DIR:INTERNAL=" build\CMakeCache.txt') do set "TASK_CACHE=%%B"
if defined TASK_CACHE set "TASK_CACHE=%TASK_CACHE:/=\%"
if defined TASK_CACHE if /I not "%TASK_CACHE%"=="%CD%\build" set "TASK_FRESH=--fresh"
cmake %TASK_FRESH% -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=%TASK_ARCH%
if errorlevel 1 exit /b 1
cmake --build build
exit /b %errorlevel%
