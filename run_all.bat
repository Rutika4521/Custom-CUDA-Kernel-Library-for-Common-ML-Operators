@echo off
setlocal
cd /d "%~dp0"
call build.bat
if errorlevel 1 exit /b 1
ctest --test-dir build --output-on-failure
if errorlevel 1 exit /b 1
build\benchmark_targets.exe --csv results\target_results.csv
if errorlevel 1 exit /b 1
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv --plot
exit /b %errorlevel%
