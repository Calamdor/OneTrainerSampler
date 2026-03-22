@echo off
setlocal

REM OneTrainer Standalone Sampler (unified — model selected at runtime)
REM Assumes OneTrainer is adjacent (../OneTrainer/) with its venv

set SCRIPT_DIR=%~dp0
set OT_DIR=%SCRIPT_DIR%..\OneTrainer
set PYTHON=%OT_DIR%\venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo ERROR: Could not find venv at %PYTHON%
    echo Make sure OneTrainer is adjacent to this folder and its venv is set up.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Set up MSVC environment for torch.compile (provides omp.h and cl.exe).
REM vswhere.exe is always at this path on 64-bit Windows regardless of VS
REM version or edition.
REM ---------------------------------------------------------------------------
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo [vcvars] vswhere.exe not found -- skipping MSVC setup.
    goto :launch
)

set VS_DIR=
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_DIR=%%i"

if not defined VS_DIR (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VS_DIR=%%i"
)

if not defined VS_DIR (
    echo [vcvars] No Visual Studio or Build Tools found -- skipping MSVC setup.
    goto :launch
)

set "VCVARS=%VS_DIR%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [vcvars] vcvars64.bat not found -- skipping MSVC setup.
    goto :launch
)

echo [vcvars] Initialising MSVC from %VCVARS%
call "%VCVARS%"
echo [vcvars] Done.

:launch

REM ---------------------------------------------------------------------------
REM Pin the TorchInductor / Triton kernel cache to a stable directory so that
REM compiled kernels persist across sessions.
REM ---------------------------------------------------------------------------
if not defined TORCHINDUCTOR_CACHE_DIR (
    set "TORCHINDUCTOR_CACHE_DIR=%USERPROFILE%\.cache\torchinductor"
)
if not defined TRITON_CACHE_DIR (
    set "TRITON_CACHE_DIR=%USERPROFILE%\.cache\triton"
)

"%PYTHON%" "%SCRIPT_DIR%onetrainer_sampler_gui.py"
if errorlevel 1 pause
