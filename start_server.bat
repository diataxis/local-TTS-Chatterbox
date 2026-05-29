@echo off
set "CONDA_PATH=%USERPROFILE%\miniconda3"
set "CONDA_ENV=chatterbox"

cd /d "%~dp0"

echo [INFO] Conda's Path : %CONDA_PATH%
echo [INFO] Conda's env : %CONDA_ENV%

if exist "%CONDA_PATH%\shell\condabin\conda-hook.ps1" (
    REM Initialize conda and run all commands automatically
    %WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& '%CONDA_PATH%\shell\condabin\conda-hook.ps1' ; conda activate '%CONDA_PATH%' ; conda activate %CONDA_ENV% ; python -m uvicorn server:app --host 0.0.0.0 --port 3200"
) else (
    echo [ERROR] Conda has not been found in %CONDA_PATH%
    echo [ERROR] Please install Miniconda or change the CONDA_PATH.
)
pause