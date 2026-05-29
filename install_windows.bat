@echo off
setlocal enabledelayedexpansion

REM ===============================================================
REM  Chatterbox TTS - Windows installation script (v2)
REM  Prérequis : conda activate chatterbox  (Python 3.11)
REM  Usage : install_windows.bat [cpu|cu124|cu126]   (défaut: cu124)
REM ===============================================================

set PYTHONNOUSERSITE=1
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=cu124"

echo === Vérification de l'environnement Python actif ===
python -c "import sys; print('Python exe :', sys.executable)"
python -c "import sys; assert 'chatterbox' in sys.executable.lower(), 'ERREUR : l''env chatterbox n''est pas actif !'"
if errorlevel 1 (
    echo.
    echo !!! L'env conda 'chatterbox' n'est pas actif. Lance 'conda activate chatterbox' d'abord.
    exit /b 1
)

echo.
echo === Variante PyTorch sélectionnée : %VARIANT% ===

echo.
echo === Mise à jour de pip / setuptools / wheel ===
python -m pip install --upgrade pip setuptools wheel || goto :error

echo.
echo === Installation de PyTorch (%VARIANT%) ===
if /I "%VARIANT%"=="cpu" (
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu || goto :error
) else if /I "%VARIANT%"=="cu124" (
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || goto :error
) else if /I "%VARIANT%"=="cu126" (
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 || goto :error
)

echo.
echo === Installation des dépendances Chatterbox ===
python -m pip install transformers==4.46.3 diffusers==0.29.0 ^
  conformer==0.3.2 resemble-perth==1.0.1 safetensors==0.5.3 ^
  librosa==0.11.0 pykakasi==2.3.0 || goto :error

echo.
echo === Installation de s3tokenizer ===
python -m pip install onnx==1.16.2 || goto :error
python -m pip install --no-deps s3tokenizer==0.2.0 || goto :error

echo.
echo === Installation de chatterbox-tts (sans pkuseg) ===
python -m pip install --no-deps chatterbox-tts || goto :error

echo.
echo === Installation de pyloudnorm + omegaconf ===
python -m pip install pyloudnorm omegaconf || goto :error

echo.
echo === Installation des dépendances FastAPI ===
python -m pip install fastapi==0.110.1 uvicorn==0.29.0 python-multipart==0.0.9 || goto :error

echo.
echo === Vérification finale ===
python -c "import torch, fastapi, uvicorn, chatterbox; print('OK - torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" || goto :error

echo.
echo ============================================================
echo   Installation terminée avec succès !
echo   Lance le serveur avec :
echo     python -m uvicorn server:app --host 0.0.0.0 --port 3200
echo ============================================================
goto :eof

:error
echo.
echo !!! ERREUR pendant l'installation - voir les messages ci-dessus.
exit /b 1