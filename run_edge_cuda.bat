@echo off
SETLOCAL EnableDelayedExpansion
SET "BASEDIR=%~dp0"
CD /D "%BASEDIR%"

where python >nul || (
  echo Python 3.10+ fehlt – zuerst installieren!
  pause & exit /b 1
)

IF NOT EXIST venv (
  python -m venv venv
)
CALL venv\Scripts\activate.bat

REM ---------- 1) PIP + NumPy 1.26 erzwingen -------------------------------
python -m pip install --upgrade pip
pip install "numpy<2" --force-reinstall

REM ---------- 2) GPU-PyTorch cu118 ----------------------------------------
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 ^
  torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118 ^
  --extra-index-url https://pypi.org/simple

REM ---------- 3) OpenMMLab (passendes Wheel) ------------------------------
pip install openmim
mim install "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2.0/index.html
pip install mmengine==0.10.7 mmsegmentation==1.0.0

REM ---------- 4) Restliche Abhängigkeiten ---------------------------------
pip install -r requirements_edge.txt

python edge_batch_runner.py
pause
