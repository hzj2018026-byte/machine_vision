@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   矿石图像分类 - 一键配置 Python 环境
echo ============================================
echo.

echo [1/4] 检查 Python ...
python --version >nul 2>&1
if errorlevel 1 goto :no_python
python --version
echo.

echo [2/4] 升级 pip ...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
echo.

echo [3/4] 安装 PyTorch ...
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_cpu
echo 检测到 NVIDIA 显卡，安装 CUDA 版（约 2.5 GB，请耐心等待）...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto :torch_done
echo CUDA 版安装失败，自动改用 CPU 版 ...
:torch_cpu
echo 安装 CPU 版 PyTorch（约 200 MB）...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto :torch_done
echo 官方源失败，尝试清华镜像 ...
python -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto :fail
:torch_done
echo.

echo [4/4] 安装其余依赖 ...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.

echo ===== 环境检查 =====
python -c "import torch, torchvision, cv2, PIL, numpy, matplotlib, requests, tqdm, tkinter; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__); print('tkinter: OK'); print('ALL DEPENDENCIES OK')"
if errorlevel 1 goto :check_fail
echo.
echo 配置完成！接下来运行：python 2.split_data.py
pause
exit /b 0

:no_python
echo.
echo [错误] 没有找到 python 命令。
echo 请先安装 Python 3.10 ~ 3.12（安装时务必勾选 "Add python.exe to PATH"）：
echo   https://www.python.org/downloads/
pause
exit /b 1

:check_fail
echo.
echo [警告] 依赖已安装，但导入检查失败，请把上面的报错发给我。
pause
exit /b 1

:fail
echo.
echo [错误] 安装失败，请把上面的报错信息发给我。
pause
exit /b 1
