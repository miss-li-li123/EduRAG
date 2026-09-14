@echo off
chcp 65001 >nul
cd /d %~dp0

rem 模型全部在本地，禁止运行时访问 HuggingFace
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo ============================================
echo  EduRAG Web 服务重启脚本
echo ============================================

echo.
echo [1/2] 停止占用 8001 端口的旧进程...
set "KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr LISTENING') do (
    echo   - 终止旧进程 PID: %%a
    taskkill /F /PID %%a >nul 2>&1
    set "KILLED=1"
)
if "%KILLED%"=="0" echo   - 未发现运行中的旧进程

rem 等待端口释放
:waitport
timeout /t 2 /nobreak >nul
netstat -ano | findstr ":8001" | findstr LISTENING >nul 2>&1
if %errorlevel%==0 goto waitport

echo.
echo [2/2] 启动服务（首次加载模型约需 10-30 秒）...
echo   访问地址: http://localhost:8001
echo   关闭服务: 直接关闭本窗口 或 按 Ctrl+C
echo ============================================
echo.

.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001

pause
