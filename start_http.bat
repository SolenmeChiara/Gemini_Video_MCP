@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ============================================================================
REM Gemini-Video-MCP —— 以 Streamable HTTP 模式启动，供 claude.ai / 手机远程连接。
REM   端口 8768，访问路径 /mcp/<secret>；secret 从 .env 的 GEMINI_MCP_HTTP_SECRET 读。
REM   没设 secret（或还是占位符）会被服务器拒绝启动，属正常保护。
REM   公网暴露怎么配见 README 的「HTTP 模式」一节（推荐 Cloudflare Tunnel）。
REM ============================================================================

REM 先清掉可能占用 8768 端口的旧实例，避免"端口已被占用"报错。
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /C:":8768 " ^| findstr /C:"LISTENING"') do (
    echo [gemini-video-mcp] 端口 8768 被旧进程 PID %%p 占用，正在结束它...
    taskkill /F /PID %%p >nul 2>&1
)
timeout /t 1 /nobreak >nul

uv run python main.py --http %*

REM 保留窗口，让错误信息停留可见，而不是一闪而过。
echo.
echo [gemini-video-mcp] 服务器已退出。
pause
