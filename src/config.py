"""配置加载：API key / 模型名 / base_url。

读取优先级（与任务要求一致）：
    1. 环境变量（如通过 `claude mcp add -e GEMINI_API_KEY=...` 注入，或 shell export）；
    2. 服务器根目录下的 `.env` 文件。

用 python-dotenv 加载 `.env`，且【不】开 override，
因此启动时已存在的环境变量优先级更高（.env 只作兜底）。
若 python-dotenv 不可用，则退回到一个几行的极简手写解析器。
"""

from __future__ import annotations

import os
from pathlib import Path

# 服务器根目录 = 本文件所在 src/ 的上一级。
_SERVER_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _SERVER_DIR / ".env"


def _load_env_file(path: Path) -> None:
    """把 .env 里的键值加载进 os.environ（不覆盖已存在的环境变量）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except Exception:
        # python-dotenv 缺失或异常时，退回极简解析（同样不覆盖已存在的环境变量）。
        pass

    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_env_file(_ENV_PATH)

# API key：没有默认值（缺失时上层工具会给出清晰的中文引导）。
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# 模型名：默认沿用 MoFox-Bot 里 utils_video 走 Google 直连用的模型标识。
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# API 根地址：默认 Google 官方生成式语言 API。
GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()

# 思考等级：默认 high——实测思考越多文学发挥越好（sol 钦点"对 flash 慷慨点"）。
# 想省钱可在 .env 里改成 minimal。有效值 minimal/low/medium/high；
# 模型不支持时服务器会自动降级为 low 重试一次。
GEMINI_THINKING_LEVEL: str = os.getenv("GEMINI_THINKING_LEVEL", "high").strip().lower()

# ---------------------------------------------------------------------------
# HTTP 远程模式配置（仅 `python main.py --http` 时用到；stdio 本地模式完全不涉及）
# ---------------------------------------------------------------------------
# 访问口令：会被拼进访问地址的路径（/mcp/<secret>），是这个端口对外的【唯一门锁】。
# 因为 --http 模式的端口通常经 Cloudflare Tunnel 之类暴露到公网，所以缺失或仍是占位符时，
# 服务器会【拒绝】以 --http 启动（见 src/server.py 里的校验），绝不允许裸奔。
# 读取优先级与其它配置一致：环境变量 GEMINI_MCP_HTTP_SECRET > 本目录 .env。
GEMINI_MCP_HTTP_SECRET: str = os.getenv("GEMINI_MCP_HTTP_SECRET", "").strip()

# 公网根地址（可选）：本服务经 Cloudflare Tunnel 等暴露后的对外地址（如 https://xxx.example.com）。
# get_upload_url 工具用它拼出沙盒可达的上传地址；不设则退回 http://localhost:8768（仅本机可用）。
GEMINI_MCP_PUBLIC_BASE_URL: str = os.getenv("GEMINI_MCP_PUBLIC_BASE_URL", "").strip().rstrip("/")
