"""Gemini-Video-MCP 启动入口。

用法：
    uv run python main.py          # stdio（默认，Claude Desktop / Claude Code 本地注册）
    uv run python main.py --http   # Streamable HTTP（0.0.0.0:8768，路径 /mcp/<secret>，供 claude.ai 远程连接）

--http 模式需要设置 GEMINI_MCP_HTTP_SECRET（详见 README「HTTP 模式」一节与 .env.example）。

API key 读取顺序：环境变量 GEMINI_API_KEY -> 本目录下的 .env 文件。
"""

import sys

from src import main
from src.config import GEMINI_API_KEY, GEMINI_MODEL

if __name__ == "__main__":
    # 只往 stderr 打印，避免污染 stdio 的 JSON-RPC 通道。
    if not GEMINI_API_KEY:
        print(
            "[gemini-video-mcp] 警告：未找到 GEMINI_API_KEY。"
            "请在 .env 中设置或用 -e GEMINI_API_KEY=... 注入，否则工具会返回配置引导。",
            file=sys.stderr,
        )
    else:
        print(f"[gemini-video-mcp] 已就绪，模型：{GEMINI_MODEL}", file=sys.stderr)

    main()
