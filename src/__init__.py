"""Gemini-Video-MCP：把 MoFox-Bot 的 Gemini 视频直传识别能力独立成 MCP 服务器。"""

from .server import main, mcp

__version__ = "0.1.0"

__all__ = [
    "main",
    "mcp",
]
