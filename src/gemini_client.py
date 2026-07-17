"""Gemini 视频直传识别客户端（从 MoFox-Bot 主程序剥离的精简版）。

移植源：
    MoFox-Bot 的 aiohttp_gemini_client.py
    （重点是其中的 get_video_response / _upload_file_and_wait_active / _delete_file /
      _build_generation_config / _default_normal_response_parser）

与原版的主要差异（有意为之）：
    - 去掉了 bot 专属的依赖：APIProvider / ModelInfo 配置对象、client_registry 注册、
      BaseClient 基类、结构化 logger、payload_content 消息构造器等，全部换成普通入参。
    - 只保留“视频直传理解”这一条链路，其余（文本对话、embedding、音频转录、流式解析）都不搬。
    - 错误一律抛成面向用户的中文 GeminiVideoError，方便上层 MCP 工具直接展示给非开发者。

工程经验（原封不动保留，务必理解再改）：
    1. 双通道上传：文件 <= 14MB 走 inline_data（base64 内嵌，单次请求）；更大走 Files API
       （resumable 上传 -> 轮询到 ACTIVE -> 生成 -> 用完删除远端文件）。
    2. thinking token 陷阱：Gemini 3.5 的思考 token 也计入 maxOutputTokens，默认档会把输出
       预算吃光导致描述被截断。对策：客户端层默认 thinking_level="minimal"（MCP 服务器
       实际传入 GEMINI_THINKING_LEVEL 配置值，未配置时为 high），输出 token 下限 2048，
       检测到 finishReason == "MAX_TOKENS" 时在返回文本末尾追加告警。
    3. 低清模式：mediaResolution 低清约 100 token/秒，标清约 300 token/秒，长视频省钱开关。
    4. GIF 原格式直传：Gemini 能以 image/gif 感知完整动画，GIF 不要抽帧转 jpg。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("gemini-video-mcp")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 视频直传：inline_data 整个请求上限 20MB，base64 会膨胀约 33%，
# 原始视频取 14MB 以内走内联，超过则走 Files API。
VIDEO_INLINE_LIMIT_BYTES = 14 * 1024 * 1024

# Files API 单文件上限（免费层存储上限 2GB；文件 48 小时后自动清理）。
FILES_API_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

# 思考等级：Gemini 3.5 系列新增 "minimal"；感知型任务（看视频）用最低思考即可。
THINKING_LEVEL_MINIMAL = "minimal"
VALID_THINKING_LEVELS = ["minimal", "low", "medium", "high"]

# Gemini 3.5 起官方弃用采样参数（temperature/topP/topK），发送虽不报错但强烈不建议。
_NO_SAMPLING_MODEL_PREFIXES = ("gemini-3.5",)

# 安全阈值全开，避免视频里无伤大雅的内容被拦截导致空响应。
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
]

# 常见容器后缀 -> Gemini 可识别的 MIME 类型。
# 视频类型尽量沿用 Gemini 官方文档给出的写法（部分是非标准写法，如 video/mov / video/avi）。
# .gif 走 image/gif，让 Gemini 以原格式感知完整动画（不抽帧）。
_EXT_TO_MIME: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/mov",
    ".webm": "video/webm",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".wmv": "video/wmv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
    # mkv 不在 Gemini 官方支持列表里，按 matroska 标准 MIME 尽力尝试；
    # 若被 API 拒绝，请先转成 mp4（见 README「已知限制」）。
    ".mkv": "video/x-matroska",
    ".gif": "image/gif",
}

# 面向用户展示的“支持的格式”清单。
SUPPORTED_EXTENSIONS = sorted(_EXT_TO_MIME.keys())


class GeminiVideoError(Exception):
    """面向用户的中文错误。

    上层 MCP 工具会直接把 str(该异常) 展示给用户，因此消息必须是
    一句话就能看懂“出了什么事、该怎么办”的中文说明。
    """


def guess_mime_type(path: str) -> str:
    """根据文件后缀推断 MIME 类型；不认识的后缀直接抛中文错误。"""
    suffix = Path(path).suffix.lower()
    mime = _EXT_TO_MIME.get(suffix)
    if not mime:
        raise GeminiVideoError(
            f"不支持的文件格式：{suffix or '（无后缀）'}。"
            f"目前支持这些格式：{', '.join(SUPPORTED_EXTENSIONS)}。"
            "如果你的视频是别的格式，请先用播放器/剪辑软件导出成 mp4 再试。"
        )
    return mime


def _build_generation_config(
    max_output_tokens: int,
    thinking_level: str | None,
    model_identifier: str,
    low_resolution: bool,
) -> dict[str, Any]:
    """构建 Gemini 的 generationConfig。

    - maxOutputTokens：注意思考 token 也算在这里面。
    - Gemini 3.5 起弃用采样参数，这里对 3.5 系列不发送 temperature/topP/topK。
    - thinking_level：由调用方传入（MCP 服务器传 GEMINI_THINKING_LEVEL 配置值，未配置时 high；
      客户端方法自身默认 minimal），无效值忽略并告警。
    - low_resolution：开低清（约 100 token/秒），长视频省钱。
    """
    config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}

    # 旧模型（非 3.5）保留低温采样，行为与原 bot 视频链路一致（temperature=0.3）。
    if not model_identifier.startswith(_NO_SAMPLING_MODEL_PREFIXES):
        config["temperature"] = 0.3
        config["topK"] = 1
        config["topP"] = 1

    if thinking_level:
        if thinking_level in VALID_THINKING_LEVELS:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        else:
            logger.warning("无效的 thinking_level=%s，已忽略（有效值：%s）", thinking_level, VALID_THINKING_LEVELS)

    if low_resolution:
        config["mediaResolution"] = "MEDIA_RESOLUTION_LOW"

    return config


def _parse_normal_response(response_data: dict) -> tuple[str, tuple[int, int, int] | None, str | None]:
    """解析 Gemini 非流式响应。

    Returns:
        (正文文本, (prompt_tokens, output_tokens, total_tokens) 或 None, finishReason 或 None)

    Raises:
        GeminiVideoError: 响应里没有可用内容（被安全策略拦截 / 结构异常等）时，
            抛出人话中文说明。
    """
    # 整个请求层面的拦截（连候选都没有）。
    prompt_feedback = response_data.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")

    candidates = response_data.get("candidates") or []
    if not candidates:
        if block_reason:
            raise GeminiVideoError(
                f"这段视频的请求被 Gemini 的安全策略拦截了（原因：{block_reason}），没有返回任何描述。"
            )
        raise GeminiVideoError("Gemini 没有返回任何候选结果，可能是视频内容无法解析或服务端异常，建议稍后重试。")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")

    content_parts: list[str] = []
    content = candidate.get("content") or {}
    for part in content.get("parts", []) or []:
        # 带 thought 标记的 part 是模型的思考过程，不能混进正文。
        if "text" in part and not part.get("thought"):
            content_parts.append(part["text"])

    text = "".join(content_parts).strip()

    # 有 finishReason 但正文为空的常见情形，给出可读解释。
    if not text:
        if finish_reason == "SAFETY":
            raise GeminiVideoError("Gemini 判定该视频内容触发了安全限制，没有生成描述。")
        if finish_reason == "RECITATION":
            raise GeminiVideoError("Gemini 因版权/复述限制没有生成描述。")
        if finish_reason in ("PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            raise GeminiVideoError(f"Gemini 因内容策略（{finish_reason}）没有生成描述。")
        # MAX_TOKENS 且正文为空：说明预算全被思考吃掉了。
        if finish_reason == "MAX_TOKENS":
            raise GeminiVideoError(
                "输出预算被模型的“思考”过程吃光了，没能挤出正文。"
                "请调大 max_output_tokens（比如 8192），或确认 thinking_level 为 minimal。"
            )
        raise GeminiVideoError(
            f"Gemini 返回了空描述（finishReason={finish_reason}），建议重试或调大 max_output_tokens。"
        )

    usage_record: tuple[int, int, int] | None = None
    usage = response_data.get("usageMetadata")
    if usage:
        usage_record = (
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
            usage.get("totalTokenCount", 0),
        )

    return text, usage_record, finish_reason


class GeminiVideoClient:
    """用 aiohttp 与 Gemini REST API 通信、专做视频理解的无状态客户端。

    每次请求都新建 aiohttp.ClientSession（与原 bot 客户端一致），
    避免把 response 带出 session 作用域导致连接被回收后读取挂起。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 600,
    ):
        """
        Args:
            api_key: Gemini API key（从环境变量或 .env 读入，绝不硬编码）。
            model: 模型标识（model_identifier），默认 gemini-3.5-flash。
            base_url: Gemini 生成式语言 API 根地址。
            timeout: 单次请求超时秒数（视频较慢，默认给到 600 秒）。
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- 基础请求 ----

    def _session_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36"
            ),
        }

    def _raise_for_http(self, status: int, body: str) -> None:
        """把 Gemini 的 HTTP 错误状态码翻译成人话中文。"""
        snippet = (body or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        if status in (400,):
            raise GeminiVideoError(
                "Gemini 拒绝了请求（400）。常见原因：视频格式不被支持、请求体有误，或该模型不支持视频。"
                f"\n服务端说明：{snippet}"
            )
        if status in (401, 403):
            raise GeminiVideoError(
                "鉴权失败（401/403）。请检查 GEMINI_API_KEY 是否正确、是否已启用 Generative Language API，"
                f"以及该 key 是否有权限访问此模型。\n服务端说明：{snippet}"
            )
        if status == 404:
            raise GeminiVideoError(
                f"找不到模型或资源（404）。请检查 GEMINI_MODEL（当前：{self.model}）是否拼写正确、是否可用。"
                f"\n服务端说明：{snippet}"
            )
        if status == 429:
            raise GeminiVideoError(
                "触发了 Gemini 的限流/配额上限（429）。请稍等一会儿再试，或检查你的免费额度是否用完。"
                f"\n服务端说明：{snippet}"
            )
        if status >= 500:
            raise GeminiVideoError(
                f"Gemini 服务端出错（{status}），通常是临时故障，请稍后重试。\n服务端说明：{snippet}"
            )
        raise GeminiVideoError(f"Gemini 返回了异常状态码 {status}。\n服务端说明：{snippet}")

    async def _request_json(self, method: str, endpoint: str, data: dict | None = None) -> dict:
        """发起非流式请求并在 session 作用域内读完 JSON。

        - 网络层 aiohttp.ClientError 最多重试 3 次（每次间隔 1 秒）；
        - HTTP 状态码错误（4xx/5xx）立即失败、不重试，并翻译成中文。
        """
        url = f"{self.base_url}/{endpoint}?key={self.api_key}"

        max_retries = 3
        last_exception: Exception | None = None

        for _attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=max(self.timeout, 30)),
                    headers=self._session_headers(),
                ) as session:
                    if method.upper() == "POST":
                        response = await session.post(url, json=data, headers={"Accept": "application/json"})
                    else:
                        response = await session.get(url)

                    if response.status >= 400:
                        self._raise_for_http(response.status, await response.text())

                    # 必须在 session 作用域内读完 body。
                    return await response.json()

            except aiohttp.ClientError as e:
                last_exception = e
                await asyncio.sleep(1)

        raise GeminiVideoError(
            "多次尝试后仍无法连接 Gemini（网络错误）。请检查网络/代理是否能访问 "
            f"generativelanguage.googleapis.com。\n底层错误：{last_exception}"
        )

    # ---- Files API（大文件通道） ----

    def _upload_base_url(self) -> str:
        """推导 Files API 的上传端点根路径（…/upload/v1beta）。"""
        root = self.base_url
        if root.endswith("/v1beta"):
            root = root[: -len("/v1beta")]
        return f"{root}/upload/v1beta"

    async def _upload_file_and_wait_active(
        self, data: bytes, mime_type: str, display_name: str = "gemini_video_mcp"
    ) -> tuple[str, str]:
        """通过 Files API 上传文件并等待其变为 ACTIVE。

        使用官方 resumable 协议：start 拿到上传 URL -> 一次性 upload+finalize -> 轮询状态。

        Returns:
            (file_uri, file_name)：生成请求用 uri，删除时用 name（形如 files/abc123）。
        """
        start_url = f"{self._upload_base_url()}/files?key={self.api_key}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=max(self.timeout, 600)),
            headers=self._session_headers(),
        ) as session:
            # 第一步：发起 resumable 上传，拿到上传 URL。
            try:
                start_resp = await session.post(
                    start_url,
                    json={"file": {"display_name": display_name}},
                    headers={
                        "X-Goog-Upload-Protocol": "resumable",
                        "X-Goog-Upload-Command": "start",
                        "X-Goog-Upload-Header-Content-Length": str(len(data)),
                        "X-Goog-Upload-Header-Content-Type": mime_type,
                    },
                )
            except aiohttp.ClientError as e:
                raise GeminiVideoError(f"发起大文件上传时网络出错：{e}") from e

            if start_resp.status >= 400:
                self._raise_for_http(start_resp.status, await start_resp.text())
            upload_url = start_resp.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                raise GeminiVideoError("Files API 没有返回上传地址（X-Goog-Upload-URL），上传无法继续。")

            # 第二步：上传全部字节并 finalize。
            try:
                upload_resp = await session.post(
                    upload_url,
                    data=data,
                    headers={
                        "Content-Length": str(len(data)),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                    },
                )
            except aiohttp.ClientError as e:
                raise GeminiVideoError(f"上传视频字节时网络出错：{e}") from e

            if upload_resp.status >= 400:
                self._raise_for_http(upload_resp.status, await upload_resp.text())
            file_info = (await upload_resp.json()).get("file") or {}
            file_uri = file_info.get("uri", "")
            file_name = file_info.get("name", "")
            state = file_info.get("state", "")
            if not file_uri or not file_name:
                raise GeminiVideoError(f"Files API 上传响应缺少 uri/name，无法继续：{file_info}")

            # 第三步：等待视频处理完成（PROCESSING -> ACTIVE）。
            poll_url = f"{self.base_url}/{file_name}?key={self.api_key}"
            waited = 0.0
            while state == "PROCESSING" and waited < 180:
                await asyncio.sleep(2.0)
                waited += 2.0
                try:
                    poll_resp = await session.get(poll_url)
                except aiohttp.ClientError as e:
                    raise GeminiVideoError(f"轮询上传状态时网络出错：{e}") from e
                if poll_resp.status >= 400:
                    self._raise_for_http(poll_resp.status, await poll_resp.text())
                file_info = await poll_resp.json()
                state = file_info.get("state", "")

            if state != "ACTIVE":
                if state == "PROCESSING":
                    raise GeminiVideoError(
                        "视频上传后 180 秒内仍未处理完成（超时）。视频可能太大或太长，"
                        "建议裁短、压缩后再试，或开启低清模式。"
                    )
                raise GeminiVideoError(f"上传的视频未能就绪（state={state}），请重试或更换视频。")

            return file_uri, file_name

    async def _delete_file(self, file_name: str) -> None:
        """删除 Files API 上的文件（尽力而为，失败不抛——文件 48 小时后也会自动清理）。"""
        try:
            url = f"{self.base_url}/{file_name}?key={self.api_key}"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._session_headers(),
            ) as session:
                await session.delete(url)
        except Exception as e:  # noqa: BLE001 - 删除失败无所谓，仅记录
            logger.debug("删除 Files API 文件失败（将自动过期清理）：%s", e)

    # ---- 主流程：视频理解 ----

    async def describe_video(
        self,
        video_bytes: bytes,
        prompt: str,
        mime_type: str,
        max_output_tokens: int = 4096,
        low_resolution: bool = False,
        thinking_level: str = THINKING_LEVEL_MINIMAL,
    ) -> dict[str, Any]:
        """把整个视频（含音轨）直接交给 Gemini 分析并返回描述。

        - 小视频（<= 14MB）走 inline_data，单次请求完成；
        - 大视频走 Files API：上传 -> 等待 ACTIVE -> 生成 -> 尽力删除远端文件；
        - low_resolution=True 时用低媒体分辨率（约 100 token/秒 vs 标清约 300 token/秒）。

        Returns:
            dict: {
                "text": 描述正文,
                "usage": {"prompt_tokens", "output_tokens", "total_tokens"} 或 None,
                "finish_reason": Gemini 的 finishReason 或 None,
                "truncated": 是否因 MAX_TOKENS 被截断（bool）,
                "channel": "inline" 或 "files_api",
            }
        """
        if len(video_bytes) > FILES_API_LIMIT_BYTES:
            raise GeminiVideoError(
                f"文件太大（{len(video_bytes) / 1024 / 1024 / 1024:.2f}GB），"
                f"超过了 Gemini Files API 的单文件上限（2GB）。请先裁短或压缩视频。"
            )

        uploaded_file_name: str | None = None

        if len(video_bytes) <= VIDEO_INLINE_LIMIT_BYTES:
            channel = "inline"
            video_part: dict[str, Any] = {
                "inline_data": {"mime_type": mime_type, "data": base64.b64encode(video_bytes).decode()}
            }
        else:
            channel = "files_api"
            logger.info("视频大小 %.1fMB 超过内联上限，改走 Files API 上传", len(video_bytes) / 1024 / 1024)
            file_uri, uploaded_file_name = await self._upload_file_and_wait_active(video_bytes, mime_type)
            video_part = {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}

        generation_config = _build_generation_config(
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            model_identifier=self.model,
            low_resolution=low_resolution,
        )

        request_data = {
            "contents": [{"role": "user", "parts": [video_part, {"text": prompt}]}],
            "generationConfig": generation_config,
            "safetySettings": GEMINI_SAFETY_SETTINGS,
        }

        try:
            endpoint = f"models/{self.model}:generateContent"
            response_data = await self._request_json("POST", endpoint, request_data)
            text, usage_record, finish_reason = _parse_normal_response(response_data)

            truncated = finish_reason == "MAX_TOKENS"
            if truncated:
                # 截断检测：思考 token 计入 maxOutputTokens，思考太多会把可见输出挤掉。
                logger.warning("[%s] 视频描述被 maxOutputTokens 截断（思考 token 也计入上限）", self.model)

            usage_dict = None
            if usage_record:
                usage_dict = {
                    "prompt_tokens": usage_record[0],
                    "output_tokens": usage_record[1],
                    "total_tokens": usage_record[2],
                }

            return {
                "text": text,
                "usage": usage_dict,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "channel": channel,
            }
        finally:
            if uploaded_file_name:
                await self._delete_file(uploaded_file_name)
