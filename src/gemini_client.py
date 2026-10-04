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
import re
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

# Interactions API 端点（相对 base_url）：agentic 视频处理走这里，
# 由模型自己决定看哪一段、用什么帧率、要不要听音轨，长视频能省下大量媒体 token。
INTERACTIONS_ENDPOINT = "interactions"

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
    status 是触发它的 HTTP 状态码（非 HTTP 错误为 None），供上层判断要不要换 key 重试。
    """

    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status


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


def _normalize_usage(raw: dict | None) -> dict[str, int] | None:
    """把各种写法的用量字段归一成 {prompt_tokens, output_tokens, total_tokens}。

    generateContent 用 promptTokenCount/candidatesTokenCount/totalTokenCount；
    Interactions API 用 usage.input_tokens/output_tokens/total_tokens（也可能是 camelCase），
    这里一并兼容，取不到的字段按 0 计。
    """
    if not isinstance(raw, dict):
        return None

    def pick(*keys: str) -> int:
        for k in keys:
            v = raw.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    prompt_tokens = pick("promptTokenCount", "prompt_tokens", "promptTokens", "input_tokens", "inputTokens")
    output_tokens = pick(
        "candidatesTokenCount", "candidates_tokens", "output_tokens", "outputTokens", "completion_tokens"
    )
    total_tokens = pick("totalTokenCount", "total_tokens", "totalTokens")
    if not total_tokens:
        total_tokens = prompt_tokens + output_tokens
    if not (prompt_tokens or output_tokens or total_tokens):
        return None
    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


# Interactions 的 steps 里，这几种条目才是"给人看的正文"。
_INTERACTION_OUTPUT_TYPES = ("model_output", "text", "output_text", "message")
# 这几种是模型自己干活的痕迹（取帧、听音轨、思考），不能混进正文。
_INTERACTION_WORK_TYPES = ("processing_call", "processing_result")


def _extract_interaction_text(response_data: dict) -> str:
    """从 Interactions API 的响应里取出正文。

    首选顶层 output_text（SDK 里的 interaction.output_text）。实测 REST 响应没有这个字段，
    正文散在 steps 里，形如：
        processing_call / processing_result（模型去取某几秒的帧或音轨）
        thought（思考，只有签名没有文本）
        model_output（说话）
    agentic 是 Think→Act→Observe 的循环，中途的 model_output 常常是"我还得再听一段"这类
    自言自语，随后它又去取了新素材——那种话已被后续动作推翻，不该出现在给用户的描述里。
    所以这里【从后往前】收集：遇到 model_output 就收，遇到 thought 跳过（正文可能被思考切成几段），
    一旦遇到取素材的动作就停——只保留最后一次取素材之后说的话。
    """
    for key in ("output_text", "outputText"):
        value = response_data.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_thought_marker(value)

    steps = None
    for container_key in ("steps", "output", "outputs"):
        candidate = response_data.get(container_key)
        if isinstance(candidate, list) and candidate:
            steps = candidate
            break
    if not steps:
        return ""

    tail: list[dict] = []
    for item in reversed(steps):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in _INTERACTION_OUTPUT_TYPES:
            tail.append(item)
        elif item_type in _INTERACTION_WORK_TYPES and tail:
            break
        # thought 之类的：不收也不打断（正文可能被思考步骤切开）
    tail.reverse()

    chunks: list[str] = []

    def collect(item: Any) -> None:
        if not isinstance(item, dict) or item.get("thought"):
            return
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
            return
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content)
            return
        if isinstance(content, list):
            for sub in content:
                if isinstance(sub, dict) and str(sub.get("type") or "text") in _INTERACTION_OUTPUT_TYPES:
                    collect(sub)

    for item in tail:
        collect(item)

    return _strip_thought_marker("".join(chunks))


def _strip_thought_marker(text: str) -> str:
    """去掉正文开头漏出来的 "thought" 标记词。

    实测 gemini-3.8-flash 走 Interactions 时，最后一条 model_output 的文本偶尔以裸的
    "thought " 开头（思考段的类型标记被一起序列化进了文本）。只在正文最开头、且后面还有内容时才去掉。
    """
    stripped = (text or "").strip()
    match = re.match(r"^thought[\s：:]+(?=\S)", stripped, flags=re.IGNORECASE)
    if match:
        return stripped[match.end() :].strip()
    return stripped


def _normalize_interaction_usage(raw: dict | None) -> dict[str, int] | None:
    """把 Interactions API 的 usage 折算成与 generateContent 一致的三个数。

    实测字段：total_tokens / total_input_tokens / total_output_tokens /
    total_tool_use_tokens（agentic 自己取的帧与音轨，按输入计费）/ total_thought_tokens。
    所以「输入」= 提示词 + 它自己取回来的媒体，这才是 agentic 真正花掉的大头。
    """
    if not isinstance(raw, dict):
        return None
    if "total_tokens" not in raw and "total_input_tokens" not in raw:
        return _normalize_usage(raw)

    def num(key: str) -> int:
        value = raw.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    prompt_tokens = num("total_input_tokens") + num("total_tool_use_tokens")
    output_tokens = num("total_output_tokens")
    total_tokens = num("total_tokens") or (prompt_tokens + output_tokens + num("total_thought_tokens"))
    if not (prompt_tokens or output_tokens or total_tokens):
        return None
    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


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
                f"\n服务端说明：{snippet}",
                status=status,
            )
        if status in (401, 403):
            raise GeminiVideoError(
                "鉴权失败（401/403）。请检查 GEMINI_API_KEY 是否正确、是否已启用 Generative Language API，"
                f"以及该 key 是否有权限访问此模型。\n服务端说明：{snippet}",
                status=status,
            )
        if status == 404:
            raise GeminiVideoError(
                f"找不到模型或资源（404）。请检查 GEMINI_MODEL（当前：{self.model}）是否拼写正确、是否可用。"
                f"\n服务端说明：{snippet}",
                status=status,
            )
        if status == 429:
            raise GeminiVideoError(
                "触发了 Gemini 的限流/配额上限（429）。请稍等一会儿再试，或检查你的免费额度是否用完。"
                f"\n服务端说明：{snippet}",
                status=status,
            )
        if status >= 500:
            raise GeminiVideoError(
                f"Gemini 服务端出错（{status}），通常是临时故障，请稍后重试。\n服务端说明：{snippet}",
                status=status,
            )
        raise GeminiVideoError(f"Gemini 返回了异常状态码 {status}。\n服务端说明：{snippet}", status=status)

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

    async def upload_video(
        self, data: bytes, mime_type: str, display_name: str = "gemini_video_mcp"
    ) -> tuple[str, str]:
        """公开包装：把字节上传到 Files API 并等到 ACTIVE，返回 (file_uri, file_name)。

        上层（server 的上传缓存）需要自己掌握上传时机与远端文件的生命周期，所以把内部方法开出来。
        注意：这条路径【不】负责删除远端文件，删不删由调用方决定（Google 侧 48 小时后自动清理）。
        """
        return await self._upload_file_and_wait_active(data, mime_type, display_name)

    async def get_file_info(self, file_name: str) -> dict | None:
        """查询 Files API 上某个文件的元信息（state / expirationTime 等）；查不到或出错返回 None。

        给上传缓存做复用前的校验：state 必须是 ACTIVE，且没到过期时间。
        """
        if not file_name:
            return None
        try:
            return await self._request_json("GET", file_name)
        except Exception as e:  # noqa: BLE001 - 查不到就当缓存失效，不该把异常抛给用户
            logger.debug("查询 Files API 文件信息失败（当作缓存失效）：%s", e)
            return None

    async def delete_file(self, file_name: str) -> None:
        """公开包装：尽力删除 Files API 上的文件（失败不抛）。"""
        await self._delete_file(file_name)

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
        file_uri: str | None = None,
        video_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """把整个视频（含音轨）直接交给 Gemini 分析并返回描述。

        - 小视频（<= 14MB）走 inline_data，单次请求完成；
        - 大视频走 Files API：上传 -> 等待 ACTIVE -> 生成 -> 尽力删除远端文件；
        - 调用方已经自己上传好（如 server 侧的上传缓存）时传 file_uri：直接引用该远端文件，
          既不再上传、也【不】删除它（生命周期归调用方管），此时 video_bytes 可以传空字节；
        - video_metadata：只看视频的一段/换抽帧率时传，形如
          {"start_offset": "39s", "end_offset": "46s", "fps": 5}，原样塞进这个 part；
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
        if not file_uri and len(video_bytes) > FILES_API_LIMIT_BYTES:
            raise GeminiVideoError(
                f"文件太大（{len(video_bytes) / 1024 / 1024 / 1024:.2f}GB），"
                f"超过了 Gemini Files API 的单文件上限（2GB）。请先裁短或压缩视频。"
            )

        uploaded_file_name: str | None = None

        if file_uri:
            # 调用方（上传缓存）已经把文件放上去了：直接引用，不上传也不删除。
            channel = "files_api"
            video_part: dict[str, Any] = {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
        elif len(video_bytes) <= VIDEO_INLINE_LIMIT_BYTES:
            channel = "inline"
            video_part = {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(video_bytes).decode()}}
        else:
            channel = "files_api"
            logger.info("视频大小 %.1fMB 超过内联上限，改走 Files API 上传", len(video_bytes) / 1024 / 1024)
            uploaded_uri, uploaded_file_name = await self._upload_file_and_wait_active(video_bytes, mime_type)
            video_part = {"file_data": {"mime_type": mime_type, "file_uri": uploaded_uri}}

        # 只看某一段 / 换抽帧率：videoMetadata 挂在视频这个 part 上。
        if video_metadata:
            video_part["video_metadata"] = dict(video_metadata)

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

    async def describe_youtube(
        self,
        youtube_url: str,
        prompt: str,
        *,
        low_resolution: bool = False,
        max_output_tokens: int = 4096,
        thinking_level: str = THINKING_LEVEL_MINIMAL,
    ) -> dict[str, Any]:
        """把 YouTube 视频页链接交给 Gemini 云端直读并返回描述（不下载、不走 Files API）。

        Gemini 原生支持在 parts 里放 {"file_data": {"file_uri": "<YouTube 视频页 URL>"}}，
        由 Gemini 服务端自行拉取该 YouTube 视频，本地无需下载、也不占用 Files API 存储。
        注意这条通道的 file_data 【不带 mime_type】（与本地大文件的 file_data 不同，那边要带）。
        仅支持公开视频；免费层对 YouTube 每天有总时长限额。

        参数与返回结构均与 describe_video 保持一致，唯一区别是 channel 固定为 "youtube"，
        且没有 inline / files_api 之分（因此也没有远端文件需要清理）。

        Returns:
            dict: {
                "text": 描述正文,
                "usage": {"prompt_tokens", "output_tokens", "total_tokens"} 或 None,
                "finish_reason": Gemini 的 finishReason 或 None,
                "truncated": 是否因 MAX_TOKENS 被截断（bool）,
                "channel": "youtube",
            }
        """
        # YouTube 直读：file_data 只放 file_uri，不带 mime_type（Gemini 服务端自行识别）。
        video_part: dict[str, Any] = {"file_data": {"file_uri": youtube_url}}

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

        endpoint = f"models/{self.model}:generateContent"
        response_data = await self._request_json("POST", endpoint, request_data)
        text, usage_record, finish_reason = _parse_normal_response(response_data)

        truncated = finish_reason == "MAX_TOKENS"
        if truncated:
            # 截断检测：思考 token 计入 maxOutputTokens，思考太多会把可见输出挤掉。
            logger.warning("[%s] YouTube 视频描述被 maxOutputTokens 截断（思考 token 也计入上限）", self.model)

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
            "channel": "youtube",
        }

    # ---- Interactions API：agentic 视频理解 ----

    async def describe_video_agentic(
        self,
        *,
        prompt: str,
        file_uri: str,
        mime_type: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """走 Interactions API 的 agentic 模式识别视频（模型自行决定看哪段、什么帧率、要不要听音轨）。

        与 generateContent 的静态通道（默认 1 FPS 全程抽帧）相比，长视频能省下大量媒体 token；
        官方指南建议 5 分钟以内的短片、或需要逐帧全程描述时仍用静态。

        Args:
            prompt: 提示词（按官方示例放在视频【之后】）。
            file_uri: 视频地址——Files API 上传后拿到的 uri，或公开的 YouTube 视频页链接。
            mime_type: 视频 MIME；YouTube 链接不要传（服务端自行识别）。
            model: 覆盖本客户端的默认模型（agentic 不是所有模型都支持，可单独指定）。

        Returns:
            与 describe_video 同构的 dict，channel 固定为 "interactions"，
            另外多一个 "model" 字段标出这次实际用的模型。

        Raises:
            GeminiVideoError: 端点不可用、模型不支持 agentic、或响应里找不到正文时。
        """
        used_model = (model or self.model).strip()

        video_input: dict[str, Any] = {"type": "video", "uri": file_uri, "processing": "agentic"}
        if mime_type:
            video_input["mime_type"] = mime_type

        request_data = {
            "model": used_model,
            "input": [video_input, {"type": "text", "text": prompt}],
        }

        response_data = await self._request_json("POST", INTERACTIONS_ENDPOINT, request_data)

        text = _extract_interaction_text(response_data)
        if not text:
            top_keys = ", ".join(sorted(response_data.keys())) if isinstance(response_data, dict) else "（非对象响应）"
            raise GeminiVideoError(
                "Interactions API（agentic 模式）返回的结果里找不到正文文本。"
                f"\n响应顶层字段：{top_keys}"
            )

        usage_dict = _normalize_interaction_usage(response_data.get("usage") or response_data.get("usageMetadata"))

        return {
            "text": text,
            "usage": usage_dict,
            "finish_reason": response_data.get("finish_reason") or response_data.get("status"),
            "truncated": False,
            "channel": "interactions",
            "model": used_model,
        }
