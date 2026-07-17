"""Gemini-Video-MCP 服务器：把视频交给 Gemini 直传识别，并把画面/帧投给调用方模型看。

传输方式（两种，默认 stdio）：
    - stdio：`python main.py`（不带参数）。Claude Desktop / Claude Code 本地注册用，默认行为，
      也是这个服务器的主用法。
    - Streamable HTTP：`python main.py --http`。绑 0.0.0.0:8768，路径 /mcp/<secret>，
      供 claude.ai（远程 MCP / Custom Connector）或手机远程连接。详见 README「HTTP 模式」一节。

工具（宁少勿多；stdio / HTTP 两模式都可用）：
    - describe_video：主工具。输入【本地】视频路径，返回按时间轴分段的详细内容描述（画面 + 音轨）。
    - describe_video_url：输入视频文件【直链】，下载到临时目录后走同一条识别管线（用完即删）。
    - view_media：把一张图片、或视频的某一帧，作为【图片内容】直接返回，让调用方模型亲眼看到画面。
    - estimate_cost：小工具。估算发这个视频大概消耗多少输入 token，让人心里有数。

仅 --http 模式额外提供一个 HTTP 端点：
    - POST /upload/<secret>：让 claude.ai 沙盒把它下载/生成的文件推到本机 temp_media/，
      再把返回的 saved_path 交给 describe_video / view_media 识别。stdio 模式【不】注册这个端点。

注意：工具一律返回【字符串】或【图片内容】，绝不返回 dict——dict 在 Claude 界面里会渲染成难读的并排 JSON。
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from .config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MCP_HTTP_SECRET,
    GEMINI_MCP_PUBLIC_BASE_URL,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
)
from .gemini_client import (
    _EXT_TO_MIME,
    FILES_API_LIMIT_BYTES,
    VIDEO_INLINE_LIMIT_BYTES,
    GeminiVideoClient,
    GeminiVideoError,
    guess_mime_type,
)

# 日志一律走 stderr——stdio 传输下 stdout 是 JSON-RPC 通道，绝不能污染。
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[gemini-video-mcp] %(message)s")
logger = logging.getLogger("gemini-video-mcp")

# 服务器级 instructions：随 MCP initialize 下发给客户端，让对端模型（如 claude.ai 里的 Claude）
# 不用人教就知道"聊天里上传的文件怎么搬运到本服务器"。
_SERVER_INSTRUCTIONS = """本服务器的工具都以【这台服务器所在电脑】上的文件路径为输入。
如果用户在聊天里上传了视频/图片（它们在你的代码沙盒里，如 /mnt/user-data/uploads/），
而不是给出服务器上的路径，请按以下配方把文件搬运过来再识别：
1. 调用 get_upload_url 工具，拿到上传地址；
2. 在你的代码沙盒里执行：requests.post(上传地址, files={"file": open("沙盒里的文件路径", "rb")})；
3. 用响应 JSON 里的 saved_path 调用 describe_video（视频识别）或 view_media（亲眼看画面）。
沙盒能访问外网时这条链路才通；失败时把报错如实告诉用户即可。"""

mcp = FastMCP(name="Gemini Video MCP Server", instructions=_SERVER_INSTRUCTIONS)
READONLY = ToolAnnotations(readOnlyHint=True)

# --http 模式启动时置 True（get_upload_url 用它判断上传端点是否真的存在）。
_HTTP_MODE_ACTIVE = False

# 输出 token 下限：低于此值时思考过程容易把可见输出挤没，强制抬到 2048。
_MIN_OUTPUT_TOKENS = 2048

# token 消耗速率（sol 在 bot 里实测的经验值）：标清约 300/秒，低清约 100/秒。
_TOKENS_PER_SEC_STANDARD = 300
_TOKENS_PER_SEC_LOW = 100

# 无法读到时长时，按文件大小粗估时长用的假设码率（约 1.5 Mbps）。
_ASSUMED_BYTES_PER_SEC = 1_500_000 / 8  # ≈ 187500 B/s ≈ 0.18 MB/s

# ---------------------------------------------------------------------------
# 远程文件通道基建常量（describe_video_url 下载 + --http 上传端点共用；stdio 本地识别不涉及）
# ---------------------------------------------------------------------------
# 临时目录：下载/上传的文件都落这里，用完即删；目录总量设 2GB 上限，超了按最旧先删腾位。
_TEMP_MEDIA_DIR = Path(__file__).resolve().parent.parent / "temp_media"

# 单个远程文件（下载 / 上传）大小上限：500MB。
_REMOTE_FILE_MAX_BYTES = 500 * 1024 * 1024
# temp_media 目录总量上限：2GB。
_TEMP_MEDIA_MAX_BYTES = 2 * 1024 * 1024 * 1024

# 下载超时：连接 15 秒 / 总计 300 秒。
_DOWNLOAD_CONNECT_TIMEOUT = 15
_DOWNLOAD_TOTAL_TIMEOUT = 300
# 下载 / 落盘的分块大小（1MB）。
_STREAM_CHUNK_BYTES = 1024 * 1024

# HTTP 响应 Content-Type -> Gemini 可识别的 MIME（URL 后缀认不出时的兜底映射）。
_CONTENT_TYPE_TO_MIME: dict[str, str] = {
    "video/mp4": "video/mp4",
    "video/quicktime": "video/mov",
    "video/webm": "video/webm",
    "video/x-msvideo": "video/avi",
    "video/avi": "video/avi",
    "video/x-flv": "video/x-flv",
    "video/x-ms-wmv": "video/wmv",
    "video/wmv": "video/wmv",
    "video/mpeg": "video/mpeg",
    "video/3gpp": "video/3gpp",
    "video/x-matroska": "video/x-matroska",
    "image/gif": "image/gif",
}

# view_media 相关：图片/视频后缀集合与缩放边长的合理区间。
_VIEW_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# 视频后缀 = 识别支持的全部后缀去掉 .gif（gif 归到图片一侧，取首帧）。
_VIEW_VIDEO_EXTS = {ext for ext in _EXT_TO_MIME if ext != ".gif"}
_VIEW_MIN_DIMENSION = 16
_VIEW_MAX_DIMENSION = 4096


# ---------------------------------------------------------------------------
# 默认提示词
# ---------------------------------------------------------------------------
# 源自 MoFox-Bot 的 batch_analysis_prompt（sol 亲测“效果好到浮夸”的那版），
# 有意【不】加任何压制文风的措辞（不禁止抒情、不要求客观简洁），保留 Gemini 自由发挥的戏剧张力。
# 原文两处重复的编号“6.”这里顺手改成 6 / 7；两行人设占位符改由 persona 参数按需插入。

_DEFAULT_PROMPT_HEAD = (
    '请观看这个视频，并将其"翻译"成文字，让它成为一个就连只支持文字的llm也能'
    "体会到视频内容的细致，生动，令人神往或感同身受的描述。"
)

_DEFAULT_PROMPT_BODY = """请提供详细的视频内容讲述，提供详细的，精细到时间戳的描述，涵盖以下方面：
1. 视频的整体内容和主题，风格如何？分类是什么？例如，是艺术作品、meme分享，抑或是随手拍片？
2. 对主要人物、台词、对象和场景的总结和详细叙述
3. 如有，也带上动作、情节和时间线发展，以及最高光的时刻
4. 视频的视觉风格和艺术特点是什么？画面质量如何？流畅还是卡顿？给人一种什么样的视觉感受？
5. 整体氛围和情感表达了什么？
6. 是否有背景音乐/音效、背景音？如有，是什么样的感觉？是什么风格的？是否具有音乐卡点？音乐在此处起到了什么作用（反差？讽刺？增强气氛？），给人什么样的听觉和感官体验？
7. 任何特殊的视觉效果或文字内容

请用中文回答，结果要详细准确。"""


def _build_default_prompt(persona: str | None, hint: str | None = None) -> str:
    """拼出默认提示词；传了 persona 就在开头位置附体一行人设，传了 hint 就注入人类观看者的前置线索。"""
    segments = [_DEFAULT_PROMPT_HEAD]
    if persona:
        segments.append(f"你的人设是：{persona.strip()}")
    if hint:
        segments.append(
            f"观看前的已知信息：人类观看者对这个视频的形容是：「{hint.strip()}」。\n"
            "这可以作为理解视频的线索（尤其当画面或声音比较抽象、难以直接归类时），"
            "但请以你实际看到、听到的内容为准，不要为了迎合这个形容而虚构不存在的细节。"
        )
    segments.append(_DEFAULT_PROMPT_BODY)
    # 用空行分隔各段，还原原版排版观感。
    return "\n\n".join(segments)


# ---------------------------------------------------------------------------
# 工具实现的公共校验
# ---------------------------------------------------------------------------


def _check_api_key() -> str | None:
    """API key 缺失时返回可读中文引导，否则返回 None。"""
    if not GEMINI_API_KEY:
        return (
            "没有找到 Gemini API key。请二选一：\n"
            "  1) 在本服务器目录放一个 .env 文件，写上 GEMINI_API_KEY=你的key（可参考 .env.example）；\n"
            "  2) 注册 MCP 时用 -e GEMINI_API_KEY=你的key 传入。\n"
            "API key 可在 Google AI Studio（https://aistudio.google.com/apikey）免费申领。"
        )
    return None


def _validate_local_file(path: str) -> tuple[Path | None, str | None]:
    """校验本地文件；返回 (Path, None) 或 (None, 中文错误)。"""
    if not path or not path.strip():
        return None, "没有提供文件路径。请把视频文件的完整路径传给 path 参数。"
    p = Path(path.strip('"').strip("'"))
    if not p.exists():
        return None, f"找不到这个文件：{p}\n请确认路径拼写正确、文件确实存在（建议用完整绝对路径）。"
    if not p.is_file():
        return None, f"这个路径不是文件（可能是文件夹）：{p}\n请指向具体的视频文件。"
    return p, None


# ---------------------------------------------------------------------------
# 识别管线的共享核心（describe_video 与 describe_video_url 都复用这一段）
# ---------------------------------------------------------------------------


async def _describe_video_bytes(
    video_bytes: bytes,
    mime_type: str,
    *,
    prompt: str | None,
    persona: str | None,
    hint: str | None,
    low_resolution: bool,
    max_output_tokens: int,
) -> str:
    """识别管线的共享核心：拿到【视频字节 + MIME】后，组装提示词、调用 Gemini、拼装中文返回文本。

    describe_video（本地文件）与 describe_video_url（直链下载）都复用这一段，确保两条入口的
    提示词组装、思考等级自动降级重试、用量/截断页脚逻辑完全一致、不漂移。
    调用方需自行保证：API key 已存在、video_bytes 非空、mime_type 已判定。
    """
    # 组装提示词（prompt 优先，其次 persona/hint 版默认模板）
    final_prompt = prompt.strip() if prompt and prompt.strip() else _build_default_prompt(persona, hint)

    # 输出 token 下限保护（思考 token 也计入此上限，太小会把正文挤没）
    try:
        tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        tokens = 4096
    effective_tokens = max(tokens, _MIN_OUTPUT_TOKENS)

    client = GeminiVideoClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL, base_url=GEMINI_BASE_URL)
    try:
        try:
            result = await client.describe_video(
                video_bytes=video_bytes,
                prompt=final_prompt,
                mime_type=mime_type,
                max_output_tokens=effective_tokens,
                low_resolution=low_resolution,
                thinking_level=GEMINI_THINKING_LEVEL,
            )
        except GeminiVideoError as e:
            # 个别模型不支持当前思考等级（会 400），自动降级为 low 重试一次
            if "thinking level" not in str(e).lower():
                raise
            logger.info("模型 %s 不支持 thinking_level=%s，自动改用 low 重试", GEMINI_MODEL, GEMINI_THINKING_LEVEL)
            result = await client.describe_video(
                video_bytes=video_bytes,
                prompt=final_prompt,
                mime_type=mime_type,
                max_output_tokens=effective_tokens,
                low_resolution=low_resolution,
                thinking_level="low",
            )
    except GeminiVideoError as e:
        return f"视频识别失败：{e}"
    except Exception as e:  # noqa: BLE001 - 兜底：任何异常都不许裸抛出 MCP 边界
        logger.exception("视频识别管线未预期异常")
        return f"发生了未预期的错误：{e}\n（如果反复出现，请把这条信息发给开发者。）"

    # 组装返回文本
    text = result["text"]
    footer_lines: list[str] = []
    if result.get("truncated"):
        footer_lines.append(
            "⚠️ 提示：描述可能被输出长度上限截断了。可调大 max_output_tokens（如 8192）后重试以获得完整内容。"
        )
    usage = result.get("usage")
    if usage:
        footer_lines.append(
            f"（用量：输入 {usage['prompt_tokens']:,} token，输出 {usage['output_tokens']:,} token，"
            f"合计 {usage['total_tokens']:,} token；通道：{result.get('channel')}）"
        )
    if footer_lines:
        text = text + "\n\n---\n" + "\n".join(footer_lines)
    return text


# ---------------------------------------------------------------------------
# temp_media 临时目录管理 + 直链下载（describe_video_url / 上传端点共用）
# ---------------------------------------------------------------------------


def _ensure_temp_media_dir() -> None:
    """确保 temp_media 目录存在（不存在则创建；失败静默——上层写文件时自会报错）。"""
    try:
        _TEMP_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _enforce_temp_media_quota() -> None:
    """temp_media 总量超过 2GB 时，按最旧优先删除文件腾位（尽力而为，失败静默）。

    只删除 temp_media 目录下的普通文件，绝不触碰目录外任何东西。属阻塞操作（一次目录遍历），
    异步上下文里请用 asyncio.to_thread 包一层调用。
    """
    try:
        if not _TEMP_MEDIA_DIR.exists():
            return
        files = [f for f in _TEMP_MEDIA_DIR.iterdir() if f.is_file()]
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                continue
        if total <= _TEMP_MEDIA_MAX_BYTES:
            return
        # 按修改时间从旧到新，逐个删除直到降到上限以下。
        files.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0.0)
        for f in files:
            if total <= _TEMP_MEDIA_MAX_BYTES:
                break
            try:
                sz = f.stat().st_size
                f.unlink()
                total -= sz
            except OSError:
                continue
    except OSError:
        pass


def _safe_temp_name(raw_name: str) -> str:
    """把调用方可控的文件名安全化：剥掉任何目录成分 + 白名单字符 + 时间戳前缀防覆盖。

    - Path(raw).name 先剥掉 ../、绝对盘符等目录穿越成分（`../../evil` -> `evil`）；
    - 基础名/后缀只保留 [A-Za-z0-9._-]，其余一律替换为下划线；
    - 加毫秒级时间戳前缀，避免同名覆盖。
    结果不含任何路径分隔符，天然无法逃出 temp_media。
    """
    base = Path(raw_name or "").name  # 关键：剥离目录穿越成分
    stem = Path(base).stem
    suffix = Path(base).suffix
    stem_safe = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:60].strip("._") or "file"
    suffix_safe = re.sub(r"[^A-Za-z0-9.]", "", suffix)[:12]
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.monotonic() * 1000) % 1000:03d}"
    return f"{ts}_{stem_safe}{suffix_safe}"


def _resolve_in_temp_media(name: str) -> Path:
    """把安全化后的文件名拼进 temp_media，并二次校验最终路径确实落在目录内（纵深防御）。"""
    root = _TEMP_MEDIA_DIR.resolve()
    dest = (root / name).resolve()
    if dest.parent != root:
        # 理论上不会发生（name 已无分隔符），仍做纵深防御。
        raise GeminiVideoError("内部错误：生成的存储路径越界，已阻止。")
    return dest


def _detect_remote_mime(url_path: str, content_type: str | None) -> tuple[str | None, str | None]:
    """判定远程视频的 MIME：优先 URL 路径后缀，其次响应 Content-Type，都认不出则给中文错误。"""
    suffix = Path(url_path).suffix.lower()
    if suffix in _EXT_TO_MIME:
        return _EXT_TO_MIME[suffix], None
    ct = (content_type or "").split(";")[0].strip().lower()
    mapped = _CONTENT_TYPE_TO_MIME.get(ct)
    if mapped:
        return mapped, None
    return None, (
        "无法确定这个链接指向的视频格式：URL 后缀和服务器返回的类型都没能认出来。\n"
        f"（URL 后缀：{suffix or '无'}；响应 Content-Type：{ct or '无'}）\n"
        "请确认这是一个直接指向视频文件的直链（以 .mp4/.mov/.webm 等结尾），而不是视频网站的播放页面。"
    )


async def _download_to_temp(url: str) -> tuple[Path | None, str | None, str | None]:
    """把 http/https 直链视频下载到 temp_media 临时文件。

    Returns:
        (临时文件 Path, MIME, None) 表示成功；或 (None, None, 中文错误) 表示失败。
        失败时本函数会清掉自己写了一半的临时文件；成功时由调用方负责最终删除。

    健壮性：仅放行 http/https；Content-Length 或流式累计超过 500MB 即中止；连接 15s / 总 300s 超时；
    非 2xx、text/html 页面都给可读中文报错。任何异常都被兜住、不裸抛。
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return None, None, f"只支持 http/https 开头的视频直链。\n（你给的协议是：{parsed.scheme or '（无）'}）"
    if not parsed.netloc:
        return None, None, "这个 URL 不完整（缺少主机名），请提供完整的视频直链。"

    await asyncio.to_thread(_enforce_temp_media_quota)
    _ensure_temp_media_dir()
    dest = _resolve_in_temp_media(_safe_temp_name(Path(parsed.path).name or "video"))

    timeout = aiohttp.ClientTimeout(total=_DOWNLOAD_TOTAL_TIMEOUT, connect=_DOWNLOAD_CONNECT_TIMEOUT)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Gemini-Video-MCP/1.0)"}
    file_handle = None
    success = False
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status < 200 or resp.status >= 300:
                    return (
                        None,
                        None,
                        (f"下载失败：服务器返回状态码 {resp.status}。\n请确认这个直链仍有效、没过期、不需要登录。"),
                    )
                ctype = resp.headers.get("Content-Type")
                ct_main = (ctype or "").split(";")[0].strip().lower()
                if ct_main.startswith("text/html"):
                    return (
                        None,
                        None,
                        (
                            "这看起来是一个网页，不是视频直链（服务器返回的是 HTML 页面）。\n"
                            "平台页面链接（B站/抖音/TikTok/YouTube 等的视频页）不是直链、无法直接下载识别。\n"
                            "请提供一个直接以 .mp4/.mov/.webm 等结尾、点开就是视频本体的直链。"
                        ),
                    )
                clen_raw = resp.headers.get("Content-Length")
                if clen_raw and clen_raw.isdigit() and int(clen_raw) > _REMOTE_FILE_MAX_BYTES:
                    return (
                        None,
                        None,
                        (
                            f"视频太大（约 {int(clen_raw) / 1024 / 1024:.0f}MB），超过 500MB 上限。\n"
                            "请先裁短/压缩，或改用本地 describe_video。"
                        ),
                    )
                mime, mime_err = _detect_remote_mime(parsed.path, ctype)
                if mime_err:
                    return None, None, mime_err

                written = 0
                file_handle = await asyncio.to_thread(open, dest, "wb")
                async for chunk in resp.content.iter_chunked(_STREAM_CHUNK_BYTES):
                    written += len(chunk)
                    if written > _REMOTE_FILE_MAX_BYTES:
                        return (
                            None,
                            None,
                            ("下载中止：文件已超过 500MB 上限。\n请先裁短/压缩视频，或改用本地 describe_video。"),
                        )
                    await asyncio.to_thread(file_handle.write, chunk)
                if written == 0:
                    return None, None, "下载到的内容是空的（0 字节），请确认这个直链有效。"
        success = True
        return dest, mime, None
    except asyncio.TimeoutError:
        return (
            None,
            None,
            (
                f"下载超时（连接 {_DOWNLOAD_CONNECT_TIMEOUT} 秒 / 总计 {_DOWNLOAD_TOTAL_TIMEOUT} 秒都没完成）。\n"
                "可能链接太慢或文件太大，请稍后重试或换更快的直链。"
            ),
        )
    except aiohttp.ClientError as e:
        return None, None, f"下载出错（网络问题）：{e}\n请检查链接是否可访问、网络/代理是否正常。"
    except OSError as e:
        return None, None, f"写入下载文件时出错（磁盘/权限问题）：{e}"
    except Exception as e:  # noqa: BLE001 - 兜底：绝不让异常裸抛出去
        logger.exception("下载视频直链未预期异常")
        return None, None, f"下载时发生了未预期的错误：{e}"
    finally:
        if file_handle is not None:
            try:
                await asyncio.to_thread(file_handle.close)
            except OSError:
                pass
        if not success:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 工具 1：describe_video（主工具，本地视频）
# ---------------------------------------------------------------------------


@mcp.tool(structured_output=False)
async def describe_video(
    path: str,
    prompt: str | None = None,
    persona: str | None = None,
    hint: str | None = None,
    low_resolution: bool = False,
    max_output_tokens: int = 8192,
) -> str:
    """把本地视频交给 Gemini 直传识别，返回按时间轴分段的详细内容描述（画面 + 音轨）。

    支持 mp4/mov/webm/avi/mkv 等常见视频格式，也直接支持 .gif（以原格式感知完整动画，不抽帧）。

    Args:
        path: 本地视频文件的路径（建议用完整绝对路径）。
        prompt: 自定义提示词。传了就【完全覆盖】默认模板（此时 persona/hint 参数被忽略）。
            不传则使用内置的“翻译给纯文字 LLM 看”的详细描述模板。
        persona: 可选的人设。传了会在默认提示词开头附体一行“你的人设是：…”，
            让 Gemini 以该人格来解说视频。仅在未提供 prompt 时生效。
        hint: 可选的前置线索——人类观看者对这个视频的形容或背景信息
            （如“这是在用鸡蛋下五子棋”“声音对应表情包”）。会注入默认模板，
            帮助模型理解抽象、玩梗类内容；同时要求模型以实际所见为准、不迎合虚构。
            仅在未提供 prompt 时生效。
        low_resolution: 低清模式。开启后约 100 token/秒（标清约 300 token/秒），长视频省钱，
            但画面细节会变粗。默认关闭。
        max_output_tokens: 最大输出 token 数，默认 8192。内部有 2048 的下限保护
            （Gemini 的“思考”token 也计入这里；默认思考等级为 high，思考会占用
            数千 token，太小会把正文挤没。思考等级可用环境变量 GEMINI_THINKING_LEVEL 调整）。

    Returns:
        Gemini 生成的视频内容描述文本（末尾可能带用量统计或截断提示）。
    """
    # 1) API key
    err = _check_api_key()
    if err:
        return err

    # 2) 文件校验
    p, err = _validate_local_file(path)
    if err:
        return err
    assert p is not None

    # 3) 格式（MIME）判定
    try:
        mime_type = guess_mime_type(str(p))
    except GeminiVideoError as e:
        return str(e)

    # 4) 先按文件大小拦截（用 stat，避免把超大文件整个读进内存才发现超限而 OOM）
    try:
        size_bytes = p.stat().st_size
    except OSError as e:
        return f"无法读取文件信息（可能没有权限或文件被占用）：{p}\n底层错误：{e}"
    if size_bytes == 0:
        return f"这个文件是空的（0 字节）：{p}"
    if size_bytes > FILES_API_LIMIT_BYTES:
        return (
            f"文件太大（{size_bytes / 1024 / 1024 / 1024:.2f}GB），"
            "超过 Gemini 单文件上限（2GB）。请先裁短或压缩视频再试。"
        )

    # 5) 读取字节（放线程池，避免阻塞事件循环）
    try:
        video_bytes = await asyncio.to_thread(p.read_bytes)
    except OSError as e:
        return f"无法读取文件（可能没有权限或文件被占用）：{p}\n底层错误：{e}"
    if not video_bytes:
        return f"这个文件是空的（0 字节）：{p}"

    # 6) 交给共享识别管线（组装提示词 + 调 Gemini + 拼装返回文本）
    return await _describe_video_bytes(
        video_bytes,
        mime_type,
        prompt=prompt,
        persona=persona,
        hint=hint,
        low_resolution=low_resolution,
        max_output_tokens=max_output_tokens,
    )


# ---------------------------------------------------------------------------
# 工具 2：describe_video_url（视频直链 → 下载 → 复用识别管线 → 用完即删）
# ---------------------------------------------------------------------------


@mcp.tool(structured_output=False)
async def describe_video_url(
    url: str,
    prompt: str | None = None,
    persona: str | None = None,
    hint: str | None = None,
    low_resolution: bool = False,
    max_output_tokens: int = 8192,
) -> str:
    """从【网络直链】下载视频再交给 Gemini 识别，返回按时间轴分段的中文描述。

    适用于：视频不在本机，但你有一个“点开就是视频本体”的直接下载链接（以 .mp4/.mov/.webm 等结尾）。
    下载的文件会存到服务器 temp_media/ 临时目录，识别完就删掉。

    ⚠️ 只支持【视频文件直链】。视频网站的播放【页面】链接（B站/抖音/TikTok/YouTube 等）不是直链、
       无法解析（那需要 yt-dlp 之类工具，本服务器暂不支持）。若链接打开是网页而非视频文件，会明确报错。

    其余参数（prompt/persona/hint/low_resolution/max_output_tokens）含义与 describe_video 完全一致。

    Args:
        url: 视频文件的 http/https 直链。
        prompt: 自定义提示词，传了就完全覆盖默认模板（此时 persona/hint 被忽略）。
        persona: 可选人设，仅在未传 prompt 时生效。
        hint: 可选前置线索，仅在未传 prompt 时生效。
        low_resolution: 低清省钱开关，默认关闭。
        max_output_tokens: 最大输出 token，默认 8192（内部有 2048 下限保护）。

    Returns:
        Gemini 生成的视频描述文本（末尾可能带用量统计或截断提示）。
    """
    err = _check_api_key()
    if err:
        return err
    if not url or not url.strip():
        return "没有提供链接。请把视频文件的 http/https 直链传给 url 参数。"

    temp_path, mime_type, err = await _download_to_temp(url.strip())
    if err:
        return err
    assert temp_path is not None and mime_type is not None

    try:
        video_bytes = await asyncio.to_thread(temp_path.read_bytes)
        if not video_bytes:
            return "下载到的文件是空的（0 字节），请确认这个直链有效。"
        return await _describe_video_bytes(
            video_bytes,
            mime_type,
            prompt=prompt,
            persona=persona,
            hint=hint,
            low_resolution=low_resolution,
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:  # noqa: BLE001 - 任何异常都不许裸抛出 MCP 边界
        logger.exception("describe_video_url 未预期异常")
        return f"发生了未预期的错误：{e}\n（如果反复出现，请把这条信息发给开发者。）"
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 工具 3：estimate_cost（费用预估）
# ---------------------------------------------------------------------------


def _probe_duration_blocking(path: str) -> float | None:
    """用 ffprobe 读视频时长（秒）；没有 ffprobe 或失败时返回 None。"""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw == "N/A":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fmt_tokens(n: float) -> str:
    """把 token 数格式化成“12,345（约 1.2 万）”这样的可读文本。"""
    n_int = int(round(n))
    if n_int >= 10000:
        return f"{n_int:,}（约 {n_int / 10000:.1f} 万）"
    return f"{n_int:,}"


@mcp.tool(annotations=READONLY, structured_output=False)
async def estimate_cost(path: str) -> str:
    """估算把某个本地视频交给 Gemini 识别大概要花多少输入 token，让你发大视频前心里有数。

    优先用 ffprobe 读真实时长；读不到（没装 ffprobe / 格式怪）则按文件大小粗估并注明。

    Args:
        path: 本地视频文件路径。

    Returns:
        一段中文预估说明（文件大小、时长、标清/低清 token 估算、上传通道）。
    """
    p, err = _validate_local_file(path)
    if err:
        return err
    assert p is not None

    # 顺带校验格式，让用户尽早发现“这个格式不支持”。
    try:
        mime_type = guess_mime_type(str(p))
    except GeminiVideoError as e:
        return str(e)

    # 兜底：任何未预期异常都不许裸抛出 MCP 边界（与 describe_video 一致）。
    try:
        size_bytes = p.stat().st_size
        size_mb = size_bytes / 1024 / 1024

        duration = await asyncio.to_thread(_probe_duration_blocking, str(p))
        if duration is not None and duration > 0:
            duration_note = f"时长：约 {duration:.1f} 秒（{duration / 60:.1f} 分钟，ffprobe 实测）"
            est_source = "实测时长"
        else:
            duration = max(size_bytes / _ASSUMED_BYTES_PER_SEC, 0.1)
            duration_note = (
                f"时长：约 {duration:.1f} 秒（{duration / 60:.1f} 分钟，"
                "⚠️ 未能读到真实时长，按 ~1.5Mbps 码率由文件大小粗估，仅供参考）"
            )
            est_source = "粗估时长"

        tokens_standard = duration * _TOKENS_PER_SEC_STANDARD
        tokens_low = duration * _TOKENS_PER_SEC_LOW

        is_gif = mime_type == "image/gif"
        channel = "inline 内联（单次请求）" if size_bytes <= VIDEO_INLINE_LIMIT_BYTES else "Files API（先上传再识别）"

        lines = [
            f"视频费用预估：{p.name}",
            f"文件大小：{size_mb:.1f} MB（{size_bytes:,} 字节）",
            duration_note,
            f"上传通道：{channel}（内联阈值 14MB）",
            "",
            f"输入 token 估算（基于{est_source}）：",
            f"  - 标清（约 300 token/秒）：{_fmt_tokens(tokens_standard)}",
            f"  - 低清（约 100 token/秒，low_resolution=True）：{_fmt_tokens(tokens_low)}",
            "",
            "说明：以上是【输入】token 估算，不含模型输出 token（取决于描述长短）。",
            "参考：标清约 300 token/秒，1 分钟视频约 1.8 万输入 token。长视频建议开 low_resolution 省钱。",
        ]
        if is_gif:
            lines.append("注：这是 GIF，以原格式直传、通常没有音轨，token 估算仅作上限参考。")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001 - 兜底：任何异常都不许裸抛出 MCP 边界
        logger.exception("estimate_cost 未预期异常")
        return f"预估费用时发生了未预期的错误：{e}\n（如果反复出现，请把这条信息发给开发者。）"


# ---------------------------------------------------------------------------
# 工具 4：view_media（把图片/视频某一帧作为【图片内容】投给调用方模型亲眼看）
# ---------------------------------------------------------------------------
# 与 describe_video 的分工：describe_video 让 Gemini 看视频写【文字】；
# view_media 是把一张图（图片本身，或视频抽的一帧）直接以 MCP 图片内容返回，
# 让【调用方】的模型（claude 等）自己看到画面。stdio / HTTP 两种模式都可用。


def _probe_dimensions_blocking(path: str) -> tuple[int, int] | None:
    """用 ffprobe 读第一路视频/图片流的宽高；没有 ffprobe 或读不到时返回 None。"""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", result.stdout.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _ffmpeg_frame_png(path: str, seek: float | None, max_dim: int) -> tuple[bytes | None, str | None]:
    """用 ffmpeg 抽一帧、按需等比缩小，PNG 字节经 stdout 返回。阻塞函数，调用方丢线程池。

    Returns (png_bytes, None) 或 (None, 中文错误)。
    - seek=None 表示不定位（图片输入 / 从头）；否则 -ss 快速定位到该秒（放在 -i 前，seek 更快）。
    - scale filter：长边缩到 max_dim 以内，等比、只缩不放（force_original_aspect_ratio=decrease）。
      单引号保护 min(a,b) 里的逗号，避免被 ffmpeg 滤镜图解析成分隔符。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "本机没有找到 ffmpeg，无法抽帧/缩放。请先安装 ffmpeg（本项目文档假设已装 ffmpeg 8.0）后重试。"
    scale = f"scale=w='min({max_dim},iw)':h='min({max_dim},ih)':force_original_aspect_ratio=decrease"
    cmd = [ffmpeg, "-nostdin", "-v", "error"]
    if seek is not None and seek > 0:
        cmd += ["-ss", f"{seek:.3f}"]
    cmd += ["-i", path, "-frames:v", "1", "-vf", scale, "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"调用 ffmpeg 抽帧失败：{e}"
    if result.returncode != 0 or not result.stdout:
        errmsg = (result.stderr or b"").decode("utf-8", "replace").strip()[:300]
        return None, f"ffmpeg 没能抽出这一帧（可能时间点超出时长或文件损坏）。\nffmpeg 说明：{errmsg or '无'}"
    return result.stdout, None


@mcp.tool(structured_output=False)
async def view_media(path: str, timestamp: float | None = None, max_dimension: int = 1024) -> Image | str:
    """把一张【图片】、或【视频的某一帧】作为图片内容直接返回，让调用方模型亲眼看到画面。

    - path 是图片（png/jpg/jpeg/webp/gif）：直接返回该图（长边超出 max_dimension 会等比缩小）。
      GIF 只返回首帧（想感知整段动画请用 describe_video）。
    - path 是视频：给了 timestamp（秒）就抽那一帧；没给则抽正中间那一帧。需要本机有 ffmpeg。

    与 describe_video 的分工：describe_video 让 Gemini 看视频写【文字】；view_media 把【画面本身】投给
    你（调用方模型）看。stdio / HTTP 模式都可用。

    Args:
        path: 本地图片或视频文件路径。
        timestamp: 仅对视频有效，抽取该秒（float）的一帧；不填则取中间帧。
        max_dimension: 返回图片的长边上限像素，默认 1024（超出等比缩小，不放大；范围 16~4096）。

    Returns:
        一张图片内容（MCP ImageContent）；出错时返回一句中文说明。
    """
    p, err = _validate_local_file(path)
    if err:
        return err
    assert p is not None

    # 归一化 max_dimension 到合理区间
    try:
        max_dim = int(max_dimension)
    except (TypeError, ValueError):
        max_dim = 1024
    max_dim = max(_VIEW_MIN_DIMENSION, min(max_dim, _VIEW_MAX_DIMENSION))

    suffix = p.suffix.lower()
    try:
        if suffix == ".gif":
            # GIF：抽首帧（ss=0）并按需缩放，转成静态 PNG 返回。
            data, ferr = await asyncio.to_thread(_ffmpeg_frame_png, str(p), 0.0, max_dim)
            if ferr:
                return ferr
            return Image(data=data, format="png")

        if suffix in _VIEW_IMAGE_EXTS:
            # 静态图片：够小就原样返回（不重编码）；长边超过 max_dimension 才用 ffmpeg 等比缩小。
            dims = await asyncio.to_thread(_probe_dimensions_blocking, str(p))
            if dims is None or max(dims) <= max_dim:
                return Image(path=str(p))
            data, ferr = await asyncio.to_thread(_ffmpeg_frame_png, str(p), None, max_dim)
            if ferr:
                # 缩放失败也别硬撑：退回原图，至少让调用方能看到画面。
                logger.warning("view_media 缩放失败，退回原图：%s", ferr)
                return Image(path=str(p))
            return Image(data=data, format="png")

        if suffix in _VIEW_VIDEO_EXTS:
            seek = timestamp
            if seek is None:
                # 没给时间点 → 取中间帧（时长用现有 ffprobe 逻辑；读不到则退回第 0 秒）。
                duration = await asyncio.to_thread(_probe_duration_blocking, str(p))
                seek = (duration / 2.0) if (duration and duration > 0) else 0.0
            else:
                try:
                    seek = float(seek)
                except (TypeError, ValueError):
                    return "timestamp 需要是一个秒数（数字），比如 12.5。"
                if seek < 0:
                    return "timestamp 不能是负数。"
            data, ferr = await asyncio.to_thread(_ffmpeg_frame_png, str(p), seek, max_dim)
            if ferr:
                return ferr
            return Image(data=data, format="png")

        return (
            f"view_media 不认识这个格式：{suffix or '（无后缀）'}。\n"
            f"图片支持 png/jpg/jpeg/webp/gif；视频支持 {', '.join(sorted(_VIEW_VIDEO_EXTS))}。"
        )
    except Exception as e:  # noqa: BLE001 - 任何异常都不许裸抛出 MCP 边界
        logger.exception("view_media 未预期异常")
        return f"处理这个文件时出错了：{e}\n（如果反复出现，请把这条信息发给开发者。）"


# ---------------------------------------------------------------------------
# 工具 5：get_upload_url（把上传地址告诉调用方模型，供 claude.ai 沙盒中转文件）
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READONLY, structured_output=False)
async def get_upload_url() -> str:
    """获取"把文件推到本服务器"的上传地址——用于把 claude.ai 聊天里上传/沙盒里生成的文件搬到服务器再识别。

    典型流程（在 claude.ai 的代码沙盒里执行）：
        1. 调本工具拿到上传地址；
        2. requests.post(上传地址, files={"file": open("/mnt/user-data/uploads/xxx.mp4", "rb")})；
        3. 用响应 JSON 里的 saved_path 调 describe_video 或 view_media。

    Returns:
        上传地址与用法说明；本地 stdio 模式下没有上传端点，会返回相应提示。
    """
    if not _HTTP_MODE_ACTIVE:
        return (
            "当前以本地 stdio 模式运行，没有上传端点——你和服务器在同一台电脑上，"
            "直接把本机文件路径传给 describe_video / view_media 即可，无需上传。"
        )
    secret = (GEMINI_MCP_HTTP_SECRET or "").strip()
    base = GEMINI_MCP_PUBLIC_BASE_URL or f"http://localhost:{_HTTP_PORT}"
    url = f"{base}/upload/{secret}"
    note = (
        ""
        if GEMINI_MCP_PUBLIC_BASE_URL
        else "\n（注意：未配置公网地址 GEMINI_MCP_PUBLIC_BASE_URL，此地址仅本机可达。）"
    )
    return (
        f"上传地址（POST，multipart 表单，字段名 file，单文件上限 500MB）：\n{url}\n"
        '用法示例：requests.post(url, files={"file": open(文件路径, "rb")})\n'
        "响应 JSON 的 saved_path 即可直接传给 describe_video / view_media。" + note
    )


# ---------------------------------------------------------------------------
# HTTP 远程模式（仅 `python main.py --http` 时走这里；stdio 默认路径完全不碰下面这些）
# ---------------------------------------------------------------------------
# 绑定地址/端口：0.0.0.0 让局域网 / 反向代理（Cloudflare Tunnel 等）能连上；
# 8768 是本 MCP 园区分配给 Gemini-Video 的专用端口（8080=Grok、8767=ASCII、8090=voice、3456=Memory、8001=bot）。
_HTTP_HOST = "0.0.0.0"
_HTTP_PORT = 8768

# 占位符集合：secret 等于其中任何一个（或为空）都视为"没真正设置"，拒绝以 --http 启动。
_PLACEHOLDER_HTTP_SECRETS = {
    "",
    "CHANGE_ME_GENERATE_A_RANDOM_SECRET",  # .env.example 里的占位值
    "change-me-to-a-long-random-string",
    "your-secret-here",
    "YOUR_HTTP_SECRET",
}


async def _handle_upload(request):  # noqa: ANN001 - Starlette Request，类型延迟导入
    """multipart 上传【单个】文件到 temp_media，返回 saved_path。仅 --http 模式注册这个端点。

    安全：secret 已在【路由路径】层校验——路径不匹配（含错 secret）根本到不了这里，Starlette 直接 404；
    单文件上限 500MB；文件名安全化 + 时间戳前缀防覆盖；绝不接受调用方指定存储路径（防路径穿越）。
    """
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from starlette.responses import JSONResponse

    # Content-Length 预检（能早退就早退）——留一点 multipart 头部余量。
    clen_raw = request.headers.get("Content-Length")
    if clen_raw and clen_raw.isdigit() and int(clen_raw) > _REMOTE_FILE_MAX_BYTES + 1024 * 1024:
        return JSONResponse({"error": "文件太大，超过 500MB 上限。"}, status_code=413)

    try:
        form = await request.form()
    except Exception as e:  # noqa: BLE001 - 表单解析失败给可读错误，不裸抛
        return JSONResponse({"error": f"解析上传表单失败：{e}"}, status_code=400)

    upload = form.get("file")
    if not isinstance(upload, StarletteUploadFile):
        return JSONResponse(
            {
                "error": (
                    "请用 multipart 表单上传【单个】文件，字段名必须是 file。"
                    "例如 requests.post(url, files={'file': open('x.mp4','rb')})。"
                )
            },
            status_code=400,
        )

    try:
        if upload.size is not None and upload.size > _REMOTE_FILE_MAX_BYTES:
            return JSONResponse({"error": "文件太大，超过 500MB 上限。"}, status_code=413)

        await asyncio.to_thread(_enforce_temp_media_quota)
        _ensure_temp_media_dir()
        try:
            dest = _resolve_in_temp_media(_safe_temp_name(upload.filename or "upload"))
        except GeminiVideoError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        written = 0
        file_handle = await asyncio.to_thread(open, dest, "wb")
        try:
            await upload.seek(0)
            while True:
                chunk = await upload.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > _REMOTE_FILE_MAX_BYTES:
                    await asyncio.to_thread(file_handle.close)
                    try:
                        dest.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return JSONResponse({"error": "文件太大，超过 500MB 上限。"}, status_code=413)
                await asyncio.to_thread(file_handle.write, chunk)
        finally:
            try:
                await asyncio.to_thread(file_handle.close)
            except OSError:
                pass

        if written == 0:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return JSONResponse({"error": "上传的文件是空的（0 字节）。"}, status_code=400)

        return JSONResponse(
            {
                "saved_path": str(dest),
                "size_mb": round(written / 1024 / 1024, 2),
                "hint": "把 saved_path 传给 describe_video（或 view_media）即可。",
            }
        )
    except Exception as e:  # noqa: BLE001 - 兜底：绝不让异常裸抛出 HTTP 边界
        logger.exception("上传端点未预期异常")
        return JSONResponse({"error": f"保存上传文件时出错：{e}"}, status_code=500)
    finally:
        try:
            await upload.close()
        except Exception:  # noqa: BLE001 - 关闭失败无所谓
            pass


def _run_http() -> None:
    """以 Streamable HTTP 传输启动，路径 /mcp/<secret>，供 claude.ai / 手机远程连接。

    这个端口通常经 Cloudflare Tunnel 之类暴露到公网，因此：
      1) secret 缺失或仍是占位符 → 直接拒绝启动并打印中文说明（不允许裸奔）；
      2) 关闭 DNS rebinding 保护 → 让 cloudflare 域名 / 局域网 IP 带来的非 localhost Host 头能通过
         （参考园区 Grok-MCP 的同款处理；secret 路径本身即访问门锁）；
      3) 额外注册 POST /upload/<secret> 上传端点（复用同一个 secret；stdio 模式不注册）。
    """
    # 关闭 DNS rebinding 保护所需的设置类（延迟导入，stdio 路径用不到）。
    from mcp.server.fastmcp.server import TransportSecuritySettings

    global _HTTP_MODE_ACTIVE
    _HTTP_MODE_ACTIVE = True

    secret = (GEMINI_MCP_HTTP_SECRET or "").strip()

    # —— 安全闸：secret 没真正设置就拒绝启动 ——
    if secret in _PLACEHOLDER_HTTP_SECRETS:
        print(
            "[gemini-video-mcp] 拒绝以 --http 启动：未设置有效的 GEMINI_MCP_HTTP_SECRET。\n"
            "  这个端口会被拼进公网访问路径 /mcp/<secret>，是唯一门锁，绝不能留空或用占位符。\n"
            "  请这样做：\n"
            '    1) 生成一段随机口令（本目录终端）：uv run python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "    2) 把它写进本目录 .env 的 GEMINI_MCP_HTTP_SECRET=（可参考 .env.example）；\n"
            "       或用环境变量注入：set GEMINI_MCP_HTTP_SECRET=你生成的口令（Windows CMD）。\n"
            "  设置好后重新运行 start_http.bat（或 uv run python main.py --http）即可。",
            file=sys.stderr,
        )
        sys.exit(1)

    # —— 字符白名单：secret 会被拼进路由路径，`{}` 会被 Starlette 编译成路径参数（等于开了通配门），
    # `?`/`#` 等则会让路由永不匹配。只放行 URL 安全字符（token_urlsafe 的产出天然满足）。——
    if not re.fullmatch(r"[A-Za-z0-9._~-]+", secret):
        print(
            "[gemini-video-mcp] 拒绝以 --http 启动：GEMINI_MCP_HTTP_SECRET 含不安全字符。\n"
            "  secret 会拼进访问路径，只能使用字母、数字与 -._~ 这几种字符。\n"
            '  推荐直接用命令生成：uv run python -c "import secrets; print(secrets.token_urlsafe(32))"',
            file=sys.stderr,
        )
        sys.exit(1)

    # 把 HTTP 相关设置写进 FastMCP 的 settings（run_streamable_http_async / streamable_http_app
    # 都在调用时才读取这些字段，因此在 mcp.run() 之前赋值即可生效；stdio 模式从不走到这里）。
    mcp.settings.host = _HTTP_HOST
    mcp.settings.port = _HTTP_PORT
    mcp.settings.streamable_http_path = f"/mcp/{secret}"
    mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    # 注册上传端点（custom_route 会把 Route 追加进 _custom_starlette_routes，
    # streamable_http_app() 构建时会 extend 进最终路由——见 SDK server.py。必须在 mcp.run() 之前注册）。
    # 只在 --http 分支执行，stdio 模式从不注册这个端点。
    _enforce_temp_media_quota()  # 启动时先清一次 temp_media（尽力而为）
    mcp.custom_route(f"/upload/{secret}", methods=["POST"])(_handle_upload)

    # 日志只打印 secret 的前 6 位做识别，绝不整串泄露到终端 / 日志。
    masked = secret[:6] + "…"
    logger.info("Streamable HTTP 模式启动：http://%s:%s/mcp/%s", _HTTP_HOST, _HTTP_PORT, masked)
    logger.info("上传端点已启用：POST http://%s:%s/upload/%s（同一个 secret）", _HTTP_HOST, _HTTP_PORT, masked)
    logger.info(
        "（本机自测地址：http://localhost:%s/mcp/<你的secret>；公网请经 Cloudflare Tunnel 暴露，见 README）", _HTTP_PORT
    )

    # —— 关闭 uvicorn 的 access 日志：它默认会把每个请求的完整路径（含 secret）打进 stdout，
    # 黑窗口截图/共享屏幕就等于泄露门锁。上面那条打码日志已足够确认服务在跑。——
    from uvicorn.config import LOGGING_CONFIG

    try:
        LOGGING_CONFIG["loggers"]["uvicorn.access"]["handlers"] = []
    except (KeyError, TypeError):
        pass

    mcp.run(transport="streamable-http")


def main() -> None:
    """入口：默认以 stdio 传输启动；带 --http 参数时改为 Streamable HTTP。

    不带任何参数时行为与历史版本完全一致（stdio），Claude Code / Desktop 本地注册不受影响。
    """
    if "--http" in sys.argv:
        _run_http()
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
