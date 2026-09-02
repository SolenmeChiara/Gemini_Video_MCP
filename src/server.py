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
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from .config import (
    GEMINI_AGENTIC_MODEL,
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

# ---------------------------------------------------------------------------
# 处理模式（static / agentic）与「细看」相关常量
# ---------------------------------------------------------------------------
# mode="auto" 的分水岭：视频时长 >= 5 分钟就交给 agentic（模型自行决定看哪段、什么帧率），
# 更短的片子官方建议仍走静态 1 FPS 全程抽帧——短片本来就不贵，agentic 反而可能漏掉细节。
_AGENTIC_AUTO_MIN_DURATION_SEC = 300.0

# 「细看」只给了 start 没给 end 时，默认往后看多少秒。
_DETAIL_DEFAULT_WINDOW_SEC = 30.0
# 「细看」抽帧率上限（再高 token 涨得离谱，实际收益有限）。
_DETAIL_MAX_FPS = 10.0

# ---------------------------------------------------------------------------
# Files API 上传缓存（同一个视频第二次识别时不再重传）
# ---------------------------------------------------------------------------
# 缓存索引文件（放在 temp_media/ 里，随目录一起被 gitignore）。
_UPLOAD_CACHE_NAME = "upload_cache.json"
# 缓存条目数上限与远端总字节上限（Files API 免费层存储 2GB，留出余量）。
_UPLOAD_CACHE_MAX_ENTRIES = 30
_UPLOAD_CACHE_MAX_BYTES = 1536 * 1024 * 1024
# 过期前多久就当它失效（避免刚好卡在过期边缘发请求）。
_UPLOAD_CACHE_EXPIRY_MARGIN_SEC = 300.0
# 读改写缓存文件的互斥锁（本进程内单事件循环，够用）。
_UPLOAD_CACHE_LOCK = asyncio.Lock()

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
# 文风上有意【不】压制（不禁止抒情、不要求客观简洁），保留 Gemini 自由发挥的戏剧张力；
# 但另立四条硬规则（_DEFAULT_PROMPT_RULES）兜住实测踩到的坑：
#   ① 知识边界——Gemini 的训练知识截止较早，实测会把 2026 年新发布的官方内容一口咬定成“同人/概念 Mod”，
#      故明确禁止仅凭“训练数据里没有”就对真伪、出处、是否官方下断言；
#   ② 零翻译——视频里的台词/字幕/歌词一律原文照录，不译成中文（sol 明确要求），描述性行文仍用中文；
#   ③ 降解读、升描述——少讲文字“意味着什么”，多讲它长什么样、怎么出现、与画面声音怎么配合；
#   ④ 禁止静默省略——实测 3.1 Pro 会把看不清、拿不准的元素直接跳过不提（省略无需标注、零风险），
#      故要求不确定内容必须带不确定度写出（疑似 X / 无法辨认 + 物理特征），不许从描述里消失。
# 原文两处重复的编号“6.”这里顺手改成 6 / 7；两行人设占位符改由 persona 参数按需插入。

_DEFAULT_PROMPT_HEAD = (
    '请观看这个视频，并将其"翻译"成文字，让它成为一个就连只支持文字的llm也能'
    "体会到视频内容的细致，生动，令人神往或感同身受的描述。"
    "（这里的「翻译」指的是模态转换——把画面、声音、文字统统落成文本；"
    "它【不】是语言之间的翻译，视频里说的、唱的、写的，一律照原文抄录，不要译成中文。）"
)

_DEFAULT_PROMPT_BODY = """请提供详细的视频内容讲述，提供详细的，精细到时间戳的描述，涵盖以下方面：
1. 视频的整体内容和主题，风格如何？分类是什么？例如，是艺术作品、meme分享，抑或是随手拍片？（这里只判断形态与用途，不要推断它出自哪部作品、是否官方）
2. 对主要人物、对象和场景的详细叙述；台词与旁白一律原文逐字转录，并交代是谁在什么画面下说的、语气与音色如何
3. 如有，也带上动作、情节和时间线发展，以及最高光的时刻
4. 视频的视觉风格和艺术特点是什么？画面质量如何？流畅还是卡顿？给人一种什么样的视觉感受？
5. 整体氛围和情感表达了什么？
6. 是否有背景音乐/音效、背景音？如有，是什么样的感觉？是什么风格的？是否具有音乐卡点？音乐在此处起到了什么作用（反差？讽刺？增强气氛？），给人什么样的听觉和感官体验？如有歌词，逐字照录原文，听不清的地方标注「听不清」，不要脑补，也不要转述大意
7. 任何特殊的视觉效果或文字内容。屏幕上出现的一切文字（字幕、标题、UI、弹幕、涂鸦、背景招牌）都请原文逐字转录，并描述它的呈现形式：字体、字号、颜色、在画面里的位置、停留多久、以什么方式出现和消失（淡入、打字机、闪现、被划掉），以及它出现在哪一个镜头、和哪个动作或哪一声音效同步"""

_DEFAULT_PROMPT_RULES = """以下四条是硬规则，优先于上面的任何要求：

一、关于你不认识的内容：请假定你的训练知识截止约在 2026 年 1 月，而视频里完全可能出现比这更晚发布的官方作品、正式发行的新内容、新角色、新版本。因此「我的训练数据里没有」绝不等于「这是同人、二创、Mod、概念演示、恶搞或假货」。禁止仅凭眼生就对内容的真伪、出处、是否官方下任何断言。遇到认不出的东西，照实描述你看到和听到的即可，需要时直接写「无法确认出处」。

二、原文照录，不做翻译：视频里的字幕、对白、旁白、歌词、屏幕上的一切文字，一律按原语言逐字转录，一个字也不要译成中文，也不要中英混写地转述大意。看不清或听不清的地方，标注「此处看不清 / 听不清」，不要脑补补全。你自己的描述性行文用中文，被引用的原文保持原样。

三、多描述，少解读：把力气花在「呈现出了什么」上，而不是「这意味着什么」。文字和语言内容尤其如此——不必分析台词或歌词的主题、寓意、潜台词、文化背景，也不用总结它想表达什么；请改为描述它长什么样、怎么出现、和画面与声音怎么配合。氛围与感受可以照常写，那是你亲眼所见的质地，不是文本分析。

四、不确定不等于不存在，禁止静默省略：画面或声音里任何你看不清、听不辨、认不出、拿不准的元素——模糊的背景、一闪而过的物体、低画质下的小字、嘈杂环境里的人声——一律不许因为「不确定」就跳过不写。正确做法是把它连同你的不确定度一起写出来：能给出最可能的猜测就写「疑似 X／像是 X」，猜不出是什么就描述你确实看到的物理特征（轮廓、颜色、位置、出现时刻与时长）再注明「无法辨认」。这与第二条的「不要脑补」并不冲突：脑补是把猜测当事实写，这里要求的是把猜测标成猜测写出来。读这段描述的一方看不到视频，你没写的东西对它就等于从不存在——取舍确定性是读者的事，你的职责是不让任何出现过的东西从描述里消失。

请用中文书写描述（引用的原文除外），结果要详细准确。"""


def _build_default_prompt(persona: str | None, hint: str | None = None) -> str:
    """拼出默认提示词；传了 persona 就在开头位置附体一行人设，传了 hint 就注入人类观看者的前置线索。

    末尾固定追加四条硬规则（知识边界 / 原文照录不翻译 / 多描述少解读 / 禁止静默省略），
    放在最后是为了让它们压过前面的描述清单与人设——实测里模型更容易服从最后读到的约束。
    """
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
    segments.append(_DEFAULT_PROMPT_RULES)
    # 用空行分隔各段，还原原版排版观感。
    return "\n\n".join(segments)


# ---------------------------------------------------------------------------
# 「细看」提示词（只看某一段时用；与上面的默认模板并列，不覆盖它）
# ---------------------------------------------------------------------------
# 传了 start/end 就说明人已经看过一遍、想回头盯住某几秒，所以这版模板不再要求通篇概览：
#   - hint 像个问句 -> 就这一段回答这个问题；
#   - 否则 -> 按秒级时间戳把这一段的动作、声音、字幕拆开写。
# 复用默认模板的四条硬规则（知识边界 / 原文照录不翻译 / 多描述少解读 / 禁止静默省略）。

_DETAIL_PROMPT_HEAD = (
    "你现在只会看到这个视频的其中一小段（原视频的第 {start} 到第 {end} 秒），"
    "而不是全片。请把注意力全部放在这一段上，不要概述全片，也不要谈论这一段之外发生了什么。"
    "回答里出现时间戳时，请用【原视频的绝对时间】（也就是从第 {start} 秒往后数），精确到秒。"
)

_DETAIL_PROMPT_QUESTION = """人类观看者看过整段视频，现在专门回放这几秒，想弄清楚一件事：

「{question}」

请先直接回答这个问题，答完再补上这一段里与之相关的细节：谁在什么时刻做了什么动作、身体和视线朝向哪里、
物件怎么移动、画面怎么切换、声音里有什么（说话请逐字照录，也包括笑声、音效、动物叫声、背景音）、
屏幕上出现了什么文字。看不清、听不清的地方照实说不确定，不要为了把问题答圆而虚构细节。"""

_DETAIL_PROMPT_DESCRIBE = """请按秒级时间戳把这一段拆开细讲，每一个明显的动作或声音变化都单独占一行，格式形如
「00:41 ——」。每一行都要交代清楚：

1. 人物/动物的动作：谁、用哪只手/哪个部位、朝哪个方向、快还是慢、动作之间怎么衔接；
2. 声音：说话内容逐字照录（连语气词、气声、含糊的字都尽量还原，听不清就标「听不清」），
   以及笑声、呼吸、动物叫声、碰撞声、音效、背景音乐的变化；
3. 屏幕上的文字：字幕、贴纸、UI，逐字照录，并说明它出现和消失的时刻；
4. 画面本身：镜头有没有推拉摇晃、有没有剪辑点、光线和构图的变化。"""


# agentic 模式专用的提示词尾巴：Interactions 会把模型的"检索计划/自言自语"一并写进 model_output，
# 实测（gemini-3.8-flash）正文开头会出现"现在对照用户的所有要求：1. …"这类工作笔记。这里直接要求它别写。
_AGENTIC_PROMPT_SUFFIX = (
    "\n\n（这一次你可以自己决定回看视频的哪几段、用什么帧率、要不要调取音轨。"
    "但最终请【只】输出给读者看的描述正文——不要写检索计划，不要复述上面的要求清单，"
    "也不要保留「让我再听一遍这段」这类过程中的自言自语。）"
)


def _build_detail_prompt(persona: str | None, hint: str | None, start: float, end: float) -> str:
    """拼出「细看」提示词：hint 像问句就针对这段答题，否则按秒级时间戳细描这一段。"""
    segments = [_DETAIL_PROMPT_HEAD.format(start=f"{start:g}", end=f"{end:g}")]
    if persona:
        segments.append(f"你的人设是：{persona.strip()}")

    question = (hint or "").strip()
    if question and _looks_like_question(question):
        segments.append(_DETAIL_PROMPT_QUESTION.format(question=question))
    else:
        if question:
            segments.append(
                f"观看前的已知信息：人类观看者对这个视频的形容是：「{question}」。"
                "这只是理解画面的线索，请以你实际看到、听到的为准，不要为了迎合它虚构细节。"
            )
        segments.append(_DETAIL_PROMPT_DESCRIBE)

    segments.append(_DEFAULT_PROMPT_RULES)
    return "\n\n".join(segments)


# 疑问句判定用的词表。中文没有词边界，按子串命中即可；英文必须整词命中，
# 且 is/are/do/did/can 这类助动词只在【句首】才算提问——否则 "this is a cat video"、
# "a clip of his dog" 这类陈述句会被误判成问题，细看模板就跑去"回答"一句描述。
_QUESTION_WORDS_CJK = (
    "吗",
    "呢",
    "什么",
    "怎么",
    "怎样",
    "如何",
    "为什么",
    "为何",
    "哪",
    "谁",
    "多少",
    "几时",
    "是不是",
    "有没有",
    "是否",
)
# 疑问代词/副词：出现在句子任何位置都算提问。
_QUESTION_WH_RE = re.compile(r"\b(what|why|how|who|where|when|which|whether)\b")
# 助动词：只有开头才算提问（一般疑问句的语序特征）。
_QUESTION_AUX_RE = re.compile(r"^(did|does|do|is|are|can)\b")


def _looks_like_question(text: str) -> bool:
    """判断 hint 是不是一个问题：以问号收尾，或含疑问词（英文助动词只认句首）。"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith(("？", "?")):
        return True
    lowered = stripped.lower()
    if any(word in lowered for word in _QUESTION_WORDS_CJK):
        return True
    return bool(_QUESTION_WH_RE.search(lowered) or _QUESTION_AUX_RE.match(lowered))


# ---------------------------------------------------------------------------
# 处理模式与「细看」参数的解析
# ---------------------------------------------------------------------------


def _parse_time_value(value: float | str | None) -> tuple[float | None, str | None]:
    """把时间参数解析成秒；接受数字（秒）或 "m:ss" / "h:mm:ss" 写法。

    Returns:
        (秒数, None) 或 (None, 中文错误)。传 None 返回 (None, None)。
    """
    if value is None:
        return None, None
    if isinstance(value, bool):  # bool 是 int 的子类，先挡掉
        return None, f"时间参数写法看不懂：{value!r}。请给秒数（如 39）或 \"分:秒\"（如 1:06）。"
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raw = str(value).strip()
        if not raw:
            return None, None
        if ":" in raw:
            parts = raw.split(":")
            if len(parts) > 3:
                return None, f"时间写法看不懂：{raw}。最多支持 时:分:秒（如 1:02:03）。"
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                return None, f"时间写法看不懂：{raw}。请给秒数（如 39）或 \"分:秒\"（如 1:06）。"
            seconds = 0.0
            for num in nums:
                seconds = seconds * 60 + num
        else:
            try:
                seconds = float(raw)
            except ValueError:
                return None, f"时间写法看不懂：{raw}。请给秒数（如 39）或 \"分:秒\"（如 1:06）。"
    if seconds < 0:
        return None, f"时间不能是负数：{value!r}。"
    return seconds, None


def _resolve_mode(mode: str | None, duration: float | None) -> tuple[str, str | None]:
    """决定这次走 static 还是 agentic。

    Returns:
        (模式, 给页脚的说明或 None)。auto 时按时长分流；读不到时长一律 static。
    """
    raw = (mode or "auto").strip().lower()
    if raw not in ("auto", "static", "agentic"):
        return "static", f"mode 值「{mode}」无法识别（可用：auto / static / agentic），已按 static 处理。"
    if raw != "auto":
        return raw, None
    if duration and duration >= _AGENTIC_AUTO_MIN_DURATION_SEC:
        return "agentic", None
    return "static", None


def _resolve_detail_range(
    start: float | str | None,
    end: float | str | None,
    fps: float | None,
    duration: float | None,
) -> tuple[tuple[float, float] | None, float | None, str | None]:
    """校验并归一「细看」参数。

    Returns:
        (区间 或 None, 归一后的 fps 或 None, 中文错误 或 None)。start/end 都没给就返回 (None, None, None)。
    """
    if start is None and end is None:
        if fps is not None:
            return None, None, "fps 只在「细看」时有意义，请同时给出 start（和 end），指明要盯住哪一段。"
        return None, None, None

    start_sec, err = _parse_time_value(start if start is not None else 0)
    if err:
        return None, None, f"start 参数有问题：{err}"
    end_sec, err = _parse_time_value(end)
    if err:
        return None, None, f"end 参数有问题：{err}"

    start_sec = start_sec or 0.0
    if end_sec is None:
        end_sec = start_sec + _DETAIL_DEFAULT_WINDOW_SEC
        if duration and duration > start_sec:
            end_sec = min(end_sec, duration)
    if duration and start_sec >= duration:
        return None, None, (
            f"start（{start_sec:g} 秒）已经超过视频总时长（约 {duration:.1f} 秒），没有可看的画面。"
        )
    if duration and end_sec > duration:
        end_sec = duration
    if end_sec <= start_sec:
        return None, None, f"end（{end_sec:g} 秒）必须大于 start（{start_sec:g} 秒）。"

    fps_value: float | None = None
    if fps is not None:
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            return None, None, f"fps 写法看不懂：{fps!r}，请给一个数字（如 5）。"
        if fps_value <= 0:
            return None, None, "fps 必须大于 0。"
        if fps_value > _DETAIL_MAX_FPS:
            return None, None, f"fps 最高 {_DETAIL_MAX_FPS:g}（你给的是 {fps_value:g}），再高 token 会涨得很离谱。"

    return (start_sec, end_sec), fps_value, None


def _build_video_metadata(start: float, end: float, fps: float | None) -> dict[str, object]:
    """拼出 videoMetadata（只看某一段 / 换抽帧率）。offset 用 proto Duration 的字符串写法。"""
    metadata: dict[str, object] = {"start_offset": f"{start:g}s", "end_offset": f"{end:g}s"}
    if fps:
        metadata["fps"] = fps
    return metadata


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
# Files API 上传缓存：同一个视频再看一次，不重传
# ---------------------------------------------------------------------------
# 键用【内容的 sha256】，不是 path+size+mtime。理由有三：
#   ① 共享识别管线 _describe_video_bytes 只拿得到字节，拿不到路径（直链下载那条入口本来就没有稳定路径）；
#   ② 内容寻址能跨改名/移动/重复下载复用同一份远端文件，path 方案换个目录就白传一次；
#   ③ 20MB 算一次 sha256 只要几十毫秒（还丢在线程池里），相对一次上传的几十秒完全可以忽略。
# 远端文件由 Google 侧 48 小时后自动清理，本地索引每次复用前都用 files.get 校验 state 与过期时间。


def _upload_cache_path() -> Path:
    return _TEMP_MEDIA_DIR / _UPLOAD_CACHE_NAME


def _load_upload_cache_blocking() -> dict[str, dict]:
    """读缓存索引；文件不存在/损坏一律当空（缓存丢了最多多传一次，不该报错）。"""
    path = _upload_cache_path()
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _save_upload_cache_blocking(cache: dict[str, dict]) -> None:
    """写缓存索引（尽力而为，写不进去只影响下次是否命中）。"""
    _ensure_temp_media_dir()
    try:
        _upload_cache_path().write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        logger.debug("写上传缓存索引失败（忽略）：%s", e)


def _parse_rfc3339(value: str | None) -> datetime | None:
    """解析 Files API 返回的 expirationTime（RFC3339）；解析不了返回 None。"""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip().replace("Z", "+00:00")
    # 秒的小数位超过 6 位时 fromisoformat 会报错，先截断到微秒。
    raw = re.sub(r"(\.\d{6})\d+", r"\1", raw)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _sha256_blocking(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_num(value: object) -> float:
    """读缓存条目里的数字字段：坏值一律当 0。

    索引文件是纯 JSON，手改坏一个 size / saved_at（改成字符串）就会让 float()/int() 抛 ValueError。
    那时上传已经花掉了，却在记账这一步炸掉整条识别管线，代价完全不对等，所以这里一律降级为 0。
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _evict_upload_cache(cache: dict[str, dict], protect_key: str | None = None) -> list[str]:
    """把缓存压回条目数/总字节上限内，返回被淘汰条目的远端文件名（供调用方尽力删除）。

    protect_key 是这次刚上传、马上就要用的那条，永远不淘汰——否则遇到一个比总上限还大的视频时，
    会把自己刚传上去的文件删掉，紧接着的识别请求就扑空了。
    """
    entries = sorted(cache.items(), key=lambda kv: _cache_num(kv[1].get("saved_at")))
    total = sum(int(_cache_num(entry.get("size"))) for _, entry in entries)
    dropped: list[str] = []
    for key, entry in entries:
        if len(cache) <= _UPLOAD_CACHE_MAX_ENTRIES and total <= _UPLOAD_CACHE_MAX_BYTES:
            break
        if key == protect_key:
            continue
        cache.pop(key, None)
        total -= int(_cache_num(entry.get("size")))
        name = entry.get("file_name")
        if isinstance(name, str) and name:
            dropped.append(name)
    return dropped


async def _get_or_upload_file(
    video_bytes: bytes, mime_type: str, client: GeminiVideoClient
) -> tuple[str, bool]:
    """拿到这段视频在 Files API 上的 file_uri：命中缓存就复用，否则上传并记进缓存。

    Returns:
        (file_uri, 是否缓存命中)

    Raises:
        GeminiVideoError: 上传失败时（缓存读写失败不会抛，最多退化成每次都传）。
    """
    digest = await asyncio.to_thread(_sha256_blocking, video_bytes)

    async with _UPLOAD_CACHE_LOCK:
        cache = await asyncio.to_thread(_load_upload_cache_blocking)
        entry = cache.get(digest)

    if entry:
        expire_at = _parse_rfc3339(entry.get("expire_time"))
        now = datetime.now(UTC)
        expired = bool(expire_at and (expire_at - now).total_seconds() <= _UPLOAD_CACHE_EXPIRY_MARGIN_SEC)
        info = None if expired else await client.get_file_info(str(entry.get("file_name") or ""))
        state = (info or {}).get("state")
        if info and state == "ACTIVE" and entry.get("file_uri"):
            logger.info("上传缓存命中：%s（%.1fMB，跳过重传）", entry.get("file_name"), len(video_bytes) / 1024 / 1024)
            return str(entry["file_uri"]), True
        logger.info(
            "上传缓存未命中（记录已失效：expired=%s，state=%s），准备重传：%s",
            expired,
            state,
            entry.get("file_name"),
        )
        async with _UPLOAD_CACHE_LOCK:
            cache = await asyncio.to_thread(_load_upload_cache_blocking)
            cache.pop(digest, None)
            await asyncio.to_thread(_save_upload_cache_blocking, cache)
    else:
        logger.info("上传缓存未命中（没有这份内容的记录），开始上传 %.1fMB", len(video_bytes) / 1024 / 1024)

    file_uri, file_name = await client.upload_video(video_bytes, mime_type)

    # 记账：顺带取一次过期时间（取不到就只靠 files.get 校验）。
    info = await client.get_file_info(file_name) or {}
    dropped: list[str] = []
    async with _UPLOAD_CACHE_LOCK:
        cache = await asyncio.to_thread(_load_upload_cache_blocking)
        cache[digest] = {
            "file_name": file_name,
            "file_uri": file_uri,
            "mime_type": mime_type,
            "size": len(video_bytes),
            "expire_time": info.get("expirationTime"),
            "saved_at": time.time(),
        }
        dropped = _evict_upload_cache(cache, protect_key=digest)
        await asyncio.to_thread(_save_upload_cache_blocking, cache)

    for name in dropped:
        logger.info("上传缓存超出上限，淘汰旧远端文件：%s", name)
        await client.delete_file(name)

    return file_uri, False


# ---------------------------------------------------------------------------
# 识别管线的共享核心（describe_video 与 describe_video_url 都复用这一段）
# ---------------------------------------------------------------------------


def _build_final_prompt_and_tokens(
    prompt: str | None,
    persona: str | None,
    hint: str | None,
    max_output_tokens: int,
    detail_range: tuple[float, float] | None = None,
) -> tuple[str, int]:
    """把 prompt/persona/hint 组装成最终提示词，并把 max_output_tokens 归一化到下限之上。

    本地文件、直链下载、YouTube 直读三条入口都共用这一段，确保提示词组装（prompt 优先，
    其次「细看」模板或 persona/hint 版默认模板）与输出 token 下限保护完全一致、不漂移。
    """
    # 组装提示词：自定义 prompt 优先；其次「细看」模板（给了 start/end 时）；最后默认模板。
    if prompt and prompt.strip():
        final_prompt = prompt.strip()
    elif detail_range:
        final_prompt = _build_detail_prompt(persona, hint, detail_range[0], detail_range[1])
    else:
        final_prompt = _build_default_prompt(persona, hint)

    # 输出 token 下限保护（思考 token 也计入此上限，太小会把正文挤没）
    try:
        tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        tokens = 4096
    effective_tokens = max(tokens, _MIN_OUTPUT_TOKENS)
    return final_prompt, effective_tokens


def _format_describe_result(result: dict, processing: str = "static", notes: list[str] | None = None) -> str:
    """把 Gemini 识别结果 dict 拼成最终返回文本（正文 + 末尾用量/截断/处理方式页脚）。

    本地文件、直链下载、YouTube 直读三条入口都共用这一段，保证页脚格式与措辞一致。
    processing 是这次实际走的处理方式说明（static / agentic / 细看区间等），会写进用量那一行。
    notes 里放回退原因之类需要让用户看见的提醒。
    """
    text = result["text"]
    footer_lines: list[str] = []
    if result.get("truncated"):
        footer_lines.append(
            "⚠️ 提示：描述可能被输出长度上限截断了。可调大 max_output_tokens（如 8192）后重试以获得完整内容。"
        )
    for note in notes or []:
        if note:
            footer_lines.append(f"ℹ️ {note}")
    usage = result.get("usage")
    if usage:
        footer_lines.append(
            f"（用量：输入 {usage['prompt_tokens']:,} token，输出 {usage['output_tokens']:,} token，"
            f"合计 {usage['total_tokens']:,} token；通道：{result.get('channel')}；处理：{processing}）"
        )
    else:
        footer_lines.append(f"（通道：{result.get('channel')}；处理：{processing}；本次没拿到用量统计）")
    if footer_lines:
        text = text + "\n\n---\n" + "\n".join(footer_lines)
    return text


def _agentic_model() -> str:
    """agentic 路径实际用的模型：配了 GEMINI_AGENTIC_MODEL 就用它，否则沿用主模型。"""
    return GEMINI_AGENTIC_MODEL or GEMINI_MODEL


async def _run_static_generate(
    client: GeminiVideoClient,
    *,
    video_bytes: bytes,
    mime_type: str,
    final_prompt: str,
    effective_tokens: int,
    low_resolution: bool,
    file_uri: str | None,
    video_metadata: dict | None,
) -> dict:
    """静态通道（generateContent）：思考等级不被支持时自动降级 low 重试一次。"""
    try:
        return await client.describe_video(
            video_bytes=video_bytes,
            prompt=final_prompt,
            mime_type=mime_type,
            max_output_tokens=effective_tokens,
            low_resolution=low_resolution,
            thinking_level=GEMINI_THINKING_LEVEL,
            file_uri=file_uri,
            video_metadata=video_metadata,
        )
    except GeminiVideoError as e:
        # 个别模型不支持当前思考等级（会 400），自动降级为 low 重试一次
        if "thinking level" not in str(e).lower():
            raise
        logger.info("模型 %s 不支持 thinking_level=%s，自动改用 low 重试", GEMINI_MODEL, GEMINI_THINKING_LEVEL)
        return await client.describe_video(
            video_bytes=video_bytes,
            prompt=final_prompt,
            mime_type=mime_type,
            max_output_tokens=effective_tokens,
            low_resolution=low_resolution,
            thinking_level="low",
            file_uri=file_uri,
            video_metadata=video_metadata,
        )


async def _describe_video_bytes(
    video_bytes: bytes,
    mime_type: str,
    *,
    prompt: str | None,
    persona: str | None,
    hint: str | None,
    low_resolution: bool,
    max_output_tokens: int,
    mode: str = "static",
    detail_range: tuple[float, float] | None = None,
    fps: float | None = None,
    mode_note: str | None = None,
) -> str:
    """识别管线的共享核心：拿到【视频字节 + MIME】后，组装提示词、调用 Gemini、拼装中文返回文本。

    describe_video（本地文件）与 describe_video_url（直链下载）都复用这一段，确保两条入口的
    提示词组装、思考等级自动降级重试、用量/截断页脚逻辑完全一致、不漂移。
    调用方需自行保证：API key 已存在、video_bytes 非空、mime_type 已判定、mode 已按时长解析完。

    - mode="static"：generateContent 全程按固定帧率抽帧（原有行为）；
    - mode="agentic"：Interactions API，模型自己决定看哪段、什么帧率（需要先上传到 Files API）；
      跑不通时自动回退 static，并把原因写进页脚。
    - detail_range/fps：只看某一段（「细看」），走 videoMetadata，恒定 static。
    """
    final_prompt, effective_tokens = _build_final_prompt_and_tokens(
        prompt, persona, hint, max_output_tokens, detail_range
    )
    video_metadata = _build_video_metadata(detail_range[0], detail_range[1], fps) if detail_range else None
    notes: list[str] = [mode_note] if mode_note else []

    client = GeminiVideoClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL, base_url=GEMINI_BASE_URL)
    result: dict | None = None
    processing = "static"

    try:
        # 需要远端文件的两种情形：agentic（Interactions 只吃 uri）、以及超过内联上限的大文件。
        file_uri: str | None = None
        if mode == "agentic" or len(video_bytes) > VIDEO_INLINE_LIMIT_BYTES:
            try:
                file_uri, _cache_hit = await _get_or_upload_file(video_bytes, mime_type, client)
            except GeminiVideoError as e:
                if mode != "agentic":
                    raise
                notes.append(f"agentic 需要先把视频传到 Files API，这一步失败了（{e}），已改用 static 处理。")
                mode = "static"
                file_uri = None

        if mode == "agentic" and file_uri:
            try:
                result = await client.describe_video_agentic(
                    prompt=final_prompt + _AGENTIC_PROMPT_SUFFIX,
                    file_uri=file_uri,
                    mime_type=mime_type,
                    model=_agentic_model(),
                )
                processing = f"agentic（模型 {result.get('model')}）"
            except GeminiVideoError as e:
                notes.append(f"agentic 模式没能用上（{e}），已自动回退 static 全程抽帧。")
                mode = "static"

        if result is None:
            result = await _run_static_generate(
                client,
                video_bytes=video_bytes,
                mime_type=mime_type,
                final_prompt=final_prompt,
                effective_tokens=effective_tokens,
                low_resolution=low_resolution,
                file_uri=file_uri,
                video_metadata=video_metadata,
            )
            processing = "static"
            if detail_range:
                fps_note = f"，fps={fps:g}" if fps else ""
                processing = f"static（细看 {detail_range[0]:g}s–{detail_range[1]:g}s{fps_note}）"
    except GeminiVideoError as e:
        return f"视频识别失败：{e}"
    except Exception as e:  # noqa: BLE001 - 兜底：任何异常都不许裸抛出 MCP 边界
        logger.exception("视频识别管线未预期异常")
        return f"发生了未预期的错误：{e}\n（如果反复出现，请把这条信息发给开发者。）"

    return _format_describe_result(result, processing, notes)


async def _describe_youtube_url(
    youtube_url: str,
    *,
    prompt: str | None,
    persona: str | None,
    hint: str | None,
    low_resolution: bool,
    max_output_tokens: int,
    mode: str = "static",
    mode_note: str | None = None,
) -> str:
    """YouTube 视频页直读的共享核心：组装提示词、调用 Gemini 云端直读、拼装中文返回文本。

    与 _describe_video_bytes 走同一套提示词组装（_build_final_prompt_and_tokens）、思考等级自动
    降级重试、返回格式化（_format_describe_result），唯一区别是底层换成 client.describe_youtube
    （file_data 直传 file_uri、服务器不下载、通道标 "youtube"），因此也没有临时文件需要清理。
    调用方需自行保证：API key 已存在、youtube_url 已判定为 YouTube 链接。
    """
    final_prompt, effective_tokens = _build_final_prompt_and_tokens(prompt, persona, hint, max_output_tokens)

    client = GeminiVideoClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL, base_url=GEMINI_BASE_URL)
    notes: list[str] = [mode_note] if mode_note else []
    result: dict | None = None
    processing = "static"
    try:
        if mode == "agentic":
            # YouTube 直读也能进 agentic：uri 直接给视频页链接，不带 mime_type。
            try:
                result = await client.describe_video_agentic(
                    prompt=final_prompt + _AGENTIC_PROMPT_SUFFIX,
                    file_uri=youtube_url,
                    mime_type=None,
                    model=_agentic_model(),
                )
                processing = f"agentic（模型 {result.get('model')}）"
            except GeminiVideoError as e:
                notes.append(f"agentic 模式没能用上（{e}），已自动回退 static 云端直读。")
                result = None

        if result is not None:
            return _format_describe_result(result, processing, notes)

        try:
            result = await client.describe_youtube(
                youtube_url,
                final_prompt,
                low_resolution=low_resolution,
                max_output_tokens=effective_tokens,
                thinking_level=GEMINI_THINKING_LEVEL,
            )
        except GeminiVideoError as e:
            # 个别模型不支持当前思考等级（会 400），自动降级为 low 重试一次
            if "thinking level" not in str(e).lower():
                raise
            logger.info("模型 %s 不支持 thinking_level=%s，自动改用 low 重试", GEMINI_MODEL, GEMINI_THINKING_LEVEL)
            result = await client.describe_youtube(
                youtube_url,
                final_prompt,
                low_resolution=low_resolution,
                max_output_tokens=effective_tokens,
                thinking_level="low",
            )
    except GeminiVideoError as e:
        return f"视频识别失败：{e}"
    except Exception as e:  # noqa: BLE001 - 兜底：任何异常都不许裸抛出 MCP 边界
        logger.exception("YouTube 视频识别管线未预期异常")
        return f"发生了未预期的错误：{e}\n（如果反复出现，请把这条信息发给开发者。）"

    return _format_describe_result(result, "static", notes)


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
        # upload_cache.json 是索引不是媒体，删了只会让下次白传一遍，永远跳过。
        files = [f for f in _TEMP_MEDIA_DIR.iterdir() if f.is_file() and f.name != _UPLOAD_CACHE_NAME]
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
# YouTube 视频页判定（describe_video_url 的分流开关）
# ---------------------------------------------------------------------------
# Gemini API 原生支持直读 YouTube 视频页（file_data.file_uri 直传、云端拉取、不下载）。
# 命中这些主机名就走 describe_youtube 直读，不再尝试当直链下载。
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def _is_youtube_url(url: str) -> bool:
    """判断 url 是否是 YouTube 视频页链接（对 netloc 做精确白名单比对，大小写不敏感）。

    刻意用 urlparse 取出 netloc 再精确匹配，而不是子串包含——避免 evil.com/?u=youtube.com、
    youtube.com.evil.net 这类把 youtube 藏在别处的链接被误判成 YouTube。
    """
    try:
        netloc = urlparse((url or "").strip()).netloc.lower()
    except (ValueError, TypeError):
        return False
    # 剥掉可能存在的用户信息与端口（正常 YouTube 页面没有，稳妥起见仍处理）。
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return netloc in _YOUTUBE_HOSTS


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
    max_output_tokens: int = 30000,
    mode: str = "auto",
    start: float | str | None = None,
    end: float | str | None = None,
    fps: float | None = None,
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
        max_output_tokens: 最大输出 token 数，默认 30000。内部有 2048 的下限保护
            （Gemini 的“思考”token 也计入这里；默认思考等级为 high，思考会占用
            数千 token，太小会把正文挤没。思考等级可用环境变量 GEMINI_THINKING_LEVEL 调整）。
        mode: 处理方式。"auto"（默认：时长 ≥5 分钟走 agentic，否则 static）、"static"（全程按固定
            帧率抽帧，短片和需要逐帧全描述时用）、"agentic"（模型自己决定看哪几段、用什么帧率，长视频省 token）。
        start: 只看某一段的起点，秒数或 "1:06" 写法。给了 start/end 就进"细看"（强制 static，只把这一段喂给模型）。
        end: 只看某一段的终点；只给 start 时默认往后看 30 秒。
        fps: 细看时的抽帧率（默认 1，上限 10）。调高能抓住快动作，token 也按倍数涨。

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

    # 6) 时长（auto 分流与「细看」区间裁剪都要用；ffprobe 读不到就按 None 处理）
    duration = await asyncio.to_thread(_probe_duration_blocking, str(p))

    # 7) 「细看」参数 + 处理模式
    detail_range, fps_value, err = _resolve_detail_range(start, end, fps, duration)
    if err:
        return err
    if detail_range:
        # 只看一段时恒定走 static：agentic 的自主取帧与"只喂这几秒"是两套思路，不叠加。
        resolved_mode, mode_note = "static", None
    else:
        resolved_mode, mode_note = _resolve_mode(mode, duration)

    # 8) 交给共享识别管线（组装提示词 + 调 Gemini + 拼装返回文本）
    return await _describe_video_bytes(
        video_bytes,
        mime_type,
        prompt=prompt,
        persona=persona,
        hint=hint,
        low_resolution=low_resolution,
        max_output_tokens=max_output_tokens,
        mode=resolved_mode,
        detail_range=detail_range,
        fps=fps_value,
        mode_note=mode_note,
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
    max_output_tokens: int = 30000,
    mode: str = "auto",
) -> str:
    """从【网络直链】下载、或从【YouTube 视频页】云端直读视频再交给 Gemini 识别，返回按时间轴分段的中文描述。

    两种输入都支持：
    - 视频文件直链（以 .mp4/.mov/.webm 等结尾、点开就是视频本体）：下载到服务器 temp_media/ 临时目录，
      识别完就删掉。
    - ✅ YouTube 视频页链接（youtube.com/watch、youtu.be 短链、shorts 等）可以直接传，服务器不下载、
      由 Gemini 云端直读；仅支持公开视频（私享/会员/年龄限制的不行），免费层每天有 YouTube 总时长限额，
      长视频照常按秒计费。

    ⚠️ B站/抖音/TikTok 等其他平台页面仍不支持（那需要 yt-dlp 之类工具，本服务器暂不支持）。
       若给的是这类平台页面链接、或链接打开是网页而非视频文件，会明确报错。

    其余参数（prompt/persona/hint/low_resolution/max_output_tokens）含义与 describe_video 完全一致。

    Args:
        url: 视频文件的 http/https 直链，或 YouTube 视频页链接（youtube.com/watch、youtu.be、shorts）。
        prompt: 自定义提示词，传了就完全覆盖默认模板（此时 persona/hint 被忽略）。
        persona: 可选人设，仅在未传 prompt 时生效。
        hint: 可选前置线索，仅在未传 prompt 时生效。
        low_resolution: 低清省钱开关，默认关闭。
        max_output_tokens: 最大输出 token，默认 30000（内部有 2048 下限保护）。
        mode: 处理方式，含义同 describe_video："auto"（默认，≥5 分钟走 agentic）/"static"/"agentic"。
            YouTube 链接读不到时长，auto 一律按 static 处理。

    Returns:
        Gemini 生成的视频描述文本（末尾可能带用量统计或截断提示）。
    """
    err = _check_api_key()
    if err:
        return err
    if not url or not url.strip():
        return "没有提供链接。请把视频文件的 http/https 直链传给 url 参数。"

    # YouTube 视频页：走 Gemini 云端直读通道（file_data.file_uri 直传，服务器不下载）。
    # _describe_youtube_url 内部已兜住所有异常、绝不裸抛出 MCP 边界。
    if _is_youtube_url(url):
        # 拿不到时长（不下载），auto 只能按 static 走；显式要 agentic 则直接把视频页链接交给 Interactions。
        yt_mode, yt_note = _resolve_mode(mode, None)
        return await _describe_youtube_url(
            url.strip(),
            prompt=prompt,
            persona=persona,
            hint=hint,
            low_resolution=low_resolution,
            max_output_tokens=max_output_tokens,
            mode=yt_mode,
            mode_note=yt_note,
        )

    temp_path, mime_type, err = await _download_to_temp(url.strip())
    if err:
        return err
    assert temp_path is not None and mime_type is not None

    try:
        video_bytes = await asyncio.to_thread(temp_path.read_bytes)
        if not video_bytes:
            return "下载到的文件是空的（0 字节），请确认这个直链有效。"
        duration = await asyncio.to_thread(_probe_duration_blocking, str(temp_path))
        resolved_mode, mode_note = _resolve_mode(mode, duration)
        return await _describe_video_bytes(
            video_bytes,
            mime_type,
            prompt=prompt,
            persona=persona,
            hint=hint,
            low_resolution=low_resolution,
            max_output_tokens=max_output_tokens,
            mode=resolved_mode,
            mode_note=mode_note,
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
