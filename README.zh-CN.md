[English](README.md) | 简体中文

# Gemini-Video-MCP

把 **MoFox-Bot 主程序里打磨成熟的 Gemini 视频直传识别能力**剥出来，做成一个独立的 MCP 服务器：
输入本地视频文件路径，输出 Gemini 生成的、按时间轴分段的详细内容描述（画面 + 音轨）。

Gemini 描述视频的效果非常好——好到会自由发挥出戏剧张力和抒情感。本服务器的默认提示词
**刻意保留了这种自由发挥**（不禁止抒情、不要求客观简洁），并要求用**中文**输出；
要压制文风或换成别的语言请自己传 `prompt`。

## 能力一览

- `describe_video` —— 主工具：**本地**视频 → 文字描述。
  - 支持 `mp4 / mov / webm / avi / mkv / flv / wmv / mpeg / mpg / m4v / 3gp / 3gpp` 等常见视频格式；
  - **也直接支持 `.gif`**（以 `image/gif` 原格式直传，Gemini 能感知完整动画，不抽帧）；
  - 小视频（≤14MB）内联上传、单次完成；大视频自动走 Files API（上传→等待就绪→识别→用完删除远端文件），单文件上限 2GB；
  - `low_resolution` 低清省钱开关；`persona` 可让 Gemini 附体任意人格来解说；`hint` 可注入人类对视频的已有形容作为理解线索（适合抽象/玩梗内容，模型仍以实际所见为准）；`prompt` 完全自定义。
- `describe_video_url` —— 从**网络直链**下载视频再识别（下载到临时目录、识别完即删）。
  - 参数与 `describe_video` 完全一致，只是把 `path` 换成 `url`；
  - **只认视频文件直链**（点开就是视频本体、以 `.mp4/.mov/.webm` 等结尾）。视频网站的播放**页面**链接（B站/抖音/TikTok/YouTube 等）**不是直链、不支持解析**（那需要 yt-dlp 之类工具，属未来野心版，本轮不做）；
  - 单文件上限 500MB；连接 15 秒 / 总 300 秒超时；遇到网页、非 2xx、超大文件都会给可读中文报错。
- `view_media` —— 把一张**图片**、或**视频的某一帧**，作为**图片内容**直接返回，让调用方模型（claude 等）**亲眼看到画面**（区别于 `describe_video` 是让 Gemini 看完写文字）。
  - 图片（`png/jpg/jpeg/webp/gif`）：直接返回（长边超过 `max_dimension`（默认 1024，可调范围 16~4096）会等比缩小，不放大）；GIF 只返回**首帧**（整段动画请用 `describe_video`）；
  - 视频：给了 `timestamp`（秒）就抽那一帧，不给则抽**正中间**那一帧；需要本机装了 **ffmpeg**（缺失会中文报错）。
- `estimate_cost` —— 小工具：发大视频前估算大概消耗多少输入 token（优先 ffprobe 读时长，读不到按大小粗估）。
- `get_upload_url` —— 小工具：拿到"把文件推到本服务器"的上传地址，配合 claude.ai 沙盒把聊天里上传的文件中转过来再识别（见下文「HTTP 模式」）。**本地 stdio 模式下没有上传端点**，调用它会直接告诉你"无需上传，同一台电脑直接传路径即可"。

## 安装

需要 [uv](https://docs.astral.sh/uv/)（与本目录其它 MCP 一致的依赖管理方式）、**Python ≥3.11**（uv 会按 `pyproject.toml` 自动处理），以及本机 **ffmpeg / ffprobe**（`view_media` 抽帧缩放、`estimate_cost` 读时长都靠它们；没装的话这两个工具会给出中文报错，`describe_video`/`describe_video_url` 不受影响）。

```bash
cd /path/to/Gemini_Video_MCP
uv venv
uv pip install -e .
```

### 配置 API key

复制 `.env.example` 为 `.env`，填入你的 key（在 [Google AI Studio](https://aistudio.google.com/apikey) 免费申领）：

```env
GEMINI_API_KEY=你的key
GEMINI_MODEL=gemini-3.5-flash
```

也可以不写 `.env`，改在注册时用 `-e GEMINI_API_KEY=...` 注入（见下）。
两者都提供时，**环境变量优先**，`.env` 只作兜底。

### 环境变量一览

| 变量 | 必填？ | 默认值 | 说明 |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | 是 | 无 | Gemini API key |
| `GEMINI_MODEL` | 否 | `gemini-3.5-flash` | 模型标识 |
| `GEMINI_BASE_URL` | 否 | `https://generativelanguage.googleapis.com/v1beta` | API 根地址，走中转/代理时才需要改 |
| `GEMINI_THINKING_LEVEL` | 否 | `high` | 思考等级：`minimal`/`low`/`medium`/`high`；想省钱改 `minimal` |
| `GEMINI_MCP_HTTP_SECRET` | 仅 `--http` 模式必填 | 无 | HTTP 模式的唯一门锁；缺失或仍是占位符会被拒绝启动 |
| `GEMINI_MCP_PUBLIC_BASE_URL` | 否 | 无（回退 `http://localhost:8768`） | 经隧道暴露后的公网地址，`get_upload_url` 用它拼上传地址 |

（表里都是占位符，实际值填你自己的，别提交进 git——`.env` 已在 `.gitignore` 里。）

## 运行（本地调试）

```bash
uv run python main.py        # 以 stdio 启动（正常情况下由 Claude 客户端拉起，不用手动跑）
```

## 注册到 Claude Code

在本目录下执行（把 key 通过 `-e` 传入）：

```bash
claude mcp add gemini-video -e GEMINI_API_KEY=你的key -- uv run --directory /path/to/Gemini_Video_MCP python main.py
```

如果你已经写好了 `.env`，可以省掉 `-e`：

```bash
claude mcp add gemini-video -- uv run --directory /path/to/Gemini_Video_MCP python main.py
```

验证：

```bash
claude mcp list
```

## 注册到 Claude Desktop

编辑 Claude Desktop 的配置文件（Windows：`%APPDATA%\Claude\claude_desktop_config.json`），
在 `mcpServers` 下加入：

```json
{
  "mcpServers": {
    "gemini-video": {
      "command": "uv",
      "args": [
        "--directory",
        "D:/path/to/Gemini_Video_MCP",
        "run",
        "python",
        "main.py"
      ],
      "env": {
        "GEMINI_API_KEY": "你的key"
      }
    }
  }
}
```

> Claude Desktop 不能把聊天里上传的视频直接喂给 MCP 工具。想让 Claude 读你电脑里的视频，
> 直接在对话里写出视频的**完整文件路径**即可（例如 `D:/videos/cat.mp4`）。

## HTTP 模式（供 claude.ai / 手机远程使用）

默认的 stdio 模式只能被"同一台电脑上的 Claude 客户端"拉起。如果你想让 **claude.ai 网页版 / 手机 App**
也用上这个视频识别工具，就需要 HTTP 模式：让服务器在本机开一个端口，把它暴露到公网，再在 claude.ai
里添加成一个 **Custom Connector（自定义连接器）**。

一句话原理：**HTTP 模式在本机 8768 端口提供服务；隧道工具（如 Cloudflare Tunnel）把这个端口暴露到公网；
访问地址里那段 secret 就是唯一门锁——谁不知道这段 secret 就进不来。**

### 第 1 步：设置访问口令（secret）

secret 会被拼进访问地址的路径（`/mcp/<secret>`），是这个端口对外的唯一防线，必须又长又随机。

1. 生成一段随机口令（在本目录开个终端跑）：
   ```bash
   uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. 把输出那串字符写进本目录 `.env` 的这一行（没有 `.env` 就参考 `.env.example` 新建）：
   ```env
   GEMINI_MCP_HTTP_SECRET=把上一步生成的那串粘到这里
   ```
   > 千万别公开这段 secret，也别截图发人——它泄露了等于把门钥匙给了别人。
   > 留空或还是占位符时，服务器会**拒绝**以 `--http` 启动（这是防止裸奔上公网的保护）。

### 第 2 步：启动 HTTP 服务

双击本目录的 **`start_http.bat`**（或在终端跑 `uv run python main.py --http`）。启动成功后本机地址是：
```
http://localhost:8768/mcp/<你的secret>
```
这个黑窗口要一直开着；关掉 = 服务停了。

### 第 3 步：用隧道工具暴露到公网（推荐 Cloudflare Tunnel）

如果你已经有一条现成的隧道（比如之前挂过别的本地服务），直接复用即可，方法通用：把这个新服务的
公网入口指向本机的 `http://localhost:8768`。以 **Cloudflare Tunnel** 为例：

1. 打开 **Cloudflare Zero Trust 面板** → Networks → Tunnels → 选中你那条 tunnel → **Configure**。
2. 在 **Public Hostname** 标签页点 **Add a public hostname**，填：
   - **Subdomain**：给这个服务起个子域名，比如 `gemini-video`；**Domain** 选你自己的域名
     （最终地址会是 `gemini-video.你的域名.com`）。
   - **Service** → Type 选 `HTTP`，URL 填 **`http://localhost:8768`**（注意是 http、是 localhost、是 8768）。
3. 保存，等一两分钟 DNS 生效。

> 只是想临时试一下、不想动面板？跑一条一次性命令即可（地址每次会变，不适合长期用）：
> ```bash
> cloudflared tunnel --url http://localhost:8768
> ```
> 它会打印一个 `https://随机名字.trycloudflare.com` 地址，用完即弃。
>
> （备选：如果想用 tailscale 而非 cloudflare，对应命令是 `tailscale funnel 8768`，
> 会给出一个 `https://<机器名>.<tailnet>.ts.net` 的公网地址，同样把 `/mcp/<secret>` 接在后面。）

### 第 4 步：填进 claude.ai 的 Custom Connector

1. 打开 **claude.ai** → 头像 → **Settings（设置）** → **Connectors（连接器）**。
2. 点 **Add custom connector（添加自定义连接器）**。
3. **Name**：随便起，比如 `Gemini 视频识别`。
4. **Remote MCP server URL**：填公网地址 + secret 路径，即：
   ```
   https://gemini-video.你的域名.com/mcp/<你的secret>
   ```
   （域名换成第 3 步实际配的那个；`<你的secret>` 换成第 1 步生成的那串。）
5. 保存。Claude 会尝试连接并列出 `describe_video` / `describe_video_url` / `view_media` / `estimate_cost` / `get_upload_url` 五个工具，出现了就成功了。

之后在 claude.ai 对话里就能让它识别视频了。注意：远程模式下 Claude 读的是**服务器那台电脑**上的
文件路径，不是你手机里的文件。

> Custom Connector 通常需要 Claude 的付费套餐（Pro / Max 等），具体入口措辞不同版本可能略有差异。

### 让 claude.ai 沙盒把文件推到本机（上传端点）

远程模式下，`describe_video` 读的是**服务器那台电脑**上的路径，claude.ai 聊天里上传/沙盒里下载的文件到不了本机。
为此 **`--http` 模式额外开了一个上传端点**：

```
POST http://localhost:8768/upload/<你的secret>
```

它用的是**同一个** `GEMINI_MCP_HTTP_SECRET`（路径里 secret 不对直接 404）。在 claude.ai 里让**沙盒**跑一段代码，
把它手上的文件用 multipart 表单 POST 上来即可（`<公网地址>` 换成你的隧道域名，`<secret>` 换成你的口令）：

```python
import requests

# 沙盒里已经有一个文件（聊天上传的、或沙盒自己下载/生成的），比如 /tmp/clip.mp4
resp = requests.post(
    "https://gemini-video.你的域名.com/upload/<secret>",
    files={"file": open("/tmp/clip.mp4", "rb")},
)
print(resp.json())
# -> {"saved_path": "/path/to/Gemini_Video_MCP/temp_media/20260714_..._clip.mp4", "size_mb": 3.2, "hint": "把 saved_path 传给 describe_video 即可"}
```

拿到返回的 `saved_path` 后，直接让 Claude 用它调 `describe_video`（或 `view_media`）就能识别了。

> **不用手动教它**：服务器带了 `get_upload_url` 工具和服务器级 instructions——claude.ai 里的 Claude
> 遇到"聊天里上传的文件"会自己调 `get_upload_url` 拿地址、在沙盒里 POST、再拿 `saved_path` 识别。
> 你只需要上传视频后说一句"看看这个"。前提：`.env` 里配了 `GEMINI_MCP_PUBLIC_BASE_URL`（公网域名），
> 且 claude.ai 沙盒能访问外网。注意 `get_upload_url` 的返回里含完整 secret（这是沙盒能 POST 的前提），
> 它会出现在那段对话里——介意的话用完那轮对话即弃，或定期换锁。

约束：单文件上限 **500MB**；文件名会被安全化（防路径穿越）+ 加时间戳前缀防覆盖；**绝不接受调用方指定存储路径**，
一律存进服务器目录下的 `temp_media/`（该目录总量超过 2GB 时按最旧自动清理）。stdio 本地模式**不**提供这个端点。

## 用法示例（在对话里）

- "帮我描述一下 `D:/videos/monkey.mp4` 这个视频" → 触发 `describe_video`。
- "用一个毒舌女高中生的口吻讲讲 `D:/videos/dance.mp4`" → Claude 会带上 `persona`。
- "这个视频是在用鸡蛋下五子棋，帮我描述 `D:/videos/eggs.mp4`" → Claude 会把你的形容放进 `hint`，帮模型看懂抽象整活。
- "帮我描述这个直链视频 `https://example.com/clip.mp4`" → 触发 `describe_video_url`（先下载再识别）。
- "给我看看 `D:/videos/monkey.mp4` 第 8 秒长什么样" → 触发 `view_media`（抽第 8 秒那一帧给 Claude 亲眼看）。
- "把 `D:/pics/meme.png` 这张图给你自己看看" → 触发 `view_media`（直接把图投给 Claude）。
- "`D:/videos/long.mp4` 这个视频识别一次大概多少 token？" → 触发 `estimate_cost`。
- 长视频想省钱：让 Claude 加 `low_resolution=True`。

## 费用说明

Gemini 按视频时长计输入 token：

| 模式 | 速率 | 1 分钟视频 | 说明 |
| --- | --- | --- | --- |
| 标清（默认） | 约 **300 token/秒** | 约 **1.8 万**输入 token | 画面细节更足 |
| 低清（`low_resolution=True`） | 约 **100 token/秒** | 约 **0.6 万**输入 token | 长视频省钱，细节变粗 |

以上只是**输入** token；输出 token 另计，取决于描述长短。发大视频前可先用 `estimate_cost` 心里有数。

## 关于 thinking token 的坑（已内建对策）

Gemini 3.5 的"思考"token 也计入 `maxOutputTokens`。如果输出预算太小或思考等级太高，
思考会把可见输出挤没、导致描述被截断。

本服务器默认 `thinking_level=high`（实测思考越多、描述的文学发挥越好），并相应把默认
`max_output_tokens` 设为 8192 给思考留足余量；检测到截断时会在结果末尾追加提示。
想省钱可在 `.env` 里设 `GEMINI_THINKING_LEVEL=minimal`（思考 token 按输出计费）。
模型不支持当前思考等级时会自动降级为 `low` 重试一次。

## 已知限制

- `.mkv` 不在 Gemini 官方支持列表里，本服务器按 `video/x-matroska` 尽力尝试；若被拒绝，请先转成 `mp4`。
- Files API 单文件上限 2GB；再大请先裁短/压缩。
- 视频时长上限（官方）：标清最长约 1 小时、低清（`low_resolution=True`）最长约 3 小时；超长视频请裁剪分段。
- 若把 `GEMINI_MODEL` 换成 Gemini 2.5 系模型：2.5 用 `thinkingBudget` 而非 `thinkingLevel`，默认发送的 `thinkingLevel` 可能被 API 400 拒绝；默认的 3.5-flash 无此问题。
- `estimate_cost` 在没有 `ffprobe` 时按文件大小粗估时长，误差可能较大（结果里会注明）。
- `view_media` 依赖本机 **ffmpeg**（抽帧/缩放，本项目按 ffmpeg 8.0 测试）；未装 ffmpeg 时会中文报错。GIF 只返回首帧（想感知整段动画请用 `describe_video`）。
- `describe_video_url` **只支持视频文件直链**，不解析平台播放页（B站/抖音/TikTok/YouTube 等）；单文件 500MB、连接 15s/总 300s 超时。
  从 claude.ai 远程调用时，下载耗时会叠加在隧道代理的超时上（如 Cloudflare 约 100 秒），大文件更易被掐断——大文件建议改用**上传端点**先把文件推到本机、再走本地路径识别。
- 上传端点（`POST /upload/<secret>`）仅 `--http` 模式存在；单文件 500MB 上限、`temp_media/` 总量 2GB 上限（超了按最旧清理）。
- 传输模式：默认 stdio（本地）；`--http` 为可选的远程模式（详见上文「HTTP 模式」）。
- **HTTP 模式经 Cloudflare 时的 100 秒超时（524）**：Cloudflare 代理对单个请求约 100 秒没响应就会掐断返回 524。
  小视频走 inline 通常 15~60 秒没问题；但大视频走 Files API 可能要好几分钟，从 claude.ai 远程调用容易被掐断。
  **要识别大文件，建议仍用本地 stdio 模式（Claude Code / Desktop）喂**，那条路径没有这个超时。

## 许可证

[MIT License](LICENSE)。想怎么用都行，出了问题也别找我担责。
