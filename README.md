English | [简体中文](README.zh-CN.md)

# Gemini-Video-MCP

An MCP server that hands local video files to **Gemini's native video understanding** and returns a
detailed, timestamped description of what's happening on screen and in the audio track. The video
pipeline was extracted and cleaned up from a larger Discord/QQ bot project (MoFox-Bot) where it had
already been battle-tested.

Gemini is *really* good at describing video — good enough that it tends to get dramatic and lyrical
about it. The default prompt in this server **deliberately leaves that flourish in** (it doesn't ask
for dry, objective summaries) and asks Gemini to write the description **in Chinese**. If you want a
flatter tone or a different output language, pass your own `prompt`.

## Tools at a glance

- **`describe_video`** — the main tool: local video → text description.
  - Supports common containers: `mp4 / mov / webm / avi / mkv / flv / wmv / mpeg / mpg / m4v / 3gp / 3gpp`;
  - **Also handles `.gif` natively** (sent as raw `image/gif`, so Gemini perceives the full animation instead of a single extracted frame);
  - Small videos (≤14MB) go inline in a single request; larger ones automatically go through the Files API (upload → wait until ready → describe → delete the remote copy when done), up to a 2GB per-file limit;
  - `low_resolution` is a cost-saving switch; `persona` lets Gemini narrate in character; `hint` lets you feed in what a human already thinks the video is about, useful for abstract or meme-y content (the model is instructed to still describe what it actually sees, not just agree with you); `prompt` fully overrides the built-in template.
- **`describe_video_url`** — downloads a video from a **direct link** and runs it through the same pipeline (downloaded to a temp folder, deleted right after).
  - Same parameters as `describe_video`, just `path` becomes `url`;
  - **Only works with direct video file links** (something that resolves straight to a video file, ending in `.mp4/.mov/.webm`/etc.). Platform *watch pages* (YouTube, TikTok, Bilibili, etc.) are **not** direct links and won't resolve — that would need something like yt-dlp, which is out of scope for now;
  - 500MB per-file cap; 15s connect / 300s total timeout; non-video pages, non-2xx responses, and oversized files all get a readable error.
- **`view_media`** — returns a **picture** (either an image file, or one frame pulled from a video) as actual **image content**, so the calling model (Claude, etc.) can look at it directly — as opposed to `describe_video`, which has Gemini watch the video and write text about it.
  - Images (`png/jpg/jpeg/webp/gif`): returned as-is, downscaled (never upscaled) if the longer side exceeds `max_dimension` (default 1024, adjustable 16–4096). GIFs return only their **first frame** (use `describe_video` if you want the full animation understood);
  - Videos: pass `timestamp` (in seconds) to grab that frame, or omit it to grab the frame at the **midpoint**; requires **ffmpeg** on the host machine (a clear error is returned if it's missing).
- **`estimate_cost`** — a small helper that estimates roughly how many input tokens a video will cost before you send it (uses `ffprobe` for real duration when available, otherwise estimates from file size).
- **`get_upload_url`** — a small helper that returns the upload endpoint URL so a claude.ai sandbox can push a chat-uploaded file to this server before describing it (see "HTTP mode" below). **Under local stdio mode there is no upload endpoint**, so calling this just returns "not needed — you're on the same machine, pass the local path directly."

## Installation

You'll need [uv](https://docs.astral.sh/uv/) for dependency management, **Python ≥3.11** (uv will
resolve this automatically from `pyproject.toml`), and **ffmpeg / ffprobe** on your `PATH` (used by
`view_media` for frame extraction/scaling and by `estimate_cost` for reading duration — `describe_video`
and `describe_video_url` don't need them; missing ffmpeg only breaks the first two, with a clear error
message).

```bash
cd /path/to/Gemini_Video_MCP
uv venv
uv pip install -e .
```

### Configure the API key

Copy `.env.example` to `.env` and fill in your key (free to get at
[Google AI Studio](https://aistudio.google.com/apikey)):

```env
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash
```

You can also skip `.env` entirely and inject the key at registration time with `-e GEMINI_API_KEY=...`
(see below). If both are present, **the environment variable wins**; `.env` is only a fallback.

### Environment variables

| Variable | Required? | Default | Notes |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | — | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash` | Model identifier |
| `GEMINI_BASE_URL` | No | `https://generativelanguage.googleapis.com/v1beta` | API base URL; only change this if you're proxying |
| `GEMINI_THINKING_LEVEL` | No | `high` | One of `minimal`/`low`/`medium`/`high`; set `minimal` to save cost |
| `GEMINI_MCP_HTTP_SECRET` | Only for `--http` mode | — | The sole access lock for HTTP mode; server refuses to start without a real value |
| `GEMINI_MCP_PUBLIC_BASE_URL` | No | none (falls back to `http://localhost:8768`) | Public URL after tunneling; used by `get_upload_url` to build the upload link |

(All values above are placeholders — fill in your own and never commit `.env`; it's already in
`.gitignore`.)

## Running locally (for debugging)

```bash
uv run python main.py        # starts in stdio mode (normally launched by your Claude client, no need to run this by hand)
```

## Registering with Claude Code

Run this from inside the project directory (passing the key via `-e`):

```bash
claude mcp add gemini-video -e GEMINI_API_KEY=your-key-here -- uv run --directory /path/to/Gemini_Video_MCP python main.py
```

If you've already set up `.env`, you can drop the `-e`:

```bash
claude mcp add gemini-video -- uv run --directory /path/to/Gemini_Video_MCP python main.py
```

Verify it registered:

```bash
claude mcp list
```

## Registering with Claude Desktop

Edit Claude Desktop's config file (Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and add
this under `mcpServers`:

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
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

> Claude Desktop can't hand a chat-uploaded video straight to an MCP tool. To have Claude read a video
> from your machine, just type out the video's **full file path** in the conversation (e.g.
> `D:/videos/cat.mp4`).

## HTTP mode (for claude.ai / mobile remote use)

The default stdio mode can only be launched by a Claude client running on the same machine. If you
want **claude.ai's web app or mobile app** to use this video tool too, you need HTTP mode: the server
opens a port locally, you expose that port to the internet, and then you add it in claude.ai as a
**Custom Connector**.

The short version: **HTTP mode serves on local port 8768; a tunnel (e.g. Cloudflare Tunnel) exposes
that port to the internet; the secret baked into the URL path is the only lock on the door — without
it, nobody gets in.**

### Step 1: set an access secret

The secret gets embedded in the URL path (`/mcp/<secret>`), and it's the *only* thing standing between
this port and the public internet, so it needs to be long and random.

1. Generate a random secret (run this from a terminal in the project directory):
   ```bash
   uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Paste the output into this line in your `.env` (copy `.env.example` first if you don't have one yet):
   ```env
   GEMINI_MCP_HTTP_SECRET=paste-the-generated-string-here
   ```
   > Never share this secret or screenshot it — leaking it is the same as handing someone the front
   > door key. If it's left blank or still a placeholder, the server will **refuse** to start with
   > `--http` (this is intentional, so it never goes online unprotected).

### Step 2: start the HTTP server

Double-click `start_http.bat` in the project directory (or run `uv run python main.py --http` in a
terminal). Once it's up, the local address is:
```
http://localhost:8768/mcp/<your-secret>
```
Keep that terminal window open — closing it stops the server.

### Step 3: expose it to the internet with a tunnel (Cloudflare Tunnel recommended)

If you already have a tunnel set up for other local services, you can reuse it — just point a new
public hostname at `http://localhost:8768`. Using **Cloudflare Tunnel** as an example:

1. Open the **Cloudflare Zero Trust dashboard** → Networks → Tunnels → select your tunnel → **Configure**.
2. On the **Public Hostname** tab, click **Add a public hostname**:
   - **Subdomain**: pick something like `gemini-video`; **Domain**: your own domain (the final address
     will be `gemini-video.yourdomain.com`).
   - **Service** → Type: `HTTP`, URL: **`http://localhost:8768`** (note: http, localhost, port 8768).
3. Save and wait a minute or two for DNS to propagate.

> Just want to try it once, without touching the dashboard? Run a throwaway command (the address
> changes every time, so it's not for long-term use):
> ```bash
> cloudflared tunnel --url http://localhost:8768
> ```
> It prints a `https://random-name.trycloudflare.com` address you can use once and discard.
>
> (Alternative: if you'd rather use Tailscale instead of Cloudflare, the equivalent is
> `tailscale funnel 8768`, which gives you a `https://<machine-name>.<tailnet>.ts.net` public address —
> append `/mcp/<secret>` the same way.)

### Step 4: add it as a Custom Connector in claude.ai

1. Open **claude.ai** → avatar → **Settings** → **Connectors**.
2. Click **Add custom connector**.
3. **Name**: anything you like, e.g. `Gemini Video`.
4. **Remote MCP server URL**: your public address plus the secret path:
   ```
   https://gemini-video.yourdomain.com/mcp/<your-secret>
   ```
   (swap in whatever domain you set up in Step 3, and the secret you generated in Step 1.)
5. Save. Claude should connect and list all five tools — `describe_video` / `describe_video_url` /
   `view_media` / `estimate_cost` / `get_upload_url` — which means it worked.

From then on you can ask Claude to describe videos right from claude.ai. Note that in remote mode,
Claude reads file paths **on the server machine**, not files on your phone.

> Custom Connectors typically require a paid Claude plan (Pro/Max/etc.); the exact menu wording may
> vary by app version.

### Letting the claude.ai sandbox push files to your machine (upload endpoint)

In remote mode, `describe_video` still only reads paths **on the server machine** — files uploaded in
a claude.ai chat or downloaded inside its sandbox can't reach your machine on their own. That's what
the **`--http` mode's extra upload endpoint** is for:

```
POST http://localhost:8768/upload/<your-secret>
```

It reuses the **same** `GEMINI_MCP_HTTP_SECRET` (a wrong secret in the path just 404s). Have claude.ai's
**sandbox** run a snippet that POSTs the file it has as multipart form data (swap in your tunnel domain
and secret):

```python
import requests

# The sandbox already has a file (chat-uploaded, or downloaded/generated by the sandbox itself), e.g. /tmp/clip.mp4
resp = requests.post(
    "https://gemini-video.yourdomain.com/upload/<secret>",
    files={"file": open("/tmp/clip.mp4", "rb")},
)
print(resp.json())
# -> {"saved_path": "/path/to/Gemini_Video_MCP/temp_media/20260714_..._clip.mp4", "size_mb": 3.2, "hint": "pass saved_path to describe_video"}
```

Once you have `saved_path`, just have Claude call `describe_video` (or `view_media`) with it.

> **You don't need to explain any of this to Claude by hand**: the server ships a `get_upload_url` tool
> plus server-level instructions, so Claude in claude.ai will automatically call `get_upload_url` when
> it sees a chat-uploaded file, POST it from the sandbox, and use the returned `saved_path`. You just
> upload a video and say "take a look at this." Prerequisites: `GEMINI_MCP_PUBLIC_BASE_URL` (your
> public domain) is set in `.env`, and the claude.ai sandbox has outbound internet access. Note that
> `get_upload_url`'s response contains the full secret (that's what lets the sandbox POST), so it will
> show up in that conversation — if that bothers you, discard that chat afterward or rotate the secret
> periodically.

Constraints: 500MB per file; filenames are sanitized (to prevent path traversal) and prefixed with a
timestamp (to prevent overwrites); **the caller can never choose a storage path** — everything lands in
the server's own `temp_media/` directory (auto-pruned, oldest first, once the directory exceeds 2GB
total). The stdio/local mode does **not** expose this endpoint.

## Example usage (in conversation)

- "Describe the video at `D:/videos/monkey.mp4` for me" → triggers `describe_video`.
- "Narrate `D:/videos/dance.mp4` like a snarky high schooler" → Claude passes a `persona`.
- "This video is people playing five-in-a-row with eggs, describe `D:/videos/eggs.mp4`" → Claude puts
  your description into `hint` so the model can make sense of something abstract or meme-y.
- "Describe this direct video link: `https://example.com/clip.mp4`" → triggers `describe_video_url`
  (downloads, then describes).
- "Show me what `D:/videos/monkey.mp4` looks like at the 8-second mark" → triggers `view_media` (pulls
  the frame at 8s for Claude to actually look at).
- "Take a look at this picture, `D:/pics/meme.png`" → triggers `view_media` (feeds the image straight
  to Claude).
- "About how many tokens would it cost to describe `D:/videos/long.mp4`?" → triggers `estimate_cost`.
- Want to save money on a long video? Have Claude set `low_resolution=True`.

## Cost notes

Gemini charges input tokens by video duration:

| Mode | Rate | 1-minute video | Notes |
| --- | --- | --- | --- |
| Standard (default) | ~**300 tokens/sec** | ~**18k** input tokens | More visual detail |
| Low-res (`low_resolution=True`) | ~**100 tokens/sec** | ~**6k** input tokens | Cheaper for long videos, coarser detail |

These figures are **input** tokens only; output tokens are billed separately depending on how long the
description ends up being. Run `estimate_cost` before sending a large video if you want to know what
you're in for.

## The thinking-token gotcha (already mitigated)

Gemini 3.5's "thinking" tokens also count against `maxOutputTokens`. If the output budget is too small,
or the thinking level too high, thinking can crowd out the visible output entirely and truncate the
description.

This server defaults to `thinking_level=high` (empirically, more thinking produces noticeably better
prose) and correspondingly defaults `max_output_tokens` to 8192 to leave room for it; if truncation is
detected, a note is appended to the result. To save money, set `GEMINI_THINKING_LEVEL=minimal` in
`.env` (thinking tokens are billed as output tokens). If the model doesn't support the configured
thinking level, the server automatically retries once at `low`.

## Known limitations

- `.mkv` isn't on Gemini's officially supported list; this server sends it as `video/x-matroska` on a
  best-effort basis. If it gets rejected, convert to `mp4` first.
- Files API cap is 2GB per file; trim or compress anything larger.
- Official video duration limits: roughly 1 hour at standard resolution, roughly 3 hours at low
  resolution (`low_resolution=True`); split up anything longer.
- If you swap `GEMINI_MODEL` for a Gemini 2.5-series model: 2.5 uses `thinkingBudget` instead of
  `thinkingLevel`, and the `thinkingLevel` this server sends by default may get rejected with a 400.
  The default 3.5-flash doesn't have this issue.
- `estimate_cost` falls back to a rough size-based duration estimate when `ffprobe` isn't available,
  which can be noticeably off (the result says so).
- `view_media` depends on **ffmpeg** on the host machine (frame extraction/scaling; tested against
  ffmpeg 8.0); a clear error is returned if it's missing. GIFs only return their first frame — use
  `describe_video` to have the full animation understood.
- `describe_video_url` **only supports direct video file links**, not platform watch pages
  (YouTube/TikTok/Bilibili/etc.); 500MB per file, 15s connect / 300s total timeout. When called
  remotely from claude.ai, download time stacks on top of the tunnel's own proxy timeout (e.g. ~100s
  for Cloudflare), so large files are more likely to get cut off — for large files, prefer the **upload
  endpoint** to push the file to the server first, then describe it via the local path.
- The upload endpoint (`POST /upload/<secret>`) only exists in `--http` mode; 500MB per file, 2GB total
  for `temp_media/` (oldest files pruned first once that's exceeded).
- Transport: stdio (local) by default; `--http` is an optional remote mode (see "HTTP mode" above).
- **The ~100-second Cloudflare proxy timeout (524) in HTTP mode**: Cloudflare's proxy cuts off any
  single request that goes unanswered for about 100 seconds, returning a 524. Small videos going inline
  are usually fine (15–60s); large videos routed through the Files API can take several minutes and are
  easily cut off when called remotely from claude.ai. **For large files, local stdio mode (Claude Code /
  Desktop) is still the way to go** — that path doesn't have this timeout at all.

## License

[MIT](LICENSE) — do whatever you want with it, no warranty implied.
