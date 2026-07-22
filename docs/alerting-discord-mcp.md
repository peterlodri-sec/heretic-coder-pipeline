# Pipeline alerting via a Discord MCP + entheai — e2e guide

Goal: when a training run finishes (or fails), get a **Discord alert** — posted by
**entheai** through a **Discord MCP server**, using the pipeline's existing status
**webhook** as the trigger. No new infra beyond a bot token and one bridge.

```
box (Status.write) ──POST──▶ alert bridge ──(terminal event)──▶ entheai + Discord MCP ──▶ #pipeline-alerts
        │                         │
        └──────────▶ live monitor (SSE)   (the same webhook already drives the ngrok dashboard)
```

`[verify]` markers flag anything you should confirm against the current version of
a third-party tool before relying on it.

---

## 0. Prerequisites

- **entheai** installed and working (`entheai --doctor` green), with its MCP client
  (it "spawns MCP servers at startup, tools exposed to the agent").
- The pipeline's live webhook already exists: every `Status.write()` POSTs to
  `PIPELINE_WEBHOOK_URL` (see `shared/webhook.py`). We reuse that.
- A Discord server (your **ENTHEAI** guild) where you can add a bot.

---

## 1. Discord side — bot + channel (5 min)

1. **Create the app + bot:** https://discord.com/developers/applications → **New
   Application** → name it `entheai-alerts` → **Bot** tab → **Reset Token**, copy
   the token (this is your `DISCORD_BOT_TOKEN`; treat it like a password).
2. **Intents:** on the Bot tab, enable **Message Content Intent** (needed if the
   MCP reads messages; not needed for send-only, but most Discord MCPs ask for it).
3. **Invite it to the ENTHEAI server:** **OAuth2 → URL Generator** → scopes
   `bot` (+ `applications.commands` if the MCP uses slash commands) → bot
   permissions: **Send Messages**, **Embed Links**, **Read Message History** →
   open the generated URL → add to your server.
4. **Make an `#pipeline-alerts` channel**, then copy its **channel ID**
   (Discord → User Settings → Advanced → Developer Mode ON → right-click the
   channel → **Copy Channel ID**). Save it as `DISCORD_ALERT_CHANNEL_ID`.

---

## 2. The Discord MCP server

Pick a maintained **stdio** Discord MCP (they expose tools like `send_message`,
`read_messages`, `list_channels`). Popular options `[verify current maintenance +
exact tool names]`:

| Server | Runtime | Spawn |
| --- | --- | --- |
| `mcp-discord` (Node) | Node ≥ 18 | `npx -y mcp-discord` |
| `discord-mcp` (Python) | Python 3.11 | `uvx discord-mcp` |

Whichever you choose, it reads the **bot token from an env var** (commonly
`DISCORD_TOKEN` or `DISCORD_BOT_TOKEN` — `[verify]` per its README) and speaks
JSON-RPC over stdio, which is exactly what entheai spawns.

Put the secret in your gitignored env (same pattern as the valyu bridge in
`entropy-om/mcp-config`):

```bash
# ~/dev/entropy-om/mcp-config/.env   (never committed)
DISCORD_BOT_TOKEN=xxxxx.yyyyy.zzzzz
DISCORD_ALERT_CHANNEL_ID=123456789012345678
```

---

## 3. Wire it into entheai (`entheai.toml`)

entheai config is `pub mcp: HashMap<String, McpServerConfig>` → one `[mcp.<name>]`
table per server, `command` + `args`, env expanded at spawn from your `.env`
(`source .env` before launch, as the mcp-config README describes). Add:

```toml
# entheai.toml
[mcp.discord]
command = "/bin/sh"
args = ["-c", "exec npx -y mcp-discord"]     # or: exec uvx discord-mcp   [verify the package]
# The server reads the token from the environment it inherits from your shell;
# make sure DISCORD_BOT_TOKEN (or the name that server expects) is exported first.
```

Then:

```bash
cd ~/dev/entropy-om/mcp-config && source .env   # export DISCORD_BOT_TOKEN etc.
entheai --doctor                                # confirms MCP servers spawn
entheai "list your MCP tools"                   # you should see the discord.* tools
```

Also add a shareable example to the team repo, matching its convention — a new
`entropy-om/mcp-config/servers/discord.mcp.json.example`:

```json
{
  "mcpServers": {
    "discord": {
      "command": "/bin/sh",
      "args": ["-c", "exec npx -y mcp-discord"],
      "description": "Discord bot (send/read messages) over stdio; token from $DISCORD_BOT_TOKEN"
    }
  }
}
```

(That same JSON drops straight into any `mcpServers`-style config — Claude Code,
Crush, etc.)

---

## 4. The alert bridge — pipeline webhook → Discord

The pipeline already POSTs `{stage_key, stage, verdict, metrics…}` to
`PIPELINE_WEBHOOK_URL` on every status write. Point a **second** webhook consumer
at a tiny bridge that, on **terminal** events, asks entheai to post to Discord.

`alert_bridge.py` (run it wherever entheai lives; expose it via your ngrok tunnel
as, e.g., `https://<your-domain>.ngrok.app/pipeline-alert`):

```python
#!/usr/bin/env python3
"""Pipeline webhook -> Discord alert via entheai's Discord MCP.
Only fires on terminal events (done / error), so you get a ping when it matters."""
import json, os, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = os.environ["ALERT_BRIDGE_TOKEN"]           # shared secret (matches PIPELINE_WEBHOOK_TOKEN)
CHANNEL = os.environ["DISCORD_ALERT_CHANNEL_ID"]

def _terminal(p):
    return p.get("stage") == "done" or p.get("verdict") in ("pass", "fail", "error", "killed")

def _alert(p):
    stage = p.get("stage_key", "?"); verdict = p.get("verdict") or p.get("stage")
    loss = (p.get("train_loss"));   repo = p.get("hf_repo")
    msg = f"**{stage.upper()} {verdict}**" + (f" · loss {loss}" if loss is not None else "")
    if repo: msg += f" · published `{repo}`"
    if p.get("error"): msg += f"\n```{str(p['error'])[:300]}```"
    # Hand it to entheai; the Discord MCP tool does the actual post.
    prompt = (f"Use the discord MCP to send this to channel {CHANNEL}, nothing else: {msg}")
    subprocess.run(["entheai", prompt], timeout=120, check=False)

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.headers.get("X-Webhook-Token") != TOKEN:
            self.send_response(403); self.end_headers(); return
        p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if _terminal(p): _alert(p)
        self.send_response(204); self.end_headers()
    def log_message(self, *a): pass

if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8791), H).serve_forever()
```

Expose it and point the pipeline at it (alongside the monitor):

```bash
# one ngrok agent can forward multiple ports/endpoints; simplest is a 2nd tunnel:
ngrok http 8791 --domain=<your-reserved>.ngrok.app          # paid: reserved, stable

# launch a run so the box POSTs terminal events to the bridge:
PIPELINE_WEBHOOK_URL="https://<your-reserved>.ngrok.app/pipeline-alert" \
PIPELINE_WEBHOOK_TOKEN="$ALERT_BRIDGE_TOKEN" \
  python controller.py --model … --family qwen …
```

Now: box finishes → POSTs `{stage:"done", verdict:"pass", hf_repo:…}` → bridge
sees a terminal event → entheai posts **"SFT PASS · loss 0.31 · published …"** to
`#pipeline-alerts`. Failures post the error tail too.

> **Simplest fallback (no MCP):** if you only ever want alerts (not an agent that
> can also *read/act* on Discord), skip the MCP and have `alert_bridge.py` POST
> directly to a **Discord channel webhook** URL (Channel → Edit → Integrations →
> Webhooks → New). One `urllib` POST, no bot. The MCP route is worth it when you
> want entheai to *do* things in Discord (triage threads, answer `!status`), not
> just fire-and-forget.

---

## 5. Test the whole chain

1. `entheai "use the discord MCP to send 'hello from entheai' to channel <ID>"`
   → message appears → **MCP wiring works**.
2. `curl -X POST http://127.0.0.1:8791/pipeline-alert -H "X-Webhook-Token: $ALERT_BRIDGE_TOKEN" -H 'Content-Type: application/json' -d '{"stage_key":"sft","stage":"done","verdict":"pass","train_loss":0.31,"hf_repo":"PeetPedro/qwen2.5-coder-32b-heretic-swe-sft"}'`
   → alert posts → **bridge → entheai → Discord works**.
3. Launch a real run with the two env vars set → you get pinged when it lands.

---

## Notes / gotchas

- **Secrets:** the bot token and `ALERT_BRIDGE_TOKEN` live only in gitignored
  `.env`. Never commit them (the mcp-config repo's `.gitignore` already excludes
  `.env`).
- **Bridge auth:** the bridge checks `X-Webhook-Token` — keep it so a public tunnel
  can't be spammed into pinging your channel.
- **Terminal-only:** the bridge fires only on done/verdict events, so you're not
  pinged for every `preparing_data → training` transition (those are for the live
  dashboard, not for alerts).
- **entheai availability:** the bridge shells out to `entheai`, so entheai must be
  installed on the bridge host. If you'd rather decouple, publish the pipeline
  event to **NATS** (`entheai.toml [nats] enabled = true`) and let a long-running
  entheai subscriber post to Discord — entheai already federates over NATS.
- **`[verify]`:** the Discord MCP package name, its env-var name for the token, and
  its exact tool names — confirm against the server's README before wiring.
