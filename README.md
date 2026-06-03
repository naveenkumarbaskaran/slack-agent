# slack-agent-ai

An AI-powered Slack bot that:

- **Answers questions** from a local directory of Markdown documentation.
- **Summarises threads** when asked.
- **Routes requests** to appropriate teams or channels.

Powered by [Anthropic Claude](https://www.anthropic.com/) (`claude-sonnet-4-6`) and the [Slack Bolt SDK](https://slack.dev/bolt-python/concepts).

---

## Quick start

### 1. Install

```bash
pip install slack-agent-ai
# or from source:
pip install -e .
```

### 2. Set environment variables

```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add your docs

Create a `./docs` directory and add Markdown (`.md`) files. The bot will use these as a knowledge base.

```
docs/
  onboarding.md
  api-reference.md
  runbooks/
    database.md
    deployments.md
```

### 4. Start the server

```bash
slack-agent serve --docs ./docs --port 3000
```

Or with explicit flags:

```bash
slack-agent serve \
  --docs ./docs \
  --port 3000 \
  --token xoxb-... \
  --signing-secret ... \
  --anthropic-key sk-ant-...
```

### 5. Configure your Slack app

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and open your app.
2. Under **Event Subscriptions**, enable events and set the Request URL to:
   ```
   http://<your-server>:3000/slack/events
   ```
3. Subscribe to these **bot events**:
   - `app_mention`
   - `message.im`
4. Add the following **OAuth scopes** under **Bot Token Scopes**:
   - `app_mentions:read`
   - `channels:history`
   - `chat:write`
   - `groups:history`
   - `im:history`
   - `im:write`
   - `reactions:write`
5. Reinstall the app to your workspace.

---

## Usage

Once running, mention the bot in any channel it has been invited to:

```
@YourBot how do I reset my API key?
@YourBot summarise this thread
@YourBot who handles database incidents?
```

You can also DM the bot directly (no @mention needed).

---

## Architecture

```
Slack Event
    │
    ▼
slack-bolt App (HTTP, port 3000)
    │  app_mention / message.im
    ▼
SlackAgent._handle_mention()
    │
    ▼
Anthropic Messages API (claude-sonnet-4-6)
    │   tool_use loop
    ├── search_docs(query, docs_dir)
    │       └── DocsIndex  (TF-IDF over ./docs/**/*.md)
    ├── get_thread(channel, ts)
    │       └── Slack conversations.replies API
    └── write_reply(channel, ts, text)
            └── Slack chat.postMessage API
```

### `DocsIndex`

A zero-dependency TF-IDF index over a directory tree of Markdown files.

- Walks the directory recursively for `*.md` files.
- Tokenises and indexes every document on startup.
- `search(query, top_k=5)` returns the most relevant documents.
- `get_snippet(entry, query)` extracts the most relevant passage from a document.
- `reload()` re-indexes the directory (useful if docs change at runtime).

### Agentic loop

The bot uses the Anthropic tool-use API in a standard while-loop:

1. User message is sent to Claude with the three tools defined.
2. Claude may call `search_docs` to look up documentation.
3. Claude may call `get_thread` to fetch message history for summarisation.
4. Claude may call `write_reply` to post additional messages.
5. Loop exits when Claude returns `end_turn`.
6. The final text block is sent back to Slack.

---

## Configuration reference

| Flag / Env var | Default | Description |
|---|---|---|
| `--docs` / — | `./docs` | Path to Markdown documentation directory |
| `--port` / — | `3000` | HTTP port for the Slack Events API |
| `--token` / `SLACK_BOT_TOKEN` | — | Slack bot token (`xoxb-...`) |
| `--signing-secret` / `SLACK_SIGNING_SECRET` | — | Slack signing secret |
| `--anthropic-key` / `ANTHROPIC_API_KEY` | — | Anthropic API key |

---

## Development

```bash
pip install -e '.[dev]'

# Lint
ruff check slack_agent/

# Type check
mypy slack_agent/

# Tests
pytest
```

### Using Socket Mode (no public URL needed)

For local development without exposing a public endpoint, install the Socket Mode extra:

```bash
pip install slack-bolt[async]  # already a transitive dep
```

Then swap the `start()` call in `agent.py` for:

```python
from slack_bolt.adapter.socket_mode import SocketModeHandler
SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

You need a `SLACK_APP_TOKEN` (`xapp-...`) with the `connections:write` scope.

---

## License

MIT
