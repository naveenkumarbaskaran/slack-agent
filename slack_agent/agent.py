"""SlackAgent: Anthropic-powered Slack bot using slack-bolt.

Features
--------
* Answers questions sourced from a local docs directory (TF-IDF retrieval
  + Claude synthesis).
* Summarizes long Slack threads on request.
* Routes requests to appropriate teams / channels.
* Runs as a Slack Bolt HTTP app (webhook mode).

Environment variables required
------------------------------
SLACK_BOT_TOKEN   - xoxb-... bot token
SLACK_SIGNING_SECRET - Slack app signing secret
ANTHROPIC_API_KEY  - Anthropic API key (or set via client)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic
from slack_bolt import App
from slack_sdk import WebClient

from slack_agent.docs_index import DocsIndex

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# Number of docs to retrieve and include in the context.
DOCS_TOP_K = 4
# Maximum characters per doc snippet sent to the model.
SNIPPET_CHARS = 600
# Maximum Slack messages fetched for thread summarisation.
THREAD_HISTORY_LIMIT = 100

SYSTEM_PROMPT = """\
You are a helpful Slack bot assistant. You have access to tools:

1. search_docs(query, docs_dir) - search the local documentation index for relevant passages.
2. get_thread(channel, ts) - fetch the full message history of a Slack thread.
3. write_reply(channel, ts, text) - post a reply message to a Slack thread.

Guidelines:
- Answer questions concisely and accurately.
- When a user asks to summarise a thread, call get_thread then provide a clear summary.
- When routing a request, identify the right team / channel and include it in your reply.
- Cite the document title when drawing from documentation.
- If you are not confident, say so rather than making things up.
- Keep Slack formatting in mind: use *bold*, _italic_, and `code` where helpful.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": (
            "Search the local documentation directory for content relevant to a query. "
            "Returns a list of matching document titles and relevant excerpts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'how to reset a password'",
                },
                "docs_dir": {
                    "type": "string",
                    "description": "Path to the docs directory (e.g. './docs'). Use the configured default if unsure.",
                },
            },
            "required": ["query", "docs_dir"],
        },
    },
    {
        "name": "get_thread",
        "description": (
            "Fetch all messages in a Slack thread. "
            "Returns a JSON array of {user, text, ts} objects."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Slack channel ID (e.g. C01234567)",
                },
                "ts": {
                    "type": "string",
                    "description": "Thread timestamp / parent message ts",
                },
            },
            "required": ["channel", "ts"],
        },
    },
    {
        "name": "write_reply",
        "description": "Post a reply message to a Slack thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Slack channel ID",
                },
                "ts": {
                    "type": "string",
                    "description": "Thread timestamp to reply to",
                },
                "text": {
                    "type": "string",
                    "description": "Message text (Slack mrkdwn formatting supported)",
                },
            },
            "required": ["channel", "ts", "text"],
        },
    },
]


class SlackAgent:
    """Manages the Slack Bolt app and the Anthropic agentic loop."""

    def __init__(
        self,
        docs_dir: str = "./docs",
        slack_bot_token: str | None = None,
        slack_signing_secret: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.docs_dir = docs_dir
        self.docs_index = DocsIndex(docs_dir)
        logger.info("Docs index built: %d documents indexed from %s", len(self.docs_index), docs_dir)

        # Anthropic client
        self.anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        # Slack Bolt app
        self.bolt_app = App(
            token=slack_bot_token or os.environ["SLACK_BOT_TOKEN"],
            signing_secret=slack_signing_secret or os.environ["SLACK_SIGNING_SECRET"],
        )
        self._slack_client: WebClient = self.bolt_app.client

        # Register event listeners
        self._register_listeners()

    # ------------------------------------------------------------------
    # Slack event listeners
    # ------------------------------------------------------------------

    def _register_listeners(self) -> None:
        """Attach all Bolt event / action listeners."""

        @self.bolt_app.event("app_mention")
        def on_mention(event: dict, say: Any) -> None:  # type: ignore[type-arg]
            """Respond to @mentions in channels."""
            self._handle_mention(event, say)

        @self.bolt_app.event("message")
        def on_dm(event: dict, say: Any) -> None:  # type: ignore[type-arg]
            """Respond to direct messages."""
            # Only handle DMs (channel_type == 'im'), skip channel messages
            # that are not mentions (handled above).
            if event.get("channel_type") == "im" and "bot_id" not in event:
                self._handle_mention(event, say)

    def _handle_mention(self, event: dict, say: Any) -> None:  # type: ignore[type-arg]
        """Core handler: strip the mention, run the agentic loop, reply."""
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        raw_text: str = event.get("text", "")

        # Strip bot mention tokens like <@U01234567>
        user_text = re.sub(r"<@[A-Z0-9]+>", "", raw_text).strip()
        if not user_text:
            say(
                text="Hi! How can I help? Try asking me a question or say *summarise this thread*.",
                thread_ts=thread_ts,
            )
            return

        logger.info("Received mention: channel=%s ts=%s text=%r", channel, thread_ts, user_text)

        try:
            reply = self._run_agent(
                user_text=user_text,
                channel=channel,
                thread_ts=thread_ts,
            )
        except Exception:
            logger.exception("Agent error for message %r", user_text)
            reply = ":warning: Sorry, something went wrong. Please try again."

        say(text=reply, thread_ts=thread_ts)

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        user_text: str,
        channel: str,
        thread_ts: str,
    ) -> str:
        """Run the Anthropic agentic tool-use loop and return the final text reply."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_text}
        ]

        while True:
            response = self.anthropic_client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            # Append assistant response to history
            messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

            if response.stop_reason == "end_turn":
                # Extract the final text block
                for block in response.content:
                    if hasattr(block, "type") and block.type == "text":
                        return block.text
                return "(no reply)"

            if response.stop_reason == "tool_use":
                tool_results = self._execute_tool_calls(
                    response.content, channel=channel, thread_ts=thread_ts
                )
                messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]
                continue

            # Unexpected stop reason — bail out
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            return "I encountered an unexpected issue. Please try again."

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool_calls(
        self,
        content_blocks: list[Any],
        channel: str,
        thread_ts: str,
    ) -> list[dict[str, Any]]:
        """Execute every tool_use block and return tool_result blocks."""
        results = []
        for block in content_blocks:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue
            tool_name: str = block.name
            tool_input: dict[str, Any] = block.input
            tool_use_id: str = block.id

            logger.info("Calling tool %r with %r", tool_name, tool_input)
            try:
                output = self._dispatch_tool(tool_name, tool_input, channel, thread_ts)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool %r raised an error", tool_name)
                output = f"Error executing {tool_name}: {exc}"
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output,
                        "is_error": True,
                    }
                )
                continue

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                }
            )
        return results

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        channel: str,
        thread_ts: str,
    ) -> str:
        if tool_name == "search_docs":
            return self._tool_search_docs(
                query=tool_input["query"],
                docs_dir=tool_input.get("docs_dir", self.docs_dir),
            )
        if tool_name == "get_thread":
            return self._tool_get_thread(
                channel=tool_input.get("channel", channel),
                ts=tool_input.get("ts", thread_ts),
            )
        if tool_name == "write_reply":
            return self._tool_write_reply(
                channel=tool_input.get("channel", channel),
                ts=tool_input.get("ts", thread_ts),
                text=tool_input["text"],
            )
        raise ValueError(f"Unknown tool: {tool_name!r}")

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_search_docs(self, query: str, docs_dir: str) -> str:
        """Search the docs index and return relevant snippets as JSON."""
        # Re-use the existing index if the path matches; otherwise build a
        # temporary one. (Hot-path: same docs_dir is the common case.)
        if str(docs_dir) != str(self.docs_dir):
            index = DocsIndex(docs_dir)
        else:
            index = self.docs_index

        if len(index) == 0:
            return json.dumps({"results": [], "note": f"No documents found in {docs_dir!r}."})

        entries = index.search(query, top_k=DOCS_TOP_K)
        if not entries:
            return json.dumps({"results": [], "note": "No relevant documents found."})

        results = []
        for entry in entries:
            snippet = index.get_snippet(entry, query, max_chars=SNIPPET_CHARS)
            results.append(
                {
                    "title": entry.title,
                    "path": entry.path,
                    "snippet": snippet,
                }
            )
        return json.dumps({"results": results}, ensure_ascii=False)

    def _tool_get_thread(self, channel: str, ts: str) -> str:
        """Fetch thread messages from Slack and return them as JSON."""
        try:
            resp = self._slack_client.conversations_replies(
                channel=channel,
                ts=ts,
                limit=THREAD_HISTORY_LIMIT,
            )
        except Exception as exc:
            raise RuntimeError(f"conversations_replies failed: {exc}") from exc

        messages = resp.get("messages", [])
        thread_data = [
            {
                "user": msg.get("user", msg.get("bot_id", "unknown")),
                "text": msg.get("text", ""),
                "ts": msg.get("ts", ""),
            }
            for msg in messages
        ]
        return json.dumps({"messages": thread_data, "count": len(thread_data)}, ensure_ascii=False)

    def _tool_write_reply(self, channel: str, ts: str, text: str) -> str:
        """Post a reply to a Slack thread and return the message ts."""
        try:
            resp = self._slack_client.chat_postMessage(
                channel=channel,
                thread_ts=ts,
                text=text,
            )
        except Exception as exc:
            raise RuntimeError(f"chat_postMessage failed: {exc}") from exc

        new_ts = resp.get("ts", "unknown")
        return json.dumps({"ok": True, "ts": new_ts})

    # ------------------------------------------------------------------
    # Start the HTTP server
    # ------------------------------------------------------------------

    def start(self, port: int = 3000) -> None:
        """Start the Bolt HTTP server on *port*."""
        logger.info("Starting SlackAgent on port %d", port)
        self.bolt_app.start(port=port)
