"""Command-line interface for slack-agent.

Usage examples
--------------
# Start the server with explicit flags:
    slack-agent serve --docs ./docs --port 3000 --token xoxb-...

# Start using environment variables:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_SIGNING_SECRET=...
    export ANTHROPIC_API_KEY=sk-ant-...
    slack-agent serve --docs ./docs
"""

from __future__ import annotations

import logging
import os
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(package_name="slack-agent-ai")
def cli() -> None:
    """Slack Agent — AI-powered Slack bot backed by local docs."""


@cli.command()
@click.option(
    "--docs",
    "docs_dir",
    default="./docs",
    show_default=True,
    help="Path to the directory containing Markdown documentation files.",
    type=click.Path(file_okay=False),
)
@click.option(
    "--port",
    default=3000,
    show_default=True,
    help="HTTP port the Bolt app listens on.",
    type=int,
)
@click.option(
    "--token",
    "slack_bot_token",
    default=None,
    help="Slack bot token (xoxb-...). Defaults to $SLACK_BOT_TOKEN.",
    envvar="SLACK_BOT_TOKEN",
)
@click.option(
    "--signing-secret",
    "slack_signing_secret",
    default=None,
    help="Slack signing secret. Defaults to $SLACK_SIGNING_SECRET.",
    envvar="SLACK_SIGNING_SECRET",
)
@click.option(
    "--anthropic-key",
    "anthropic_api_key",
    default=None,
    help="Anthropic API key. Defaults to $ANTHROPIC_API_KEY.",
    envvar="ANTHROPIC_API_KEY",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable debug logging.")
def serve(
    docs_dir: str,
    port: int,
    slack_bot_token: str | None,
    slack_signing_secret: str | None,
    anthropic_api_key: str | None,
    verbose: bool,
) -> None:
    """Start the Slack Agent HTTP server.

    Connects to Slack via the Events API (HTTP mode). You must configure
    your Slack app to forward events to http://<host>:<port>/slack/events.
    """
    _setup_logging(verbose)
    log = logging.getLogger("slack_agent")

    # Validate required credentials
    token = slack_bot_token or os.environ.get("SLACK_BOT_TOKEN")
    secret = slack_signing_secret or os.environ.get("SLACK_SIGNING_SECRET")
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    missing = []
    if not token:
        missing.append("SLACK_BOT_TOKEN (--token)")
    if not secret:
        missing.append("SLACK_SIGNING_SECRET (--signing-secret)")
    if not api_key:
        missing.append("ANTHROPIC_API_KEY (--anthropic-key)")

    if missing:
        console.print(
            "[bold red]Error:[/bold red] The following required values are not set:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
        sys.exit(1)

    console.print(f"[bold green]Starting Slack Agent[/bold green]")
    console.print(f"  Docs directory : [cyan]{docs_dir}[/cyan]")
    console.print(f"  Listening on   : [cyan]http://0.0.0.0:{port}/slack/events[/cyan]")
    console.print(f"  Model          : [cyan]claude-sonnet-4-6[/cyan]")

    # Lazy import to keep CLI startup fast
    from slack_agent.agent import SlackAgent  # noqa: PLC0415

    agent = SlackAgent(
        docs_dir=docs_dir,
        slack_bot_token=token,
        slack_signing_secret=secret,
        anthropic_api_key=api_key,
    )

    log.info("Docs index: %d documents loaded from %s", len(agent.docs_index), docs_dir)

    try:
        agent.start(port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")


@cli.command("reindex")
@click.option(
    "--docs",
    "docs_dir",
    default="./docs",
    show_default=True,
    help="Path to the docs directory to index.",
    type=click.Path(file_okay=False),
)
def reindex(docs_dir: str) -> None:
    """Print a summary of documents that would be indexed from DOCS_DIR."""
    _setup_logging(verbose=False)
    from slack_agent.docs_index import DocsIndex  # noqa: PLC0415

    index = DocsIndex(docs_dir)
    if len(index) == 0:
        console.print(f"[yellow]No Markdown files found in {docs_dir!r}.[/yellow]")
        return

    console.print(f"[bold]Indexed {len(index)} document(s) from {docs_dir!r}:[/bold]")
    for entry in index._docs:
        console.print(f"  [cyan]{entry.title}[/cyan]  ({entry.path})")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
