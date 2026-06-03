"""Slack Agent — answers questions from docs, summarizes threads, routes requests."""

from slack_agent.agent import SlackAgent
from slack_agent.docs_index import DocsIndex

__all__ = ["SlackAgent", "DocsIndex"]
