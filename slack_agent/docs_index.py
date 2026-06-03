"""Simple TF-IDF index over a directory of Markdown files."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class DocEntry:
    path: str
    title: str
    content: str
    terms: Counter = field(default_factory=Counter, repr=False)


class DocsIndex:
    """TF-IDF index over a directory of Markdown (.md) files.

    Usage::

        idx = DocsIndex("./docs")
        results = idx.search("how to authenticate", top_k=3)
        for r in results:
            print(r.title, r.path)
    """

    def __init__(self, docs_dir: str | Path) -> None:
        self.docs_dir = Path(docs_dir)
        self._docs: list[DocEntry] = []
        self._df: Counter = Counter()  # document frequency per term
        self._build()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _build(self) -> None:
        """Walk docs_dir, parse Markdown files, build the index."""
        self._docs.clear()
        self._df.clear()

        for md_path in self._iter_markdown():
            content = md_path.read_text(encoding="utf-8", errors="replace")
            title = self._extract_title(content, md_path)
            tokens = self._tokenize(content)
            term_counts = Counter(tokens)
            entry = DocEntry(
                path=str(md_path),
                title=title,
                content=content,
                terms=term_counts,
            )
            self._docs.append(entry)
            # Update document frequency
            for term in set(tokens):
                self._df[term] += 1

    def _iter_markdown(self) -> Iterator[Path]:
        if not self.docs_dir.exists():
            return
        for root, _dirs, files in os.walk(self.docs_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".md"):
                    yield Path(root) / fname

    @staticmethod
    def _extract_title(content: str, path: Path) -> str:
        """Return the first H1 heading, or fall back to the filename stem."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return path.stem.replace("_", " ").replace("-", " ").title()

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[DocEntry]:
        """Return the *top_k* most relevant docs for *query* using TF-IDF cosine similarity."""
        if not self._docs:
            return []

        n_docs = len(self._docs)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # IDF for query terms
        query_idf: dict[str, float] = {}
        for term in set(query_tokens):
            df = self._df.get(term, 0)
            query_idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

        scores: list[tuple[float, DocEntry]] = []
        for entry in self._docs:
            score = 0.0
            total_terms = sum(entry.terms.values()) or 1
            for term in set(query_tokens):
                tf = entry.terms.get(term, 0) / total_terms
                score += tf * query_idf[term]
            if score > 0:
                scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scores[:top_k]]

    def get_snippet(self, entry: DocEntry, query: str, max_chars: int = 800) -> str:
        """Return a snippet from *entry* that is most relevant to *query*."""
        query_terms = set(self._tokenize(query))
        lines = entry.content.splitlines()
        best_score = -1
        best_start = 0
        window = 20  # lines
        for i in range(len(lines)):
            chunk_lines = lines[i : i + window]
            chunk = " ".join(chunk_lines).lower()
            score = sum(1 for t in query_terms if t in chunk)
            if score > best_score:
                best_score = score
                best_start = i
        snippet = "\n".join(lines[best_start : best_start + window])
        return snippet[:max_chars]

    def reload(self) -> None:
        """Re-index the docs directory (call after adding/changing files)."""
        self._build()

    def __len__(self) -> int:
        return len(self._docs)
