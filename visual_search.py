#!/usr/bin/env python3
"""
visual_search.py — Interactive search over a multimodal video index.

Given a natural-language query, retrieves the most relevant moments across
all ingested YouTube videos and shows:
  • the matched timestamp (with a clickable youtu.be deep link)
  • the visual caption (from moondream)
  • the surrounding transcript
  • the path to the saved frame thumbnail
  • a relevance score

Usage:
    python visual_search.py                       # interactive REPL
    python visual_search.py -q "dovetail joint"   # single-shot query
    python visual_search.py -q "..." -k 8         # top-K control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "youtube_visual_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

console = Console()


# ── Loading ───────────────────────────────────────────────────────────────────


def load_store() -> Chroma:
    """Open the ChromaDB collection populated by youtube_visual_ingest.py."""
    if not CHROMA_DIR.exists():
        console.print(
            Panel(
                f"[bold red]No vector store found at {CHROMA_DIR}.[/]\n\n"
                f"Run: [cyan]python youtube_visual_ingest.py <YOUTUBE_URL>[/] first.",
                title="[bold red]Not initialised[/]",
                border_style="red",
            )
        )
        sys.exit(1)

    console.print(f"[dim]Loading embedding model '{EMBEDDING_MODEL}'...[/]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


# ── Search ────────────────────────────────────────────────────────────────────


def search(store: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    """Return the top-K (Document, relevance_score) results for `query`.

    `similarity_search_with_relevance_scores` returns scores in [0, 1] where 1
    is most relevant — easier to interpret than raw distances.
    """
    return store.similarity_search_with_relevance_scores(query, k=k)


# ── Rendering ─────────────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    """Render seconds as M:SS (or H:MM:SS for long videos)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def render_results(results: List[Tuple[Document, float]], query: str) -> None:
    """Pretty-print the matched moments in a Rich table + per-hit detail panels."""
    if not results:
        console.print("[yellow]No matches found.[/]")
        return

    # Compact table for at-a-glance comparison.
    table = Table(
        title=f"Top {len(results)} matches for: [italic]{query}[/]",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Time", width=8)
    table.add_column("Video", overflow="fold")
    table.add_column("Visual caption", overflow="fold")

    for i, (doc, score) in enumerate(results, start=1):
        ts = format_timestamp(doc.metadata.get("timestamp", 0))
        title = doc.metadata.get("video_title") or doc.metadata.get("video_id", "?")
        caption = doc.metadata.get("caption", "").strip()
        table.add_row(
            str(i),
            f"{score:.2f}",
            ts,
            title[:40] + ("…" if len(title) > 40 else ""),
            caption[:80] + ("…" if len(caption) > 80 else ""),
        )

    console.print(table)

    # Detailed per-hit panels with the clickable link, full caption + transcript.
    for i, (doc, score) in enumerate(results, start=1):
        md = doc.metadata
        body = Text()
        body.append("📺 ", style="bold")
        body.append(f"{md.get('video_title') or md.get('video_id')}\n", style="bold")
        body.append("⏱  Timestamp: ", style="bold")
        body.append(f"{format_timestamp(md.get('timestamp', 0))}  ", style="cyan")
        body.append(f"(score {score:.3f})\n", style="dim")
        body.append("🔗 ", style="bold")
        body.append(f"{md.get('youtube_link', '')}\n", style="blue underline")
        body.append("🖼  Frame: ", style="bold")
        body.append(f"{md.get('frame_path', '')}\n", style="dim")
        body.append("\n👁  Visual: ", style="bold")
        body.append(f"{md.get('caption', '').strip()}\n")
        body.append("\n🗣  Transcript: ", style="bold")
        body.append(md.get("transcript", "").strip() or "(none)")

        console.print(
            Panel(
                body,
                title=f"[bold cyan]Match #{i}[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )


# ── REPL ──────────────────────────────────────────────────────────────────────


def interactive_loop(store: Chroma, k: int) -> None:
    """Simple read-query-print loop."""
    console.print(
        Panel(
            "[bold]Visual RAG Search[/] — type a natural-language query and hit Enter.\n"
            "Examples:\n"
            "  [dim]• \"the moment they show the circuit board\"[/]\n"
            "  [dim]• \"someone holding a soldering iron\"[/]\n"
            "  [dim]• \"diagram of the flight controller\"[/]\n\n"
            "Type [bold]exit[/] or press [bold]Ctrl-D[/] to quit.",
            title="[bold cyan]Multimodal Search[/]",
            border_style="cyan",
        )
    )

    while True:
        try:
            query = console.input("\n[bold green]search>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/]")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]bye![/]")
            return

        results = search(store, query, k)
        render_results(results, query)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the visual RAG index for matching video moments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-q", "--query", help="One-shot query (skips the REPL).")
    parser.add_argument(
        "-k", "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = load_store()

    if args.query:
        results = search(store, args.query, args.top_k)
        render_results(results, args.query)
    else:
        interactive_loop(store, args.top_k)


if __name__ == "__main__":
    main()
