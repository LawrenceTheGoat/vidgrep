#!/usr/bin/env python3
"""
ingest.py — Document ingestion pipeline for the RAG chatbot.

Loads .txt, .pdf, and .md files from the ./docs/ folder, splits them into
overlapping chunks, embeds them with sentence-transformers, and stores the
results in a persistent ChromaDB vector store.

Usage:
    python ingest.py           # ingest any new docs (skips existing collection)
    python ingest.py --reset   # wipe the collection and re-ingest everything
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

# ── Configuration ──────────────────────────────────────────────────────────────

DOCS_DIR = Path("./docs")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters (in characters; all-MiniLM-L6-v2 handles ~190 words max,
# so 500-char chunks are comfortably within its context window).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

console = Console()


# ── Loaders ───────────────────────────────────────────────────────────────────

# Map file suffixes to their LangChain loader class.
LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_documents(docs_dir: Path) -> List[Document]:
    """
    Walk docs_dir and load every supported file into LangChain Document objects.

    Each Document carries page_content (raw text) and metadata including the
    source filename, which is used later for source citations.

    Args:
        docs_dir: Path to the directory containing source documents.

    Returns:
        A flat list of Document objects from all loaded files.

    Raises:
        SystemExit: If the docs directory doesn't exist or contains no supported files.
    """
    if not docs_dir.exists():
        console.print(f"[bold red]Error:[/] Docs directory '{docs_dir}' does not exist.")
        console.print("Create it and add .txt, .pdf, or .md files, then re-run.")
        sys.exit(1)

    # Gather all files with a supported extension.
    supported_files = [
        f for f in sorted(docs_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in LOADER_MAP
    ]

    if not supported_files:
        console.print(
            f"[bold red]Error:[/] No supported files (.txt, .pdf, .md) found in '{docs_dir}'."
        )
        sys.exit(1)

    all_docs: List[Document] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading documents...", total=len(supported_files))

        for file_path in supported_files:
            suffix = file_path.suffix.lower()
            loader_cls = LOADER_MAP[suffix]

            try:
                # TextLoader needs an explicit encoding to handle UTF-8 docs.
                if loader_cls is TextLoader:
                    loader = loader_cls(str(file_path), encoding="utf-8")
                else:
                    loader = loader_cls(str(file_path))

                docs = loader.load()

                # Normalise the source metadata to just the filename so that
                # citations are readable (not an absolute machine-specific path).
                for doc in docs:
                    doc.metadata["source"] = file_path.name

                all_docs.extend(docs)
                progress.update(task, advance=1, description=f"[cyan]Loaded:[/] {file_path.name}")

            except Exception as exc:
                console.print(f"[yellow]Warning:[/] Could not load '{file_path.name}': {exc}")
                progress.update(task, advance=1)

    return all_docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to break on paragraph boundaries first,
    then sentences, then words — preserving as much semantic coherence as possible.

    Args:
        documents: Raw Document objects loaded from disk.

    Returns:
        A list of smaller Document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Prefer breaking at paragraph/sentence/word boundaries in that order.
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ── Embedding & Storage ───────────────────────────────────────────────────────

def build_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initialise the sentence-transformers embedding model.

    The model is downloaded once and cached locally by the sentence-transformers
    library (~22 MB). Subsequent calls load from cache instantly.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings instance.
    """
    console.print(f"[dim]Loading embedding model '{EMBEDDING_MODEL}'...[/]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def store_chunks(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    chroma_dir: Path,
    reset: bool,
) -> Chroma:
    """
    Embed chunks and persist them in ChromaDB.

    If reset=True the existing collection is wiped first. Otherwise, chunks are
    added incrementally (useful for ingesting new documents without re-processing
    the entire corpus).

    Args:
        chunks:     Document chunks to embed and store.
        embeddings: Embedding function to use.
        chroma_dir: Path where ChromaDB persists its files.
        reset:      If True, delete and recreate the collection.

    Returns:
        The populated Chroma vector store instance.
    """
    if reset and chroma_dir.exists():
        console.print("[yellow]--reset flag detected — wiping existing vector store...[/]")
        shutil.rmtree(chroma_dir)

    chroma_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[bold green]Embedding and storing chunks...[/]", spinner="dots"):
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_dir),
            collection_name=COLLECTION_NAME,
        )

    return vector_store


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG chatbot's ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB collection before ingesting.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help=f"Directory containing source documents (default: {DOCS_DIR}).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_DIR,
        help=f"ChromaDB persistence directory (default: {CHROMA_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(
        Panel(
            Text("RAG Document Ingestion Pipeline", justify="center", style="bold cyan"),
            subtitle="[dim]Lawrence Wang · UWaterloo Systems Design Engineering[/]",
        )
    )

    # 1. Load raw documents from disk.
    console.print("\n[bold]Step 1/3:[/] Loading documents from disk...")
    raw_docs = load_documents(args.docs_dir)
    console.print(f"  [green]✓[/] Loaded [bold]{len(raw_docs)}[/] page(s) / document(s).")

    # 2. Split into overlapping chunks.
    console.print("\n[bold]Step 2/3:[/] Splitting into chunks...")
    chunks = split_documents(raw_docs)
    console.print(
        f"  [green]✓[/] Created [bold]{len(chunks)}[/] chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    # 3. Embed and store in ChromaDB.
    console.print("\n[bold]Step 3/3:[/] Embedding and storing in ChromaDB...")
    embeddings = build_embedding_function()
    store_chunks(chunks, embeddings, args.chroma_dir, reset=args.reset)

    console.print(
        Panel(
            f"[bold green]Ingestion complete![/]\n\n"
            f"  Documents processed : {len(raw_docs)}\n"
            f"  Chunks stored       : {len(chunks)}\n"
            f"  Vector store path   : {args.chroma_dir.resolve()}\n\n"
            f"Run [bold cyan]python chat.py[/] to start chatting.",
            title="[bold]Summary[/]",
        )
    )


if __name__ == "__main__":
    main()
