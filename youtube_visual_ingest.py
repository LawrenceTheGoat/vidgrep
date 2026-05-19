#!/usr/bin/env python3
"""
youtube_visual_ingest.py — Multimodal ingestion pipeline for YouTube videos.

Pipeline:
  1. yt-dlp downloads the video + auto-generated subtitles.
  2. ffmpeg extracts one frame every N seconds.
  3. Ollama (moondream model) captions each frame.
  4. Captions are aligned with the transcript segment at the same timestamp.
  5. The combined "visual + audio" text is embedded with sentence-transformers
     and persisted to ChromaDB with rich metadata (timestamp, frame path, etc.).

Usage:
    python youtube_visual_ingest.py <YOUTUBE_URL>
    python youtube_visual_ingest.py <URL> --frame-interval 3 --reset
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

# ── Configuration ──────────────────────────────────────────────────────────────

VIDEOS_DIR = Path("./videos")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "youtube_visual_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "moondream"

# Sample one frame every N seconds. 5s is a good speed/coverage tradeoff;
# tutorials with fast-changing content may want 3s, talking-head videos can use 10s.
FRAME_INTERVAL_SECONDS = 5

# Prompt sent to the vision model. Moondream is small and responds best to
# very simple prompts — elaborate instructions cause it to return empty output.
CAPTION_PROMPT = "What is in this image?"

console = Console()


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class TranscriptSegment:
    """A single timestamped chunk of speech from the video's captions."""

    start: float
    duration: float
    text: str


@dataclass
class Frame:
    """A single extracted frame, paired with its timestamp."""

    timestamp: float
    path: Path


# ── Helpers: yt-dlp + ffmpeg ──────────────────────────────────────────────────


def require_binary(name: str) -> None:
    """Exit with a friendly error if `name` is not on PATH."""
    if shutil.which(name) is None:
        console.print(
            f"[bold red]Error:[/] '{name}' is not installed. "
            f"Run [cyan]brew install {name}[/] and try again."
        )
        sys.exit(1)


def extract_video_id(url: str) -> str:
    """Pull the 11-character video ID out of any standard YouTube URL form."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    console.print(f"[bold red]Error:[/] Could not parse a YouTube video ID from '{url}'.")
    sys.exit(1)


def download_video(url: str, work_dir: Path) -> Path:
    """
    Use yt-dlp to download the video at a modest resolution.

    480p is plenty for vision captioning and keeps disk usage in check
    (typical 10-minute video at 480p ≈ 80-150 MB).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(work_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "best[height<=480][ext=mp4]/best[height<=480]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--progress",
        url,
    ]

    with console.status("[bold cyan]Downloading video with yt-dlp...[/]", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[bold red]yt-dlp failed:[/]\n{result.stderr}")
        sys.exit(1)

    # Find the actual downloaded file (yt-dlp may have picked .webm / .mkv).
    for candidate in work_dir.glob("video.*"):
        if candidate.suffix in {".mp4", ".webm", ".mkv"}:
            return candidate

    console.print("[bold red]Error:[/] yt-dlp completed but no video file was found.")
    sys.exit(1)


def extract_frames(video_path: Path, frames_dir: Path, interval: int) -> List[Frame]:
    """
    Use ffmpeg to extract one frame every `interval` seconds.

    Frames are saved as JPEG at 480px width (height preserved by aspect ratio)
    to keep them small for the vision model — moondream doesn't benefit from
    very high resolution and smaller frames mean faster captioning.
    """
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(frames_dir / "frame_%05d.jpg")

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval},scale=480:-1",
        "-q:v",
        "5",  # Quality 1 (best) - 31 (worst); 5 is a good size/quality balance.
        output_pattern,
    ]

    with console.status("[bold cyan]Extracting frames with ffmpeg...[/]", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[bold red]ffmpeg failed:[/]\n{result.stderr}")
        sys.exit(1)

    # The Nth frame (1-indexed) corresponds to timestamp (N-1) * interval.
    frames: List[Frame] = []
    for frame_path in sorted(frames_dir.glob("frame_*.jpg")):
        # frame_00001.jpg → index 1 → timestamp 0
        idx = int(frame_path.stem.split("_")[1])
        timestamp = (idx - 1) * interval
        frames.append(Frame(timestamp=float(timestamp), path=frame_path))

    return frames


# ── Helpers: transcript ───────────────────────────────────────────────────────


def fetch_transcript(video_id: str) -> List[TranscriptSegment]:
    """
    Pull auto-generated or manually uploaded captions via youtube-transcript-api.

    Uses the v1.x API (`YouTubeTranscriptApi().fetch()`). Falls back through:
      1. Try the configured preferred languages directly.
      2. If none match, list available transcripts and pick the first generated
         one, then translate to English if possible.
      3. As a last resort, return whatever the first available transcript is.

    If a video has no captions at all we return an empty list and continue —
    the system will still work using only visual captions.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        console.print(
            "[bold red]Missing dependency:[/] youtube-transcript-api is not installed. "
            "Run [cyan]pip install youtube-transcript-api[/]."
        )
        sys.exit(1)

    api = YouTubeTranscriptApi()
    preferred_langs = ["en", "en-US", "en-GB"]
    snippets = None

    # Attempt 1: direct fetch in preferred languages (handles English videos quickly).
    try:
        fetched = api.fetch(video_id, languages=preferred_langs)
        snippets = list(fetched)
    except Exception as direct_exc:
        # Attempt 2 / 3: enumerate available transcripts and pick a sensible one,
        # translating to English when the original is non-English.
        try:
            transcript_list = api.list(video_id)
            chosen = None

            # Prefer manually-created English, then auto English, then anything translatable.
            for t in transcript_list:
                if t.language_code in preferred_langs and not t.is_generated:
                    chosen = t
                    break
            if chosen is None:
                for t in transcript_list:
                    if t.language_code in preferred_langs:
                        chosen = t
                        break
            if chosen is None:
                for t in transcript_list:
                    if t.is_translatable:
                        try:
                            chosen = t.translate("en")
                        except Exception:
                            continue
                        break
            if chosen is None:
                # Last resort: take whatever is first.
                for t in transcript_list:
                    chosen = t
                    break

            if chosen is not None:
                snippets = list(chosen.fetch())
        except Exception as list_exc:
            console.print(
                f"[yellow]Warning:[/] Could not fetch transcript for {video_id}: "
                f"{type(direct_exc).__name__}: {direct_exc}"
            )
            console.print("[yellow]Proceeding with visual captions only.[/]")
            return []

    if not snippets:
        console.print(
            f"[yellow]Warning:[/] No transcript snippets available for {video_id}."
        )
        return []

    # The v1.x API returns FetchedTranscriptSnippet objects with `.text`, `.start`,
    # `.duration` attributes (no longer dicts).
    return [
        TranscriptSegment(
            start=float(s.start),
            duration=float(getattr(s, "duration", 0.0)),
            text=(s.text or "").replace("\n", " ").strip(),
        )
        for s in snippets
    ]


def transcript_for_window(
    segments: List[TranscriptSegment],
    center: float,
    half_window: float,
) -> str:
    """
    Concatenate transcript segments that overlap [center - half_window, center + half_window].

    This gives us the spoken context around a given frame's timestamp, which is
    much more useful for retrieval than just the segment at the exact second.
    """
    lo = center - half_window
    hi = center + half_window
    parts = [
        seg.text
        for seg in segments
        if (seg.start + seg.duration) >= lo and seg.start <= hi
    ]
    return " ".join(parts).strip()


# ── Helpers: Ollama vision ────────────────────────────────────────────────────


def check_ollama_ready() -> None:
    """Verify the Ollama daemon is running and the vision model is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
    except Exception:
        console.print(
            Panel(
                "[bold red]Ollama is not running.[/]\n\n"
                "Start it with: [cyan]ollama serve[/]  (in a separate terminal)\n"
                "Or as a background service: [cyan]brew services start ollama[/]",
                title="[bold red]Setup required[/]",
                border_style="red",
            )
        )
        sys.exit(1)

    available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    if VISION_MODEL not in available:
        console.print(
            Panel(
                f"[bold red]Model '{VISION_MODEL}' is not pulled.[/]\n\n"
                f"Run: [cyan]ollama pull {VISION_MODEL}[/]",
                title="[bold red]Setup required[/]",
                border_style="red",
            )
        )
        sys.exit(1)


def caption_frame(frame_path: Path) -> str:
    """Call Ollama's /api/generate with the frame as a base64 image and return the caption."""
    with frame_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "prompt": CAPTION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {
            # 150 tokens fits a 2-3 sentence description; longer hurts signal-to-noise.
            "num_predict": 150,
            "temperature": 0.2,
        },
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        return f"[caption failed: {exc}]"

    return resp.json().get("response", "").strip()


# ── Storage ───────────────────────────────────────────────────────────────────


def build_documents(
    frames: List[Frame],
    captions: List[str],
    transcript_segments: List[TranscriptSegment],
    video_id: str,
    video_url: str,
    video_title: Optional[str],
    half_window: float,
) -> List[Document]:
    """Combine each frame's caption + surrounding transcript into a Document."""
    docs: List[Document] = []
    for frame, caption in zip(frames, captions):
        spoken = transcript_for_window(transcript_segments, frame.timestamp, half_window)

        # The text we embed contains both signals so retrieval can hit on either.
        # The labels also help the LLM (in any downstream chat) understand context.
        content = (
            f"[Visual at {frame.timestamp:.0f}s]: {caption}\n"
            f"[Spoken near {frame.timestamp:.0f}s]: {spoken or '(no transcript)'}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "video_id": video_id,
                    "video_url": video_url,
                    "video_title": video_title or video_id,
                    "timestamp": float(frame.timestamp),
                    "frame_path": str(frame.path.resolve()),
                    "caption": caption,
                    "transcript": spoken,
                    "youtube_link": f"https://youtu.be/{video_id}?t={int(frame.timestamp)}",
                },
            )
        )
    return docs


def store_documents(docs: List[Document], chroma_dir: Path, reset: bool) -> None:
    """Embed and persist documents into the shared ChromaDB collection."""
    if reset and chroma_dir.exists():
        console.print("[yellow]--reset flag detected — wiping existing vector store...[/]")
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Loading embedding model '{EMBEDDING_MODEL}'...[/]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    with console.status("[bold green]Embedding and storing chunks...[/]", spinner="dots"):
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(chroma_dir),
            collection_name=COLLECTION_NAME,
            # Explicit cosine distance — required so relevance scores stay in
            # [0, 1] when querying with similarity_search_with_relevance_scores.
            collection_metadata={"hnsw:space": "cosine"},
        )


# ── Metadata helper ────────────────────────────────────────────────────────────


def fetch_video_title(url: str) -> Optional[str]:
    """Best-effort title fetch via yt-dlp. Returns None on failure."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "title", "--no-warnings", "--skip-download", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube video into the visual RAG store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="YouTube video URL.")
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=FRAME_INTERVAL_SECONDS,
        help=f"Seconds between extracted frames (default: {FRAME_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--transcript-window",
        type=float,
        default=10.0,
        help="Half-window in seconds for pairing transcript with each frame (default: 10).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB collection before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(
        Panel(
            "[bold cyan]YouTube Visual RAG — Ingestion Pipeline[/]",
            subtitle="[dim]Multimodal: frames + transcript → embeddings[/]",
        )
    )

    # 0. Pre-flight checks.
    require_binary("yt-dlp")
    require_binary("ffmpeg")
    check_ollama_ready()

    # 1. Identify the video.
    video_id = extract_video_id(args.url)
    work_dir = VIDEOS_DIR / video_id
    console.print(f"\n[bold]Video ID:[/] {video_id}")

    title = fetch_video_title(args.url)
    if title:
        console.print(f"[bold]Title:[/] {title}")

    # 2. Download.
    console.print("\n[bold]Step 1/5:[/] Downloading video...")
    video_path = download_video(args.url, work_dir)
    console.print(f"  [green]✓[/] Saved to [dim]{video_path}[/]")

    # 3. Extract frames.
    console.print("\n[bold]Step 2/5:[/] Extracting frames...")
    frames = extract_frames(video_path, work_dir / "frames", args.frame_interval)
    console.print(f"  [green]✓[/] Extracted [bold]{len(frames)}[/] frames "
                  f"(1 every {args.frame_interval}s).")

    # 4. Fetch transcript.
    console.print("\n[bold]Step 3/5:[/] Fetching transcript...")
    transcript = fetch_transcript(video_id)
    console.print(f"  [green]✓[/] Fetched [bold]{len(transcript)}[/] transcript segments.")

    # Persist the transcript alongside the video for debugging / re-use.
    (work_dir / "transcript.json").write_text(
        json.dumps([seg.__dict__ for seg in transcript], indent=2),
        encoding="utf-8",
    )

    # 5. Caption frames with the vision model.
    console.print(f"\n[bold]Step 4/5:[/] Captioning frames with [cyan]{VISION_MODEL}[/]...")
    captions: List[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Captioning...", total=len(frames))
        for frame in frames:
            cap = caption_frame(frame.path)
            captions.append(cap)
            progress.update(task, advance=1,
                            description=f"[cyan]Captioned t={int(frame.timestamp)}s")

    # Persist the captions for debugging.
    (work_dir / "captions.json").write_text(
        json.dumps(
            [{"timestamp": f.timestamp, "caption": c} for f, c in zip(frames, captions)],
            indent=2,
        ),
        encoding="utf-8",
    )

    # 6. Build documents and store.
    console.print("\n[bold]Step 5/5:[/] Embedding and storing in ChromaDB...")
    docs = build_documents(
        frames=frames,
        captions=captions,
        transcript_segments=transcript,
        video_id=video_id,
        video_url=args.url,
        video_title=title,
        half_window=args.transcript_window,
    )
    store_documents(docs, CHROMA_DIR, reset=args.reset)

    console.print(
        Panel(
            f"[bold green]Ingestion complete![/]\n\n"
            f"  Video      : {title or video_id}\n"
            f"  Frames     : {len(frames)} (every {args.frame_interval}s)\n"
            f"  Transcript : {len(transcript)} segments\n"
            f"  Documents  : {len(docs)} indexed\n"
            f"  Work dir   : {work_dir.resolve()}\n\n"
            f"Run [bold cyan]python visual_search.py[/] to start searching.",
            title="[bold]Summary[/]",
        )
    )


if __name__ == "__main__":
    main()
