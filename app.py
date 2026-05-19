#!/usr/bin/env python3
"""
app.py — VidGrep web UI (Streamlit).

VidGrep is a multimodal RAG system that searches YouTube videos by both
what is shown on screen and what is being said.

Run with:
    streamlit run app.py

Features:
  • Semantic visual + transcript search across all ingested videos
  • Result cards with embedded YouTube player jumping to the matched timestamp
  • Frame thumbnails + Moondream caption + windowed transcript per match
  • Sidebar: ingest form (runs youtube_visual_ingest.py as a subprocess) and
    a library view of every ingested video with its first-frame thumbnail
  • Cached vector store (loaded once per session) for instant search
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from langchain_core.documents import Document

# Re-use the search primitives from the CLI module so we have one source of truth.
from visual_search import load_store, search, format_timestamp


# ── Paths & constants ─────────────────────────────────────────────────────────

VIDEOS_DIR = Path("./videos")
CHROMA_DIR = Path("./chroma_db")
INGEST_SCRIPT = Path("./youtube_visual_ingest.py")
YOUTUBE_THUMB_TPL = "https://img.youtube.com/vi/{vid}/hqdefault.jpg"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VidGrep — Multimodal YouTube Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def get_store():
    """Load the Chroma store once per session (the embedding model is the slow part)."""
    if not CHROMA_DIR.exists():
        return None
    try:
        return load_store()
    except SystemExit:
        # load_store() calls sys.exit on missing DB; treat as None for the UI.
        return None


# ── Library helpers ───────────────────────────────────────────────────────────


def list_ingested_videos() -> List[dict]:
    """Walk ./videos/<id>/ and return a list of dicts describing each ingestion."""
    if not VIDEOS_DIR.exists():
        return []

    out = []
    for video_dir in sorted(VIDEOS_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        caps_file = video_dir / "captions.json"
        if not caps_file.exists():
            continue
        try:
            captions = json.loads(caps_file.read_text())
        except Exception:
            captions = []

        frames = sorted((video_dir / "frames").glob("frame_*.jpg"))
        out.append({
            "video_id": video_dir.name,
            "title": video_dir.name,  # not stored; UI uses ID as fallback
            "thumb": YOUTUBE_THUMB_TPL.format(vid=video_dir.name),
            "frames_count": len(frames),
            "first_frame": str(frames[0]) if frames else None,
            "captions_count": len(captions),
            "url": f"https://www.youtube.com/watch?v={video_dir.name}",
        })
    return out


# ── Sidebar: library + ingest ─────────────────────────────────────────────────


def render_sidebar():
    """Sidebar with ingest form and a thumbnail library of ingested videos."""
    st.sidebar.title("🎬 VidGrep")
    st.sidebar.caption(
        "grep, but for YouTube. Search videos by what's shown on screen "
        "and what's spoken. Powered by local Moondream + ChromaDB."
    )

    # ── Ingest form ──
    with st.sidebar.expander("➕ Ingest a new video", expanded=False):
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="ingest_url",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            interval = st.number_input(
                "Frame interval (s)", min_value=1, max_value=60, value=5, step=1
            )
        with col_b:
            reset = st.checkbox("Reset DB", value=False,
                                help="Wipe all existing embeddings before ingesting")

        if st.button("Ingest", type="primary", use_container_width=True):
            if not url.strip():
                st.warning("Please paste a YouTube URL first.")
            else:
                run_ingest(url.strip(), interval, reset)

    # ── Library ──
    st.sidebar.divider()
    st.sidebar.subheader("📚 Library")

    videos = list_ingested_videos()
    if not videos:
        st.sidebar.info("No videos ingested yet. Use the form above to add one.")
        return

    st.sidebar.caption(
        f"**{len(videos)}** video(s) · "
        f"**{sum(v['captions_count'] for v in videos)}** indexed moments"
    )

    for v in videos:
        with st.sidebar.container():
            cols = st.sidebar.columns([1, 2])
            with cols[0]:
                st.image(v["thumb"], use_container_width=True)
            with cols[1]:
                st.markdown(f"**{v['video_id']}**")
                st.caption(f"{v['captions_count']} moments")
                st.markdown(f"[↗ Open on YouTube]({v['url']})")
            st.sidebar.divider()


def run_ingest(url: str, interval: int, reset: bool) -> None:
    """Spawn the ingest CLI as a subprocess and stream its output into the UI."""
    cmd = [sys.executable, str(INGEST_SCRIPT), url, "--frame-interval", str(interval)]
    if reset:
        cmd.append("--reset")

    log_area = st.empty()
    log_lines: List[str] = []

    with st.spinner("Running ingest pipeline — this can take a few minutes…"):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            log_lines.append(line)
            # Show only the last ~20 lines to keep things tidy.
            log_area.code("\n".join(log_lines[-20:]), language="text")
        proc.wait()

    if proc.returncode == 0:
        st.success(f"Ingest complete for `{url}`")
        # Clear cached store so the new video shows up in searches immediately.
        get_store.clear()
        st.rerun()
    else:
        st.error(f"Ingest failed (exit code {proc.returncode}). See log above.")


# ── Result rendering ──────────────────────────────────────────────────────────


def yt_id_from_url(url: str) -> Optional[str]:
    """Extract a YouTube video ID from any common URL shape."""
    if not url:
        return None
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def render_result_card(doc: Document, score: float, idx: int) -> None:
    """Render a single search hit as a two-column card."""
    md = doc.metadata
    timestamp = float(md.get("timestamp", 0))
    video_url = md.get("video_url", "")
    video_id = md.get("video_id") or yt_id_from_url(video_url) or ""
    title = md.get("video_title") or video_id or "Unknown video"
    caption = (md.get("caption") or "").strip()
    transcript = (md.get("transcript") or "").strip()
    frame_path = md.get("frame_path")
    yt_link = md.get("youtube_link") or f"https://youtu.be/{video_id}?t={int(timestamp)}"

    with st.container(border=True):
        header_cols = st.columns([6, 2])
        with header_cols[0]:
            st.markdown(f"### #{idx} · {title}")
            st.caption(
                f"⏱ **{format_timestamp(timestamp)}**   ·   "
                f"📊 relevance **{score:.3f}**   ·   "
                f"🔗 [open at this moment]({yt_link})"
            )
        with header_cols[1]:
            st.markdown(f"<div style='text-align:right; font-size:2rem;'>"
                        f"{score_emoji(score)}</div>",
                        unsafe_allow_html=True)

        body_cols = st.columns([1, 1])

        # Left: extracted frame thumbnail.
        with body_cols[0]:
            if frame_path and Path(frame_path).exists():
                st.image(frame_path, caption=f"Frame at {format_timestamp(timestamp)}",
                         use_container_width=True)
            else:
                st.info("Frame thumbnail not available.")

        # Right: embedded YouTube player jumping to the matched timestamp.
        with body_cols[1]:
            if video_url:
                try:
                    st.video(video_url, start_time=int(timestamp))
                except Exception:
                    st.markdown(f"[▶ Watch on YouTube]({yt_link})")
            else:
                st.markdown(f"[▶ Watch on YouTube]({yt_link})")

        # Captions + transcript below the media row.
        st.markdown("**👁 Visual caption**")
        st.write(caption or "_(no caption)_")

        if transcript:
            st.markdown("**🗣 Spoken transcript (windowed)**")
            st.write(transcript)


def score_emoji(score: float) -> str:
    """A tiny visual cue for relevance."""
    if score >= 0.6:
        return "🟢"
    if score >= 0.4:
        return "🟡"
    return "🔴"


# ── Main page ─────────────────────────────────────────────────────────────────


def render_main():
    """Top-of-page search bar and the result grid."""
    st.title("🎬 VidGrep")
    st.markdown(
        "**grep, but for YouTube.** Search videos by **what is on screen** *and* "
        "what is being said — returns ranked timestamped moments with an embedded "
        "auto-seeking player. Powered by locally-hosted **Moondream** (vision) "
        "+ **sentence-transformers** + **ChromaDB**."
    )

    store = get_store()
    if store is None:
        st.warning(
            "No vector store yet. Use the **➕ Ingest a new video** form in the "
            "sidebar to index your first video."
        )
        return

    # Search controls
    cols = st.columns([6, 1, 1])
    with cols[0]:
        query = st.text_input(
            "Search query",
            placeholder="e.g. 'someone holding a soldering iron', 'a circuit board', "
                        "'a diagram on screen'",
            label_visibility="collapsed",
            key="query",
        )
    with cols[1]:
        k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1,
                            label_visibility="collapsed")
    with cols[2]:
        go = st.button("Search", type="primary", use_container_width=True)

    if not (query and (go or query)):
        st.info(
            "💡 **Tip:** Visual search beats keyword search for moments that are "
            "*shown* rather than spoken. Try queries like *'someone pointing at a "
            "diagram'* or *'close-up of hands typing'*."
        )
        return

    with st.spinner("Searching…"):
        results: List[Tuple[Document, float]] = search(store, query, int(k))

    if not results:
        st.warning("No matches found. Try a broader query.")
        return

    st.markdown(f"#### Top {len(results)} matches for: _{query}_")
    for i, (doc, score) in enumerate(results, start=1):
        render_result_card(doc, score, i)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
