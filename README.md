# 🎬 VidGrep — Multimodal RAG for YouTube

> **grep, but for YouTube.** Search any video by what is *shown on screen* and what is *spoken* — return ranked, timestamped moments with an embedded auto-seeking player.

This repo bundles two pipelines:

1. **VidGrep (Visual RAG)** — Search **YouTube videos by what is on screen *and* what is being said**, fully locally via Ollama + Moondream vision model. Each frame's caption is aligned with the surrounding transcript so semantic queries hit either modality.
2. **Text RAG companion** — Ask grounded questions over `.txt` / `.pdf` / `.md` documents (Claude-powered).

**VidGrep stack:** Python · `yt-dlp` · `ffmpeg` · Ollama + Moondream · LangChain · ChromaDB · sentence-transformers · Streamlit
**Text RAG stack:** Python 3.10+ · LangChain · ChromaDB · sentence-transformers · Claude 3.5 Haiku · Rich CLI

---

## 🎥 Demo

[![Watch the VidGrep demo on YouTube](https://img.youtube.com/vi/5PwcvqdmVJY/maxresdefault.jpg)](https://youtu.be/5PwcvqdmVJY)

▶ [Watch on YouTube](https://youtu.be/5PwcvqdmVJY) — a walkthrough of the Streamlit UI ingesting a YouTube video and running multimodal queries with timestamped result cards.

---

## Architecture

```
┌──────────────┐    chunk + embed    ┌──────────────┐
│  ./docs/     │ ─────────────────▶  │  ChromaDB    │
│  .txt / .pdf │   (ingest.py)       │  ./chroma_db │
│  .md files   │                     └──────┬───────┘
└──────────────┘                            │
                                     cosine similarity
                                     top-k retrieval
                                            │
┌──────────────┐    context + query  ┌──────▼───────┐
│  Claude      │ ◀────────────────── │  chat.py     │
│  3.5 Haiku   │                     │  (retriever) │
│  (Anthropic) │ ─────────────────▶  └──────────────┘
│              │   grounded answer
└──────────────┘   with citations
```

**Ingestion (one-time):** `ingest.py` walks `./docs/`, splits files into 500-character overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and persists the vectors in ChromaDB.

**Chat (every run):** `chat.py` embeds the user's question, retrieves the 4 most relevant chunks from ChromaDB (cosine similarity), builds a structured prompt, and calls Claude — which is instructed to answer only from provided context and cite sources.

---

## Setup

### 1. Clone / enter the project

```bash
cd ~/Projects/rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The sentence-transformers model (`all-MiniLM-L6-v2`, ~22 MB) downloads automatically on first run and is cached locally.

### 4. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

---

## Usage

### Step 1 — Ingest your documents

Add `.txt`, `.pdf`, or `.md` files to `./docs/`, then run:

```bash
python ingest.py
```

A sample document (`docs/sample.txt`) covering LangChain and RAG concepts is included so you can test immediately.

**Options:**

| Flag | Description |
|------|-------------|
| `--reset` | Wipe the existing vector store and re-ingest from scratch |
| `--docs-dir PATH` | Use a different document directory (default: `./docs`) |
| `--chroma-dir PATH` | Use a different ChromaDB path (default: `./chroma_db`) |

```bash
# Re-ingest after adding new documents or changing chunk size
python ingest.py --reset
```

### Step 2 — Start chatting

```bash
python chat.py
```

The chatbot will:
1. Display a table of the top-4 retrieved chunks and their relevance scores
2. Show a "Thinking…" spinner while Claude generates the answer
3. Render the answer as formatted Markdown with source citations

**Options:**

| Flag | Description |
|------|-------------|
| `--top-k N` | Retrieve N chunks per query (default: 4) |
| `--chroma-dir PATH` | ChromaDB path to query (default: `./chroma_db`) |

```bash
# Retrieve more context for complex questions
python chat.py --top-k 6
```

Type `exit`, `quit`, or press `Ctrl+D` to stop.

---

## Example session

```
You: What embedding model does all-MiniLM-L6-v2 use under the hood?

┌─ Retrieved Context ──────────────────────────────────────────────────────┐
│ #  Source       Relevance  Preview                                        │
│ 1  sample.txt   0.872      Architecture: MiniLM (a distilled BERT varia… │
│ 2  sample.txt   0.841      Dimension: 384-dimensional output vectors…     │
│ 3  sample.txt   0.798      Speed: ~14,200 sentences/second on a modern… │
│ 4  sample.txt   0.763      For production at scale, consider larger mod… │
└───────────────────────────────────────────────────────────────────────────┘

┌─ Assistant ───────────────────────────────────────────────────────────────┐
│                                                                            │
│  all-MiniLM-L6-v2 is built on the **MiniLM** architecture, which is a    │
│  distilled variant of BERT. It has 6 transformer layers and produces     │
│  **384-dimensional** embedding vectors.                                   │
│                                                                            │
│  **Sources**                                                               │
│  - `sample.txt`: "Architecture: MiniLM (a distilled BERT variant) with   │
│    6 transformer layers."                                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Visual RAG — YouTube pipeline

Index any YouTube video by **both** the spoken transcript **and** what is visually on screen, then search across all of them with natural language.

### How it works

```
YouTube URL
    │
    ├──► yt-dlp ──► video.mp4
    │                  │
    │                  └──► ffmpeg ──► frames (1 every 5s, 480p JPEG)
    │                                       │
    │                                       └──► Ollama (moondream)
    │                                              │
    │                                              └──► visual caption
    │
    └──► youtube-transcript-api ──► timestamped transcript segments
                                              │
                                              └──► transcript window per frame

  combined "[Visual]: …  [Spoken]: …"  ──►  sentence-transformers
                                                    │
                                                    └──►  ChromaDB
                                                              │
                                                              └──►  visual_search.py
                                                                       │
                                                                       └──► timestamps + frames
                                                                           + youtu.be deep links
```

### Setup (one-time, in addition to text RAG setup)

```bash
# Install the system tools
brew install yt-dlp ffmpeg ollama

# Start the Ollama daemon (one of these — `brew services` is persistent)
brew services start ollama
# (or: ollama serve   in a separate terminal)

# Pull the vision model (~1.7 GB)
ollama pull moondream
```

Then re-install Python deps to pick up `youtube-transcript-api` and `requests`:

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Ingest a video — this can take a few minutes for the captioning step
python youtube_visual_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Tune the frame sampling rate (smaller = more frames = slower but finer-grained)
python youtube_visual_ingest.py "<URL>" --frame-interval 3

# Wipe and start over
python youtube_visual_ingest.py "<URL>" --reset

# Search across everything you've ingested
python visual_search.py                          # interactive
python visual_search.py -q "circuit board"       # one-shot
python visual_search.py -q "diagram" -k 8        # top-K control
```

Results show: timestamp, relevance score, the visual caption, the surrounding transcript, the local frame path, and a clickable `youtu.be?t=...` deep link.

### Visual RAG options

**Ingestion (`youtube_visual_ingest.py`)**

| Flag | Description |
|------|-------------|
| `url` | YouTube video URL (positional) |
| `--frame-interval N` | Seconds between extracted frames (default: 5) |
| `--transcript-window S` | Half-window in seconds for pairing transcript with each frame (default: 10) |
| `--reset` | Wipe the existing visual collection before ingesting |

**Search (`visual_search.py`)**

| Flag | Description |
|------|-------------|
| `-q, --query` | One-shot query (skips the REPL) |
| `-k, --top-k` | Number of results to return (default: 5) |

---

### Visual RAG — Web UI

A Streamlit UI is included for a polished demo experience. It provides:

  • A search bar with semantic visual querying
  • Result cards with an embedded YouTube player that **auto-jumps to the matched timestamp**
  • Frame thumbnails, Moondream captions, and the windowed transcript per match
  • A sidebar with the ingest form and a thumbnail library of every ingested video

```bash
streamlit run app.py
```

Then open the URL it prints (default: `http://localhost:8501`). Use the **➕ Ingest a new video** form in the sidebar to add videos directly from the UI — it spawns the CLI ingest in a subprocess and streams the log.

---

## Project structure

```
rag-chatbot/
├── docs/                       # Source documents for TEXT RAG
│   └── sample.txt
├── videos/                     # Auto-created by visual ingest
│   └── <video_id>/
│       ├── video.mp4
│       ├── frames/             # Extracted JPEG frames
│       ├── transcript.json
│       └── captions.json
├── chroma_db/                  # Shared vector store (both pipelines)
├── ingest.py                   # TEXT RAG ingestion
├── chat.py                     # TEXT RAG chat (Claude API)
├── youtube_visual_ingest.py    # VISUAL RAG ingestion
├── visual_search.py            # VISUAL RAG search CLI
├── app.py                      # VISUAL RAG web UI (Streamlit)
├── requirements.txt
└── README.md
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| `all-MiniLM-L6-v2` embeddings | Fast CPU inference (~14k sent/s), Apache 2.0 license, no API key needed |
| Chunk size 500 chars, overlap 50 | Fits within the model's 256-token context; overlap prevents losing context at boundaries |
| Top-k = 4 | Balances recall vs. context window cost; adjustable via `--top-k` |
| Temperature = 0.2 | Prioritises factual consistency over creativity for a Q&A use case |
| `RecursiveCharacterTextSplitter` | Prefers semantic break points (paragraphs → sentences → words) over hard cuts |
| Persist ChromaDB to disk | Vector store survives process restarts — no re-ingestion needed between chat sessions |
| Source citations in system prompt | Forces Claude to be auditable; prevents hallucination drift |

---

## Extending the project

- **Add a web UI:** Wrap `chat.py` logic in a FastAPI endpoint and pair with a Svelte or React frontend.
- **Hybrid search:** Add BM25 retrieval alongside vector search and merge results with Reciprocal Rank Fusion.
- **Reranking:** Pass top-20 vector results through `cross-encoder/ms-marco-MiniLM-L-6-v2` before sending top-4 to Claude.
- **Multi-turn memory:** Use LangChain's `RunnableWithMessageHistory` to maintain conversation context across turns.
- **Evaluation:** Integrate [RAGAS](https://github.com/explodinggradients/ragas) to benchmark faithfulness and answer relevance.

---

## Author

**Lawrence Wang** — Systems Design Engineering, University of Waterloo
