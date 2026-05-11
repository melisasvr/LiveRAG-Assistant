# ⚡ LiveRAG Assistant

> Real-time AI & ML news Q&A powered by **Groq + ChromaDB + LangChain + Streamlit**

A production-grade **Live RAG** system that continuously ingests fresh AI/ML and tech news, embeds it into a persistent vector store, and answers questions with up-to-date knowledge with clear source citations and "Knowledge as of [timestamp]" transparency.

---

## 🏗️ Architecture

```
RSS Feeds (17 sources)
       │
       ▼
┌──────────────────┐     APScheduler      ┌────────────────────┐
│  Ingestion Layer  │ ──── every 10min ───▶│  ChromaDB (local)  │
│  fetcher.py       │                     │  Persistent vector │
│  trafilatura      │                     │  store with meta   │
│  newspaper3k      │                     │  (publish_date,    │
└──────────────────┘                      │   source, url, cat)│
                                          └────────────────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  Hybrid Retrieval  │
                                          │  similarity search │
                                          │  + recency re-rank │
                                          └─────────┬──────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  Groq LLM Chain    │
                                          │  Llama 3.3 70B     │
                                          │  Time-aware prompt │
                                          └─────────┬──────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  Streamlit UI      │
                                          │  Chat / News /     │
                                          │  Live vs Static /  │
                                          │  Analytics         │
                                          └────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd liverag
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Seed the knowledge base

```bash
# Quick test (summaries only, 3 feeds)
python ingest.py --no-full --feeds 3

# Full ingestion (all feeds, full article content)
python ingest.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
liverag/
├── app.py                      # Streamlit frontend (main entry point)
├── ingest.py                   # CLI ingestion runner
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml             # Dark theme config
│
├── config/
│   └── settings.py             # All configuration (feeds, model, retrieval)
│
├── ingestion/
│   ├── fetcher.py              # RSS feed parser + content extractor
│   └── scheduler.py            # APScheduler-based background ingestion
│
├── retrieval/
│   ├── vector_store.py         # ChromaDB manager (ingest + hybrid retrieval)
│   └── rag_chain.py            # LangChain + Groq RAG pipeline
│
└── data/
    └── chroma_db/              # Persistent vector store (auto-created)
```

---

## 🎛️ Key Features

### 1. Continuous Ingestion (every 10 min)
- 17 RSS feeds across AI/ML, Tech News, Research, Industry
- Full article content via `trafilatura` → `newspaper3k` fallback
- Smart deduplication: SHA-256 URL hash prevents re-indexing
- Polite fetch delays to avoid rate limiting

### 2. Time-Aware Retrieval
```python
docs = vs.retrieve(
    query="latest LLM research",
    k=8,
    recency="24h",          # Filter by publish date
    category="Research",    # Optional category filter
    recency_boost=0.25,     # Weight for freshness vs relevance
)
```

### 3. Metadata-Rich Vector Store
Every document chunk stores:
- `published_at` / `published_ts` — for time filtering
- `source` — feed source name
- `url` — original article link
- `category` — AI/ML, Tech News, Research, Industry
- `doc_id` — deduplication hash
- `chunk_index` / `chunk_total` — chunk position

### 4. Source Citations
Every answer includes:
```
📰 [Title] — Source Name | 2025-01-15 | AI/ML
```
With `Knowledge as of [timestamp]` badge.

### 5. Live vs Static Demo Tab
Demonstrates the concrete value of Live RAG by:
1. Answering with fresh live context (citing dates)
2. Noting what static RAG would have said
3. Highlighting what has changed

---

## ⚙️ Configuration

Edit `config/settings.py` to customise:

| Setting | Default | Description |
|---------|---------|-------------|
| `FETCH_INTERVAL_MINUTES` | 10 | How often to re-fetch feeds |
| `MAX_ARTICLES_PER_FEED` | 20 | Articles fetched per feed per cycle |
| `GROQ_MODEL` | `llama-3.1-70b-versatile` | LLM model |
| `TOP_K_RETRIEVAL` | 8 | Docs retrieved per query |
| `CHUNK_SIZE` | 800 | Text chunk size (tokens) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer |

### Adding RSS feeds

```python
# In config/settings.py
RSS_FEEDS.append({
    "url": "https://yoursite.com/feed.xml",
    "category": "AI/ML",
    "source": "Your Site Name",
})
```

---

## 🔌 Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq → Llama 3.3 70B |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent, local) |
| RAG Framework | LangChain |
| Feed parsing | feedparser |
| Content extraction | trafilatura + newspaper3k |
| Scheduler | APScheduler |
| Frontend | Streamlit |
| Charts | Plotly |

---

## 🧪 CLI Reference

```bash
# One-time ingestion (full content)
python ingest.py

# Fast ingestion (summaries only)
python ingest.py --no-full

# Test with first 3 feeds
python ingest.py --feeds 3 --no-full

# Continuous watch mode (every 15 min)
python ingest.py --watch --interval 15
```

---

## 🛣️ Roadmap / Extensions

- [ ] **Topic Alerts**: Monitor for keywords and notify on new articles
- [ ] **GraphRAG Integration**: Hybrid search across static book knowledge + live news
- [ ] **Multi-domain support**: Finance, Science, Healthcare feeds
- [ ] **Semantic clustering**: Group related articles automatically
- [ ] **Export**: Download Q&A sessions as PDF reports
- [ ] **Multi-user**: Per-user conversation history

---
## Contributing
- Contributions are welcome! Whether it's improving the graph schema, enhancing retrieval strategies, adding new data sources, or fixing bugs, feel free to get involved. How to Contribute
- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-improvement)
- Commit your changes (git commit -m 'Add amazing improvement')
- Push to the branch (git push origin feature/amazing-improvement)
- Open a Pull Request

---

## 📄 License

- MIT Linces
```
Copyright (c) 2026 LiveRAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
