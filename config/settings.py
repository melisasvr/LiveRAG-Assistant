"""
LiveRAG Assistant — Central Configuration
"""
from dataclasses import dataclass, field
from typing import List, Dict

# ─── RSS Feed Sources ────────────────────────────────────────────────────────

RSS_FEEDS: List[Dict[str, str]] = [
    # AI / ML
    {"url": "https://www.marktechpost.com/feed/", "category": "AI/ML", "source": "MarkTechPost"},
    {"url": "https://machinelearningmastery.com/blog/feed/", "category": "AI/ML", "source": "ML Mastery"},
    {"url": "https://openai.com/news/rss.xml", "category": "AI/ML", "source": "OpenAI"},
    {"url": "https://blogs.microsoft.com/ai/feed/", "category": "AI/ML", "source": "Microsoft AI"},
    {"url": "https://ai.googleblog.com/feeds/posts/default", "category": "AI/ML", "source": "Google AI Blog"},
    {"url": "https://huggingface.co/blog/feed.xml", "category": "AI/ML", "source": "HuggingFace"},
    {"url": "https://www.deepmind.com/blog/rss.xml", "category": "AI/ML", "source": "DeepMind"},
    # Tech News
    {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "Tech News", "source": "TechCrunch AI"},
    {"url": "https://www.wired.com/feed/category/artificial-intelligence/latest/rss", "category": "Tech News", "source": "Wired AI"},
    {"url": "https://venturebeat.com/category/ai/feed/", "category": "Tech News", "source": "VentureBeat AI"},
    {"url": "https://thenextweb.com/neural/feed/", "category": "Tech News", "source": "The Next Web"},
    {"url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml", "category": "Tech News", "source": "The Verge AI"},
    # Research
    {"url": "https://arxiv.org/rss/cs.LG", "category": "Research", "source": "ArXiv ML"},
    {"url": "https://arxiv.org/rss/cs.AI", "category": "Research", "source": "ArXiv AI"},
    {"url": "https://arxiv.org/rss/cs.CL", "category": "Research", "source": "ArXiv NLP"},
    # Industry
    {"url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml", "category": "Industry", "source": "ZDNet AI"},
    {"url": "https://spectrum.ieee.org/rss/ai.xml", "category": "Industry", "source": "IEEE Spectrum AI"},
]

# ─── Vector Store ────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "liverag_articles"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── LLM ─────────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"     # Primary
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"     # Fallback / faster
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ─── Ingestion ───────────────────────────────────────────────────────────────

FETCH_INTERVAL_MINUTES = 10          # How often to re-fetch feeds
MAX_ARTICLES_PER_FEED = 20           # Cap per feed per cycle
ARTICLE_MAX_CHARS = 8000             # Truncate fetched body to this
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ─── Retrieval ───────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL = 8                  # Docs to retrieve
RECENCY_WEIGHTS = {                  # Extra weight for recent docs
    "24h": 0.40,
    "week": 0.25,
    "month": 0.10,
    "all": 0.00,
}

# ─── UI ──────────────────────────────────────────────────────────────────────

APP_TITLE = "LiveRAG Assistant"
APP_ICON = "⚡"
AUTO_REFRESH_SECONDS = 300           # Streamlit auto-refresh interval
MAX_CHAT_HISTORY = 20