"""
LiveRAG — RSS Feed Ingestion Pipeline
Fetches, deduplicates, and indexes fresh articles into ChromaDB.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import feedparser
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ── Article Model ──────────────────────────────────────────────────────────

@dataclass
class Article:
    url: str
    title: str
    content: str
    summary: str
    source: str
    category: str
    published_at: datetime
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    doc_id: str = ""
    word_count: int = 0

    def __post_init__(self):
        self.doc_id = _make_doc_id(self.url)
        self.word_count = len(self.content.split())

    @property
    def age_hours(self) -> float:
        now = datetime.now(timezone.utc)
        pub = self.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return (now - pub).total_seconds() / 3600

    def to_metadata(self) -> Dict:
        pub = self.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "published_at": pub.isoformat(),
            "published_ts": int(pub.timestamp()),
            "fetched_at": self.fetched_at.isoformat(),
            "word_count": self.word_count,
            "doc_id": self.doc_id,
        }


def _make_doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_date(entry) -> datetime:
    """Try to extract publish date from feed entry."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


# ── Content Fetcher ────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LiveRAG/1.0; +https://github.com/liverag)"
    )
}

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=False)
def _fetch_full_content(url: str, max_chars: int = 8000) -> Optional[str]:
    """Fetch full article text via trafilatura (best) or newspaper3k fallback."""
    try:
        import trafilatura
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            return text[:max_chars]
    except Exception as e:
        logger.debug(f"trafilatura failed for {url}: {e}")

    try:
        from newspaper import Article as NpArticle
        a = NpArticle(url)
        a.download()
        a.parse()
        if a.text and len(a.text) > 200:
            return a.text[:max_chars]
    except Exception as e:
        logger.debug(f"newspaper3k failed for {url}: {e}")

    return None


# ── Feed Parser ────────────────────────────────────────────────────────────

def fetch_feed(
    feed_cfg: Dict[str, str],
    max_articles: int = 20,
    max_age_hours: float = 72,
    fetch_full_content: bool = True,
) -> List[Article]:
    """Parse one RSS feed and return Article objects."""
    url = feed_cfg["url"]
    source = feed_cfg.get("source", url)
    category = feed_cfg.get("category", "General")

    articles: List[Article] = []

    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        logger.warning(f"feedparser failed for {url}: {e}")
        return articles

    for entry in parsed.entries[:max_articles]:
        try:
            published_at = _parse_date(entry)
            now = datetime.now(timezone.utc)
            pub = published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_h = (now - pub).total_seconds() / 3600
            if age_h > max_age_hours:
                continue

            article_url = getattr(entry, "link", "") or ""
            if not article_url:
                continue

            title = getattr(entry, "title", "Untitled")

            # Summary from feed
            summary = ""
            for attr in ("summary", "description", "content"):
                val = getattr(entry, attr, None)
                if val:
                    if isinstance(val, list):
                        val = val[0].get("value", "")
                    # Strip HTML tags simply
                    import re
                    summary = re.sub(r"<[^>]+>", " ", str(val)).strip()[:500]
                    break

            # Full content
            content = summary
            if fetch_full_content:
                full = _fetch_full_content(article_url)
                if full:
                    content = full
                time.sleep(0.3)  # Polite delay

            if not content.strip():
                continue

            articles.append(
                Article(
                    url=article_url,
                    title=title,
                    content=content,
                    summary=summary,
                    source=source,
                    category=category,
                    published_at=published_at,
                )
            )
        except Exception as e:
            logger.debug(f"Failed to process entry from {url}: {e}")
            continue

    logger.info(f"[{source}] Fetched {len(articles)} articles")
    return articles


def fetch_all_feeds(
    feed_configs: List[Dict],
    max_per_feed: int = 20,
    max_age_hours: float = 72,
    fetch_full_content: bool = True,
) -> List[Article]:
    """Fetch articles from all configured RSS feeds."""
    all_articles: List[Article] = []
    for cfg in feed_configs:
        try:
            articles = fetch_feed(
                cfg,
                max_articles=max_per_feed,
                max_age_hours=max_age_hours,
                fetch_full_content=fetch_full_content,
            )
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Feed fetch error [{cfg.get('source')}]: {e}")
    logger.info(f"Total articles fetched across all feeds: {len(all_articles)}")
    return all_articles