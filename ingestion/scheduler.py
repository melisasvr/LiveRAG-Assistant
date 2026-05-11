"""
LiveRAG — Background Ingestion Scheduler
APScheduler-based continuous ingestion with status tracking.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from ingestion.fetcher import fetch_all_feeds, Article
from retrieval.vector_store import LiveVectorStore
from config.settings import RSS_FEEDS, FETCH_INTERVAL_MINUTES, MAX_ARTICLES_PER_FEED

logger = logging.getLogger(__name__)


class IngestionStatus:
    """Thread-safe status tracker for the ingestion pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.last_new_articles: int = 0
        self.last_skipped: int = 0
        self.total_runs: int = 0
        self.is_running: bool = False
        self.last_error: Optional[str] = None
        self.status_msg: str = "Not started"
        self.history: list = []  # List of run summaries

    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "last_run": self.last_run.isoformat() if self.last_run else None,
                "next_run": self.next_run.isoformat() if self.next_run else None,
                "last_new_articles": self.last_new_articles,
                "last_skipped": self.last_skipped,
                "total_runs": self.total_runs,
                "is_running": self.is_running,
                "last_error": self.last_error,
                "status_msg": self.status_msg,
            }


# Global singleton
_status = IngestionStatus()


def get_status() -> IngestionStatus:
    return _status


def run_ingestion_cycle(
    vector_store: LiveVectorStore,
    on_complete: Optional[Callable] = None,
    fetch_full: bool = True,
) -> None:
    """
    One complete ingestion cycle:
    1. Fetch fresh articles from all RSS feeds
    2. Embed + upsert into ChromaDB (skip duplicates)
    3. Update status
    """
    global _status
    with _status._lock:
        if _status.is_running:
            logger.warning("Ingestion cycle already running, skipping.")
            return
        _status.is_running = True
        _status.status_msg = "⏳ Fetching articles..."
        _status.last_error = None

    try:
        logger.info("=== Ingestion Cycle Start ===")
        articles = fetch_all_feeds(
            RSS_FEEDS,
            max_per_feed=MAX_ARTICLES_PER_FEED,
            max_age_hours=72,
            fetch_full_content=fetch_full,
        )

        with _status._lock:
            _status.status_msg = f"📥 Embedding {len(articles)} articles..."

        new_count, skipped = vector_store.ingest_articles(articles, skip_existing=True)
        now = datetime.now(timezone.utc)

        with _status._lock:
            _status.last_run = now
            _status.last_new_articles = new_count
            _status.last_skipped = skipped
            _status.total_runs += 1
            _status.status_msg = f"✅ Done — {new_count} new, {skipped} skipped"
            _status.history.append({
                "time": now.isoformat(),
                "new": new_count,
                "skipped": skipped,
            })
            if len(_status.history) > 50:
                _status.history = _status.history[-50:]

        logger.info(f"=== Ingestion Cycle Done: {new_count} new, {skipped} skipped ===")

        if on_complete:
            on_complete(new_count, skipped)

    except Exception as e:
        logger.error(f"Ingestion cycle error: {e}", exc_info=True)
        with _status._lock:
            _status.last_error = str(e)
            _status.status_msg = f"❌ Error: {str(e)[:80]}"
    finally:
        with _status._lock:
            _status.is_running = False


class LiveIngestionScheduler:
    """
    Wraps APScheduler to run ingestion cycles at a fixed interval.
    Exposes start/stop/trigger_now controls.
    """

    def __init__(
        self,
        vector_store: LiveVectorStore,
        interval_minutes: int = FETCH_INTERVAL_MINUTES,
        fetch_full_content: bool = True,
    ):
        self.vector_store = vector_store
        self.interval_minutes = interval_minutes
        self.fetch_full_content = fetch_full_content
        self._scheduler = BackgroundScheduler(daemon=True)
        self._job = None

    def start(self):
        if self._scheduler.running:
            logger.info("Scheduler already running.")
            return
        self._job = self._scheduler.add_job(
            func=self._run,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="ingestion_job",
            name="LiveRAG Ingestion",
            replace_existing=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info(f"Scheduler started. Interval: {self.interval_minutes} min")

        # Update next run time
        next_run = self._job.next_run_time
        with _status._lock:
            _status.next_run = next_run
            _status.status_msg = f"🕐 Scheduler active. Next run: {next_run.strftime('%H:%M UTC') if next_run else 'soon'}"

    def stop(self):
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")

    def trigger_now(self):
        """Manually trigger an ingestion cycle in a background thread."""
        thread = threading.Thread(
            target=run_ingestion_cycle,
            args=(self.vector_store,),
            kwargs={"fetch_full": self.fetch_full_content},
            daemon=True,
        )
        thread.start()
        return thread

    def _run(self):
        run_ingestion_cycle(self.vector_store, fetch_full=self.fetch_full_content)
        if self._job:
            next_run = self._job.next_run_time
            with _status._lock:
                _status.next_run = next_run

    @property
    def is_running(self) -> bool:
        return self._scheduler.running