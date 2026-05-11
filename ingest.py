"""
LiveRAG — CLI ingestion runner
Run this to seed the knowledge base without the Streamlit UI.

Usage:
  python ingest.py                  # Fetch & index all feeds
  python ingest.py --no-full        # Use feed summaries only (faster)
  python ingest.py --feeds 3        # Only first 3 feeds (test mode)
  python ingest.py --watch          # Continuous mode (every 10 min)
"""

import argparse
import logging
import sys
import time
import os

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("liverag.ingest")


def main():
    parser = argparse.ArgumentParser(description="LiveRAG ingestion CLI")
    parser.add_argument("--no-full", action="store_true", help="Skip full content fetch")
    parser.add_argument("--feeds", type=int, default=None, help="Limit number of feeds")
    parser.add_argument("--watch", action="store_true", help="Continuous ingestion mode")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval (minutes)")
    args = parser.parse_args()

    from config.settings import (
        RSS_FEEDS, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
        MAX_ARTICLES_PER_FEED,
    )
    from retrieval.vector_store import LiveVectorStore
    from ingestion.fetcher import fetch_all_feeds

    logger.info("Initialising vector store…")
    vs = LiveVectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )

    feeds = RSS_FEEDS[:args.feeds] if args.feeds else RSS_FEEDS
    fetch_full = not args.no_full

    def run_once():
        logger.info(f"Fetching from {len(feeds)} feeds (full_content={fetch_full})…")
        articles = fetch_all_feeds(feeds, max_per_feed=MAX_ARTICLES_PER_FEED, fetch_full_content=fetch_full)
        new, skipped = vs.ingest_articles(articles)
        stats = vs.get_stats()
        logger.info(
            f"Done — {new} new, {skipped} skipped | "
            f"KB: {stats['unique_articles']} articles, {stats['total_chunks']} chunks"
        )

    if args.watch:
        logger.info(f"Watch mode — running every {args.interval} minutes. Ctrl+C to stop.")
        while True:
            try:
                run_once()
            except KeyboardInterrupt:
                logger.info("Stopped.")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            time.sleep(args.interval * 60)
    else:
        run_once()


if __name__ == "__main__":
    main()