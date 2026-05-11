"""
LiveRAG — ChromaDB Vector Store Manager
Handles embedding, deduplication, and metadata-aware retrieval.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.fetcher import Article

logger = logging.getLogger(__name__)


class LiveVectorStore:
    """
    ChromaDB-backed vector store with:
    - Incremental upserts (dedup by doc_id)
    - Metadata filtering by recency, category, source
    - Hybrid retrieval (similarity + recency boost)
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "liverag_articles",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        os.makedirs(persist_dir, exist_ok=True)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        logger.info(
            f"Vector store ready. Documents: {self.store._collection.count()}"
        )

    # ── Ingestion ─────────────────────────────────────────────────────────

    def get_existing_ids(self) -> set:
        """Return all doc_ids already in the store."""
        try:
            result = self.store._collection.get(include=["metadatas"])
            return {m.get("doc_id", "") for m in result["metadatas"] if m}
        except Exception:
            return set()

    def ingest_articles(
        self, articles: List[Article], skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        Embed and store articles.
        Returns (new_count, skipped_count).
        """
        existing_ids = self.get_existing_ids() if skip_existing else set()
        new_docs: List[Document] = []
        skipped = 0

        for article in articles:
            if skip_existing and article.doc_id in existing_ids:
                skipped += 1
                continue

            chunks = self.splitter.split_text(article.content)
            for i, chunk in enumerate(chunks):
                metadata = article.to_metadata()
                metadata["chunk_index"] = i
                metadata["chunk_total"] = len(chunks)
                # Unique ID per chunk
                chunk_id = f"{article.doc_id}_{i}"
                new_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata,
                    )
                )

        if new_docs:
            # Add in batches to avoid memory spikes
            batch_size = 50
            for i in range(0, len(new_docs), batch_size):
                batch = new_docs[i : i + batch_size]
                self.store.add_documents(batch)
            logger.info(f"Indexed {len(new_docs)} chunks from {len(articles) - skipped} new articles")

        return len(articles) - skipped, skipped

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _recency_filter(self, recency: str) -> Optional[Dict]:
        """Build a ChromaDB `where` filter for recency."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        windows = {
            "24h": 24 * 3600,
            "week": 7 * 24 * 3600,
            "month": 30 * 24 * 3600,
        }
        if recency not in windows:
            return None
        cutoff_ts = now_ts - windows[recency]
        return {"published_ts": {"$gte": cutoff_ts}}

    def retrieve(
        self,
        query: str,
        k: int = 8,
        recency: str = "all",
        category: Optional[str] = None,
        source: Optional[str] = None,
        recency_boost: float = 0.25,
    ) -> List[Document]:
        """
        Hybrid retrieval: similarity search with optional metadata filters
        and a post-retrieval recency re-rank.
        """
        where_filter = {}
        recency_f = self._recency_filter(recency)
        if recency_f:
            where_filter.update(recency_f)
        if category:
            where_filter["category"] = {"$eq": category}
        if source:
            where_filter["source"] = {"$eq": source}

        fetch_k = k * 3  # Over-fetch then re-rank
        try:
            if where_filter:
                docs_and_scores = self.store.similarity_search_with_score(
                    query, k=fetch_k, filter=where_filter
                )
            else:
                docs_and_scores = self.store.similarity_search_with_score(
                    query, k=fetch_k
                )
        except Exception as e:
            logger.warning(f"Filtered search failed, falling back: {e}")
            docs_and_scores = self.store.similarity_search_with_score(query, k=fetch_k)

        if not docs_and_scores:
            return []

        # Re-rank: combine similarity + recency boost
        now_ts = datetime.now(timezone.utc).timestamp()
        ranked = []
        for doc, sim_score in docs_and_scores:
            pub_ts = doc.metadata.get("published_ts", 0)
            age_hours = (now_ts - pub_ts) / 3600 if pub_ts else 9999
            # Recency score: 1.0 if <1h old, decays to 0 at 168h (1 week)
            recency_score = max(0.0, 1.0 - age_hours / 168)
            combined = (1 - recency_boost) * (1 - sim_score) + recency_boost * recency_score
            ranked.append((doc, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return knowledge base statistics."""
        try:
            result = self.store._collection.get(include=["metadatas"])
            metadatas = result["metadatas"] or []
            total_chunks = len(metadatas)

            now_ts = datetime.now(timezone.utc).timestamp()
            last_24h = sum(
                1 for m in metadatas
                if (now_ts - m.get("published_ts", 0)) < 86400
            )
            last_week = sum(
                1 for m in metadatas
                if (now_ts - m.get("published_ts", 0)) < 7 * 86400
            )

            sources = {}
            categories = {}
            for m in metadatas:
                s = m.get("source", "Unknown")
                c = m.get("category", "Unknown")
                sources[s] = sources.get(s, 0) + 1
                categories[c] = categories.get(c, 0) + 1

            # Most recent article
            pub_timestamps = [m.get("published_ts", 0) for m in metadatas if m.get("published_ts")]
            last_updated = (
                datetime.fromtimestamp(max(pub_timestamps), tz=timezone.utc).isoformat()
                if pub_timestamps else "N/A"
            )

            return {
                "total_chunks": total_chunks,
                "unique_articles": len({m.get("doc_id") for m in metadatas}),
                "last_24h_chunks": last_24h,
                "last_week_chunks": last_week,
                "sources": sources,
                "categories": categories,
                "last_updated": last_updated,
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"total_chunks": 0, "unique_articles": 0}

    def get_recent_articles(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Return most recent article metadata for 'What's New' panel."""
        try:
            cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
            result = self.store._collection.get(
                where={"published_ts": {"$gte": cutoff_ts}},
                include=["metadatas"],
            )
            metadatas = result["metadatas"] or []
            # Deduplicate by doc_id, keep most recent
            seen = {}
            for m in metadatas:
                did = m.get("doc_id", "")
                if did not in seen or m.get("published_ts", 0) > seen[did].get("published_ts", 0):
                    seen[did] = m
            articles = sorted(seen.values(), key=lambda x: x.get("published_ts", 0), reverse=True)
            return articles[:limit]
        except Exception as e:
            logger.error(f"Recent articles error: {e}")
            return []