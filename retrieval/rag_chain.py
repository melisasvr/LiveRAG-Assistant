"""
LiveRAG — RAG Chain
LangChain + Groq LLM with time-aware prompting and source citations.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are LiveRAG Assistant, an AI expert on AI, machine learning, and technology news. 
You have access to a continuously updated knowledge base of fresh news articles and research papers.

KNOWLEDGE AS OF: {knowledge_timestamp}

Your job:
1. Answer questions using ONLY the provided context documents.
2. Always cite your sources with [Source Name, Date] format.
3. If documents are outdated or insufficient, say so clearly.
4. Highlight when information is very recent (< 24h old).
5. Be concise but comprehensive.

RECENCY CONTEXT: {recency_note}

If you cannot answer from the provided context, say: 
"I don't have current information on this topic. My knowledge base may not have indexed this yet. Try refreshing or broadening the time filter."
"""

HUMAN_PROMPT = """CONTEXT DOCUMENTS:
{context}

---
USER QUESTION: {question}

Provide a well-structured answer with:
- Direct answer to the question
- Key details from the sources  
- Source citations [Source, Date]
- Any caveats about recency or completeness
"""

WHATS_NEW_PROMPT = """You are summarizing today's top AI and technology news for a daily briefing.

ARTICLES FROM THE LAST {hours}h:
{context}

Create a concise "What's New" briefing with:
1. **Top Stories** (3-5 bullet points of the most important developments)
2. **Key Themes** (2-3 recurring themes across articles)
3. **Notable Companies/Research** mentioned

Keep it under 300 words. Use bullet points. Be factual and cite sources.
Knowledge as of: {timestamp}
"""

COMPARE_PROMPT = """You are demonstrating the difference between Live RAG and Static RAG.

LIVE RAG CONTEXT (fresh articles from the last {hours}h):
{live_context}

USER QUESTION: {question}

1. Answer using LIVE RAG context (cite dates of your sources)
2. Then note what a STATIC RAG system trained on older data WOULD have said (briefly)
3. Highlight what has CHANGED or is NEW since what static RAG would know

Be specific about how the live data changes or improves the answer.
"""


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class LiveRAGChain:
    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        logger.info(f"LiveRAGChain initialized with model: {model}")

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a readable context block."""
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            source = m.get("source", "Unknown")
            title = m.get("title", "Untitled")
            pub_iso = m.get("published_at", "")
            pub_str = pub_iso[:10] if pub_iso else "Unknown date"
            url = m.get("url", "")
            category = m.get("category", "")
            parts.append(
                f"[DOC {i}] {title}\n"
                f"Source: {source} | Date: {pub_str} | Category: {category}\n"
                f"URL: {url}\n"
                f"Content: {doc.page_content}\n"
            )
        return "\n---\n".join(parts)

    def _recency_note(self, docs: List[Document]) -> str:
        """Generate a recency summary for the system prompt."""
        if not docs:
            return "No documents retrieved."
        now_ts = datetime.now(timezone.utc).timestamp()
        ages_h = []
        for doc in docs:
            pub_ts = doc.metadata.get("published_ts", 0)
            if pub_ts:
                ages_h.append((now_ts - pub_ts) / 3600)
        if not ages_h:
            return "Publication dates unknown."
        min_age = min(ages_h)
        max_age = max(ages_h)
        return (
            f"Youngest article: {min_age:.1f}h ago | "
            f"Oldest article: {max_age:.1f}h ago | "
            f"{sum(1 for a in ages_h if a < 24)} of {len(ages_h)} docs from last 24h."
        )

    def answer(
        self,
        question: str,
        docs: List[Document],
    ) -> Tuple[str, List[Dict]]:
        """
        Generate an answer from retrieved docs.
        Returns (answer_text, source_citations).
        """
        now = datetime.now(timezone.utc)
        context = self._format_context(docs)
        recency_note = self._recency_note(docs)

        prompt = self.chat_prompt.format_messages(
            knowledge_timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
            recency_note=recency_note,
            context=context,
            question=question,
        )

        response = self.llm.invoke(prompt)
        answer_text = response.content

        # Build source citations
        sources = []
        seen_urls = set()
        for doc in docs:
            url = doc.metadata.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                pub_iso = doc.metadata.get("published_at", "")
                pub_str = pub_iso[:10] if pub_iso else "Unknown"
                sources.append({
                    "title": doc.metadata.get("title", "Untitled"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "url": url,
                    "published": pub_str,
                    "category": doc.metadata.get("category", ""),
                })

        return answer_text, sources

    def whats_new(self, docs: List[Document], hours: int = 24) -> str:
        """Generate a 'What's New Today' briefing from recent articles."""
        if not docs:
            return "No recent articles found. Try running the ingestion pipeline."

        context = self._format_context(docs)
        now = datetime.now(timezone.utc)

        prompt = WHATS_NEW_PROMPT.format(
            hours=hours,
            context=context,
            timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
        )
        response = self.llm.invoke(prompt)
        return response.content

    def live_vs_static_demo(
        self, question: str, docs: List[Document], hours: int = 24
    ) -> str:
        """Demonstrate Live RAG vs Static RAG for a given question."""
        context = self._format_context(docs)
        prompt = COMPARE_PROMPT.format(
            hours=hours,
            live_context=context,
            question=question,
        )
        response = self.llm.invoke(prompt)
        return response.content