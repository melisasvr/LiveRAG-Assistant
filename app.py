"""
LiveRAG Assistant — Streamlit Frontend
Real-time AI/ML news Q&A with live knowledge base.
"""

import os
import sys
import logging
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("liverag.app")

from config.settings import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    FETCH_INTERVAL_MINUTES, TOP_K_RETRIEVAL,
    APP_TITLE, APP_ICON, MAX_CHAT_HISTORY,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

  .live-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.2rem 1.8rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .live-badge {
    background: #238636; color: #fff;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
  .header-title { font-family:'Space Mono',monospace; font-size:1.4rem; font-weight:700; color:#58a6ff; margin:0; }
  .header-sub { font-size:0.82rem; color:#8b949e; margin:0; }

  .source-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:0.6rem 0.9rem; margin:0.3rem 0; font-size:0.82rem;
  }
  .source-card a { color:#58a6ff; text-decoration:none; }
  .source-card a:hover { text-decoration:underline; }
  .source-date { color:#3fb950; font-family:'Space Mono',monospace; font-size:0.72rem; }
  .news-source-tag {
    background:#21262d; border:1px solid #30363d; border-radius:4px;
    padding:1px 6px; font-size:0.72rem; color:#8b949e; font-family:'Space Mono',monospace;
  }

  .stat-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:0.8rem 1rem; text-align:center; }
  .stat-value { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:#58a6ff; }
  .stat-label { font-size:0.75rem; color:#8b949e; margin-top:0.2rem; }

  .news-item { background:#161b22; border-left:3px solid #3fb950; border-radius:0 8px 8px 0; padding:0.6rem 0.9rem; margin:0.4rem 0; font-size:0.85rem; }
  .kb-timestamp { background:#21262d; border:1px solid #30363d; border-radius:20px; padding:4px 12px; font-family:'Space Mono',monospace; font-size:0.72rem; color:#8b949e; display:inline-block; margin-top:0.4rem; }

  .stButton > button { background:#238636; color:#fff; border:none; border-radius:8px; font-weight:600; transition:background 0.2s; }
  .stButton > button:hover { background:#2ea043; }
  .stTabs [data-baseweb="tab"] { color:#8b949e; }
  .stTabs [aria-selected="true"] { color:#58a6ff; border-bottom-color:#58a6ff; }
  .stTextInput > div > div > input { background:#161b22 !important; border:1px solid #30363d !important; color:#e6edf3 !important; border-radius:8px !important; }
  .stSelectbox > div > div { background:#161b22 !important; border:1px solid #30363d !important; color:#e6edf3 !important; }
  hr { border-color:#30363d; }
  .status-ok { color:#3fb950; font-weight:600; }
  .status-err { color:#f85149; font-weight:600; }
  .status-wrn { color:#d29922; font-weight:600; }

  /* Chat bubbles */
  .user-bubble {
    background:#1c2128; border:1px solid #30363d;
    border-radius:18px 18px 4px 18px;
    padding:0.8rem 1.1rem; margin:0.5rem 0 0.5rem 3rem;
    color:#e6edf3; font-size:0.95rem;
  }
  .assistant-bubble {
    background:#0d2137; border:1px solid #1f6feb;
    border-radius:4px 18px 18px 18px;
    padding:0.8rem 1.1rem; margin:0.5rem 3rem 0.5rem 0;
    color:#e6edf3; font-size:0.95rem;
  }
  .bubble-label { font-size:0.72rem; color:#8b949e; margin-bottom:0.3rem; font-family:'Space Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "rag_chain": None,
        "scheduler": None,
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "ingestion_started": False,
        "whats_new_cache": None,
        "whats_new_ts": None,
        # Pending question from suggested-question buttons
        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Resource helpers ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading vector store…")
def _load_vector_store():
    from retrieval.vector_store import LiveVectorStore
    return LiveVectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )

def get_vector_store():
    if st.session_state.vector_store is None:
        st.session_state.vector_store = _load_vector_store()
    return st.session_state.vector_store

def get_rag_chain(api_key: str):
    if st.session_state.rag_chain is None and api_key:
        from retrieval.rag_chain import LiveRAGChain
        st.session_state.rag_chain = LiveRAGChain(
            groq_api_key=api_key,
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return st.session_state.rag_chain

def get_scheduler(vector_store):
    if st.session_state.scheduler is None:
        from ingestion.scheduler import LiveIngestionScheduler
        st.session_state.scheduler = LiveIngestionScheduler(
            vector_store=vector_store,
            interval_minutes=FETCH_INTERVAL_MINUTES,
        )
    return st.session_state.scheduler


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚡ LiveRAG Control")
        st.divider()

        api_key = st.text_input(
            "Groq API Key", value=st.session_state.groq_api_key,
            type="password", placeholder="gsk_...", help="Get yours at console.groq.com",
        )
        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key
            st.session_state.rag_chain = None

        st.divider()
        st.markdown("#### 🔍 Retrieval Settings")
        recency = st.selectbox("Time Filter", ["all", "24h", "week", "month"], index=0)
        top_k = st.slider("Results to retrieve", 3, 15, TOP_K_RETRIEVAL)
        recency_boost = st.slider("Recency boost", 0.0, 0.5, 0.25, 0.05)
        category = st.selectbox("Category filter", ["All", "AI/ML", "Tech News", "Research", "Industry"])
        category_filter = None if category == "All" else category

        st.divider()
        st.markdown("#### 📡 Live Ingestion")
        vs = get_vector_store()
        scheduler = get_scheduler(vs)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True):
                if not scheduler.is_running:
                    scheduler.start()
                    st.session_state.ingestion_started = True
                    st.success("Started!")
        with col2:
            if st.button("⏹ Stop", use_container_width=True):
                scheduler.stop()
                st.warning("Stopped.")

        if st.button("🔄 Fetch Now", use_container_width=True):
            scheduler.trigger_now()
            st.success("Triggered!")

        from ingestion.scheduler import get_status
        s = get_status().to_dict()
        st.markdown("**Status:**")
        if s["is_running"]:
            st.markdown('<span class="status-wrn">⏳ Running…</span>', unsafe_allow_html=True)
        elif s["last_error"]:
            st.markdown(f'<span class="status-err">❌ {s["last_error"][:60]}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-ok">{s["status_msg"]}</span>', unsafe_allow_html=True)
        if s["last_run"]:
            st.caption(f"Last: {s['last_run'][:16].replace('T',' ')} UTC")
        if s["total_runs"] > 0:
            st.caption(f"Runs: {s['total_runs']} | New: {s['last_new_articles']} | Skip: {s['last_skipped']}")

        st.divider()
        st.markdown("#### 📊 Knowledge Base")
        stats = vs.get_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Articles", stats.get("unique_articles", 0))
            st.metric("Last 24h", stats.get("last_24h_chunks", 0))
        with c2:
            st.metric("Chunks", stats.get("total_chunks", 0))
            st.metric("Last Week", stats.get("last_week_chunks", 0))

        lu = stats.get("last_updated", "N/A")
        if lu != "N/A":
            lu = lu[:16].replace("T", " ") + " UTC"
        st.caption(f"Last indexed: {lu}")
        st.divider()
        st.caption(f"LiveRAG v1.0 · {GROQ_MODEL}")

    return recency, top_k, recency_boost, category_filter


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for s in sources:
            st.markdown(
                f'<div class="source-card">'
                f'<a href="{s["url"]}" target="_blank">📰 {s["title"][:80]}</a><br>'
                f'<span class="news-source-tag">{s["source"]}</span>'
                f'&nbsp;<span class="source-date">{s["published"]}</span>'
                f'&nbsp;<span class="news-source-tag">{s["category"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def run_query(question: str, vs, api_key: str, recency, top_k, recency_boost, category_filter):
    """Run retrieval + generation. Returns (answer, sources)."""
    docs = vs.retrieve(
        query=question, k=top_k, recency=recency,
        category=category_filter, recency_boost=recency_boost,
    )
    if not docs:
        return (
            "⚠️ No relevant articles found. Try 'Fetch Now' in the sidebar or broaden the time filter.",
            []
        )
    rag = get_rag_chain(api_key)
    answer, sources = rag.answer(question, docs)
    return answer, sources


# ── Tab: Chat ─────────────────────────────────────────────────────────────────

def render_chat_tab(recency, top_k, recency_boost, category_filter):
    vs = get_vector_store()
    api_key = st.session_state.groq_api_key

    # ── Process any pending question from suggestion buttons ──
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("🔍 Retrieving + generating…"):
                answer, sources = run_query(question, vs, api_key, recency, top_k, recency_boost, category_filter)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "sources": sources, "timestamp": now_str,
            })

    # ── Suggested questions (only when history is empty) ──
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Suggested Questions")
        suggestions = [
            "What are the latest breakthroughs in large language models?",
            "What new AI products were announced this week?",
            "What does recent research say about AI safety?",
            "Which AI startups raised funding recently?",
            "What are the latest developments in multimodal AI?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.rerun()
        st.divider()

    # ── Render chat history ──
    for msg in st.session_state.chat_history[-MAX_CHAT_HISTORY:]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-label">🧑 You</div>'
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Render assistant content safely using st.container
            st.markdown('<div class="bubble-label">⚡ LiveRAG Assistant</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"])
                if msg.get("timestamp"):
                    st.markdown(
                        f'<span class="kb-timestamp">🕐 Knowledge as of {msg["timestamp"]}</span>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Input form ──
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            question = st.text_input(
                "question", label_visibility="collapsed",
                placeholder="Ask about AI, ML, or tech news…",
            )
        with col_btn:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            return

        # Add user message to history immediately
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🔍 Retrieving live knowledge… 🤖 Generating answer…"):
            answer, sources = run_query(question, vs, api_key, recency, top_k, recency_boost, category_filter)

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "sources": sources, "timestamp": now_str,
        })
        st.rerun()

    # ── Clear button ──
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Tab: What's New ───────────────────────────────────────────────────────────

def render_whats_new_tab():
    vs = get_vector_store()
    api_key = st.session_state.groq_api_key

    st.markdown("### 📰 What's New Today")
    hours = st.select_slider("Time window", [6, 12, 24, 48, 72], value=24)

    if st.button("✨ Generate Briefing"):
        if not api_key:
            st.warning("Enter your Groq API key to generate a briefing.")
        else:
            docs = vs.get_recent_articles(hours=hours, limit=20)
            if not docs:
                st.info("No recent articles yet. Run ingestion first.")
            else:
                from langchain_core.documents import Document
                doc_objs = [
                    Document(page_content=f"{m.get('title','')} — {m.get('source','')}", metadata=m)
                    for m in docs
                ]
                with st.spinner("Generating briefing…"):
                    rag = get_rag_chain(api_key)
                    briefing = rag.whats_new(doc_objs, hours=hours)
                st.session_state.whats_new_cache = briefing
                st.session_state.whats_new_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if st.session_state.whats_new_cache:
        st.markdown(
            f'<span class="kb-timestamp">Generated: {st.session_state.whats_new_ts}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.whats_new_cache)

    st.divider()
    st.markdown("#### 🕐 Recently Indexed Articles")
    recent = vs.get_recent_articles(hours=hours, limit=15)
    if recent:
        for m in recent:
            pub = m.get("published_at", "")[:10]
            st.markdown(
                f'<div class="news-item">'
                f'<a href="{m.get("url","#")}" target="_blank" style="color:#58a6ff;text-decoration:none;">'
                f'{m.get("title","Untitled")[:100]}</a><br>'
                f'<span class="news-source-tag">{m.get("source","")}</span>'
                f'&nbsp;<span class="source-date">{pub}</span>'
                f'&nbsp;<span class="news-source-tag">{m.get("category","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No recent articles found. Run ingestion to populate the knowledge base.")


# ── Tab: Live vs Static ───────────────────────────────────────────────────────

def render_compare_tab():
    vs = get_vector_store()
    api_key = st.session_state.groq_api_key

    st.markdown("### ⚡ Live RAG vs Static RAG Demo")
    st.markdown(
        "This demo shows the **concrete value** of Live RAG over a static knowledge base. "
        "Ask a time-sensitive question and see how fresh data changes the answer."
    )

    demo_questions = [
        "What are the most recent LLM model releases?",
        "What AI funding rounds happened recently?",
        "What new AI safety research has been published?",
        "Which companies announced AI products this week?",
    ]

    q = st.selectbox("Choose a demo question or type your own:", ["Custom…"] + demo_questions)
    if q == "Custom…":
        q = st.text_input("Your question:", placeholder="Ask something time-sensitive…")

    hours = st.slider("Live context window (hours)", 12, 168, 48)

    if st.button("🚀 Run Comparison") and q:
        if not api_key:
            st.error("Groq API key required.")
            return
        docs = vs.retrieve(q, k=8, recency="week", recency_boost=0.4)
        if not docs:
            st.warning("No articles yet. Run ingestion first.")
            return
        rag = get_rag_chain(api_key)
        with st.spinner("Running comparison…"):
            comparison = rag.live_vs_static_demo(q, docs, hours=hours)
        st.divider()
        st.markdown(comparison)
        st.divider()
        st.markdown("**📚 Live Context Used:**")
        sources, seen = [], set()
        for doc in docs:
            url = doc.metadata.get("url", "")
            if url not in seen:
                seen.add(url)
                sources.append({
                    "title": doc.metadata.get("title", ""),
                    "source": doc.metadata.get("source", ""),
                    "url": url,
                    "published": doc.metadata.get("published_at", "")[:10],
                    "category": doc.metadata.get("category", ""),
                })
        render_sources(sources)


# ── Tab: Analytics ────────────────────────────────────────────────────────────

def render_analytics_tab():
    vs = get_vector_store()
    stats = vs.get_stats()

    st.markdown("### 📊 Knowledge Base Analytics")
    cols = st.columns(4)
    for col, (label, val, icon) in zip(cols, [
        ("Total Articles", stats.get("unique_articles", 0), "📄"),
        ("Total Chunks",   stats.get("total_chunks", 0),    "🧩"),
        ("Last 24h",       stats.get("last_24h_chunks", 0), "🔥"),
        ("Last Week",      stats.get("last_week_chunks", 0),"📅"),
    ]):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{icon} {val:,}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### By Category")
        categories = stats.get("categories", {})
        if categories:
            import plotly.express as px, pandas as pd
            df = pd.DataFrame([{"Category": k, "Chunks": v} for k, v in categories.items()]).sort_values("Chunks", ascending=False)
            fig = px.bar(df, x="Category", y="Chunks", color="Chunks", color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", margin=dict(t=20,b=20,l=20,r=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    with col_r:
        st.markdown("#### Top Sources")
        sources = stats.get("sources", {})
        if sources:
            import plotly.express as px, pandas as pd
            df = pd.DataFrame([{"Source": k, "Chunks": v} for k, v in sources.items()]).sort_values("Chunks", ascending=False).head(10)
            fig = px.bar(df, x="Chunks", y="Source", orientation="h", color="Chunks", color_continuous_scale="Greens", template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", margin=dict(t=20,b=20,l=20,r=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    from ingestion.scheduler import get_status
    status = get_status()
    if status.history:
        st.divider()
        st.markdown("#### Ingestion History")
        import pandas as pd, plotly.graph_objects as go
        df = pd.DataFrame(status.history)
        df["time"] = pd.to_datetime(df["time"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["time"], y=df["new"], name="New Articles", marker_color="#238636"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                          margin=dict(t=20,b=20,l=20,r=20), xaxis_title="Run Time", yaxis_title="New Articles")
        st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        f'<div class="live-header">'
        f'<span class="live-badge">● LIVE</span>'
        f'<div><p class="header-title">⚡ {APP_TITLE}</p>'
        f'<p class="header-sub">Real-time AI &amp; ML news · Powered by Groq + ChromaDB + LangChain</p></div>'
        f'<div style="margin-left:auto"><span class="kb-timestamp">{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    recency, top_k, recency_boost, category_filter = render_sidebar()

    tab_chat, tab_new, tab_compare, tab_analytics = st.tabs([
        "💬 Chat", "📰 What's New", "⚡ Live vs Static", "📊 Analytics",
    ])

    with tab_chat:
        render_chat_tab(recency, top_k, recency_boost, category_filter)
    with tab_new:
        render_whats_new_tab()
    with tab_compare:
        render_compare_tab()
    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()