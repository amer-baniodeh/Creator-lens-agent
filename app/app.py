"""
app.py
------
Streamlit UI for the Creative Intelligence Copilot.
Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.agent.agent import get_agent
from src.utils.logger import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Creative Intelligence Copilot",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _ensure_legal_corpus() -> dict:
    """
    Startup check: if the eu-regulations Pinecone namespace is empty, auto-ingest
    the bundled legal source docs (data/legal/*.txt) so grounded compliance checks
    work out of the box — no manual notebook run required after a fresh deploy.

    @st.cache_resource means this runs once per server process, not once per
    user session, so it won't re-check Pinecone on every page load.
    """
    from pinecone import Pinecone
    from src.utils.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
    from src.ingestion.legal_docs import ingest_legal_document, LEGAL_NAMESPACE

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    existing = stats.get("namespaces", {}).get(LEGAL_NAMESPACE, {}).get("vector_count", 0)

    if existing > 0:
        logger.info(f"Legal corpus already present ({existing} vectors in '{LEGAL_NAMESPACE}')")
        return {"status": "already_present", "vector_count": existing}

    logger.info(f"Legal corpus empty — auto-ingesting bundled documents into '{LEGAL_NAMESPACE}'")
    legal_dir = REPO_ROOT / "data" / "legal"
    documents = [
        {
            "file_path": str(legal_dir / "hwg.txt"),
            "law_name": "HWG",
            "source_url": "https://www.gesetze-im-internet.de/heilmwerbg/",
        },
        {
            "file_path": str(legal_dir / "uwg_5a.txt"),
            "law_name": "UWG",
            "source_url": "https://www.gesetze-im-internet.de/uwg_2004/__5a.html",
        },
        {
            "file_path": str(legal_dir / "uwg_3a.txt"),
            "law_name": "UWG",
            "source_url": "https://www.gesetze-im-internet.de/uwg_2004/__3a.html",
        },
    ]

    total = 0
    try:
        for doc in documents:
            summary = ingest_legal_document(**doc)
            total += summary["vectors_upserted"]
        logger.info(f"Auto-ingested legal corpus: {total} vectors")
        return {"status": "ingested", "vector_count": total}
    except Exception as e:
        logger.error(f"Legal corpus auto-ingestion failed: {e}")
        return {"status": "failed", "error": str(e)}


with st.spinner("Preparing legal knowledge base..."):
    _legal_corpus_status = _ensure_legal_corpus()

# ── Session state init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = get_agent(verbose=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_videos" not in st.session_state:
    st.session_state.ingested_videos = []


def _run_agent(prompt: str) -> str:
    """Run the agent with knowledge base context injected."""
    kb_context = ""
    if st.session_state.ingested_videos:
        video_list = "\n".join(
            f"- \"{v['title']}\" by {v.get('channel', 'Unknown')} ({v.get('url', 'N/A')})"
            for v in st.session_state.ingested_videos
        )
        kb_context = (
            f"[Knowledge base contains these videos:\n{video_list}\n"
            "Use query_corpus to search their transcripts. "
            "Always cite the video title, channel name, and URL in your answer.]\n\n"
        )
    return st.session_state.agent.invoke(
        {"input": kb_context + prompt}
    ).get("output", "No response generated.")


def _auto_summary(summary: dict) -> str:
    """Generate an automatic analysis of a freshly ingested video."""
    title = summary["title"]
    channel = summary.get("channel", "Unknown")
    url = summary.get("url", "N/A")

    prompt = (
        f'I just ingested the video "{title}" by {channel} ({url}). '
        "Please analyse it by doing the following:\n"
        "1. Use query_corpus to retrieve transcript content from this specific video.\n"
        "2. Use check_compliance on the retrieved content to check for EU healthcare advertising violations.\n"
        "3. Provide a structured summary with these sections:\n\n"
        f"**Video:** {title} by {channel}\n"
        f"**URL:** {url}\n\n"
        "**Compliance:** Are there any compliance issues? List any flagged phrases.\n\n"
        "**Narrative structure:** What is the story arc? (e.g. problem → journey → solution, "
        "before/after, testimonial, educational)\n\n"
        "**Brand fit:** Based on the content and tone, is this creator a good fit for a "
        "prescription skincare brand? Any concerns?\n\n"
        "**Key quotes:** Pull 2-3 notable quotes from the transcript.\n\n"
        "Keep it concise and actionable."
    )
    return _run_agent(prompt)


def _handle_ingestion(summary: dict):
    """Store ingestion result and run auto-summary."""
    st.session_state.ingested_videos.append(summary)
    st.success(
        f"Ingested **{summary['title']}** "
        f"by **{summary.get('channel', 'Unknown')}** — "
        f"{summary['chunk_count']} chunks"
    )
    with st.spinner("Analysing video for compliance, narrative & brand fit..."):
        try:
            analysis = _auto_summary(summary)
            st.session_state.messages.append({
                "role": "assistant",
                "content": analysis,
            })
            st.rerun()
        except Exception as e:
            logger.error(f"Auto-summary failed: {e}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Add content")

    ingest_tab, paste_tab = st.tabs(["YouTube URL", "Paste transcript"])

    with ingest_tab:
        url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        url_title = st.text_input("Title (optional)", placeholder="Formel Skin — Patient Story", key="url_title")

        if st.button("Ingest from YouTube", use_container_width=True):
            if url_input:
                with st.spinner("Fetching transcript and uploading to Pinecone..."):
                    try:
                        from src.ingestion.embedder import ingest_video
                        summary = ingest_video(url_input, url_title or None)
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                        summary = None
                if summary:
                    _handle_ingestion(summary)
            else:
                st.warning("Please enter a YouTube URL.")

    with paste_tab:
        st.caption(
            "YouTube blocks requests from cloud servers. "
            "Paste the transcript here instead — copy it from YouTube's "
            "\"Show transcript\" button on the video page."
        )
        paste_title = st.text_input("Video title", placeholder="Formel Skin — My Acne Journey", key="paste_title")
        paste_channel = st.text_input("Channel name", placeholder="@SkinCareSarah", key="paste_channel")
        paste_url = st.text_input("Video URL (optional)", placeholder="https://youtube.com/watch?v=...", key="paste_url")
        paste_text = st.text_area("Transcript", placeholder="Paste the full transcript here...", height=200)

        if st.button("Ingest transcript", use_container_width=True):
            if paste_title and paste_text:
                with st.spinner("Chunking and uploading to Pinecone..."):
                    try:
                        from src.ingestion.embedder import ingest_transcript
                        summary = ingest_transcript(
                            text=paste_text,
                            title=paste_title,
                            channel=paste_channel or "Unknown",
                            url=paste_url or "",
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                        summary = None
                if summary:
                    _handle_ingestion(summary)
            else:
                st.warning("Please enter a title and paste the transcript.")

    if st.session_state.ingested_videos:
        st.markdown("---")
        st.markdown("### Knowledge base")
        for v in st.session_state.ingested_videos:
            channel = v.get("channel", "Unknown")
            url = v.get("url", "")
            url_line = f"  `{url}`" if url else ""
            st.markdown(
                f"- **{v['title']}** by *{channel}*  \n"
                f"  {v['chunk_count']} chunks{url_line}"
            )

    st.markdown("---")
    with st.expander("Legal corpus status"):
        status = _legal_corpus_status.get("status")
        count = _legal_corpus_status.get("vector_count", 0)
        if status == "already_present":
            st.caption(f"✅ {count} legal provisions loaded (HWG, UWG §5a, §3a)")
        elif status == "ingested":
            st.caption(f"✅ Auto-ingested {count} legal provisions on startup")
        else:
            st.caption(f"⚠️ Legal corpus unavailable — {_legal_corpus_status.get('error', 'unknown error')}")
            st.caption("Compliance checks will fall back to ungrounded LLM judgment.")

# ── Main chat area ───────────────────────────────────────────────────────────
st.markdown("## Creative Intelligence Copilot")
st.caption("Ingest a YouTube video to get an automatic compliance, narrative, and brand-fit analysis — then ask follow-up questions.")

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_compliance_flag"):
            st.error(msg["content"])
        elif msg.get("is_compliance_pass"):
            st.success(msg["content"])
        else:
            st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about your content..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = _run_agent(user_input)
            except Exception as e:
                answer = f"Error: {e}"
                logger.error(f"Agent error: {e}")

        msg_data = {"role": "assistant", "content": answer}
        if "🚨" in answer or "Non-compliant" in answer:
            st.error(answer)
            msg_data["is_compliance_flag"] = True
        elif "✅" in answer or "Compliant" in answer:
            st.success(answer)
            msg_data["is_compliance_pass"] = True
        else:
            st.markdown(answer)

    st.session_state.messages.append(msg_data)
