"""
app.py
------
Streamlit UI for the Creative Intelligence Copilot.
Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent.agent import get_agent
from src.utils.logger import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Creative Intelligence Copilot",
    page_icon="🎯",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = get_agent(verbose=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_videos" not in st.session_state:
    st.session_state.ingested_videos = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Ingest a video")
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    title_input = st.text_input("Title (optional)", placeholder="Formel Skin — Patient Story")

    if st.button("Ingest", use_container_width=True):
        if url_input:
            with st.spinner("Fetching transcript and uploading to Pinecone..."):
                try:
                    from src.ingestion.embedder import ingest_video
                    summary = ingest_video(url_input, title_input or None)
                    st.session_state.ingested_videos.append(summary)
                    st.success(
                        f"Ingested **{summary['title']}** — "
                        f"{summary['chunk_count']} chunks, "
                        f"{summary['vectors_upserted']} vectors"
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        else:
            st.warning("Please enter a YouTube URL.")

    if st.session_state.ingested_videos:
        st.markdown("---")
        st.markdown("### Knowledge base")
        for v in st.session_state.ingested_videos:
            st.markdown(
                f"- **{v['title']}** — {v['chunk_count']} chunks  \n"
                f"  `{v['url']}`"
            )

# ── Main chat area ───────────────────────────────────────────────────────────
st.markdown("## Creative Intelligence Copilot")
st.caption("Ask about hooks, narrative structure, compliance, and content patterns across your ingested videos.")

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_compliance_flag"):
            st.error(msg["content"])
        elif msg.get("is_compliance_pass"):
            st.success(msg["content"])
        else:
            st.markdown(msg["content"])

# Chat input (top-level — fixes the position bug)
if user_input := st.chat_input("Ask a question about your content..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Inject knowledge base context so the agent knows what's been ingested
                kb_context = ""
                if st.session_state.ingested_videos:
                    video_list = "\n".join(
                        f"- \"{v['title']}\" ({v['url']}, {v['chunk_count']} chunks)"
                        for v in st.session_state.ingested_videos
                    )
                    kb_context = (
                        f"\n\n[Knowledge base contains these videos:\n{video_list}\n"
                        "Use query_corpus to search their transcripts before answering. "
                        "Use check_compliance to verify any claims found.]\n\n"
                    )

                enriched_input = kb_context + user_input if kb_context else user_input
                response = st.session_state.agent.invoke({"input": enriched_input})
                answer = response.get("output", "No response generated.")
            except Exception as e:
                answer = f"Error: {e}"
                logger.error(f"Agent error: {e}")

        # Highlight compliance flags
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
