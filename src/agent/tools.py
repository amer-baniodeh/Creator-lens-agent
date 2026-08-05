"""
tools.py
--------
Defines the three LangChain Tool objects used by the agent:
  1. ingest_video    — fetch, chunk, embed, upsert a YouTube video
  2. query_corpus    — semantic search over the Pinecone corpus
  3. check_compliance — flag illegal claims in any text

Each tool takes a single string input and returns a string output,
which is the format LangChain agents expect.
"""

from __future__ import annotations

import json

from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.utils.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    TOP_K_RESULTS,
)
from src.utils.logger import logger


# ── Tool 1: ingest_video ──────────────────────────────────────────────────────

def _ingest_video_fn(input_str: str) -> str:
    """
    Input: a YouTube URL (optionally with a title separated by '|')
    Examples:
        "https://youtube.com/watch?v=abc123"
        "https://youtube.com/watch?v=abc123 | My Video Title"
    """
    parts = [p.strip() for p in input_str.split("|")]
    url = parts[0]
    title = parts[1] if len(parts) > 1 else None

    try:
        from src.ingestion.embedder import ingest_video
        summary = ingest_video(url, title)
        return (
            f"Video ingested successfully.\n"
            f"Title: {summary['title']}\n"
            f"Video ID: {summary['video_id']}\n"
            f"Chunks created: {summary['chunk_count']}\n"
            f"Vectors upserted to Pinecone: {summary['vectors_upserted']}"
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return f"Ingestion failed: {str(e)}"


ingest_video_tool = Tool(
    name="ingest_video",
    func=_ingest_video_fn,
    description=(
        "Use this tool to ingest a YouTube video into the knowledge base. "
        "Input: a YouTube URL. Optionally add a title after a pipe character: "
        "'https://youtube.com/watch?v=xyz | Video Title'. "
        "Use this before querying a video that hasn't been ingested yet."
    ),
)


# ── Tool 2: query_corpus ──────────────────────────────────────────────────────

def _build_vectorstore() -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def _query_corpus_fn(question: str) -> str:
    """
    Input: a natural language question about the ingested video corpus.
    Returns the most relevant transcript excerpts with source metadata.
    """
    try:
        vectorstore = _build_vectorstore()
        docs = vectorstore.similarity_search(question, k=TOP_K_RESULTS)

        if not docs:
            return "No relevant content found in the knowledge base for that question."

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = f"{meta.get('title', 'Unknown')} ({meta.get('url', '')})"
            results.append(f"[{i}] Source: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"Query failed: {str(e)}"


query_corpus_tool = Tool(
    name="query_corpus",
    func=_query_corpus_fn,
    description=(
        "Use this tool to search the knowledge base of ingested YouTube videos. "
        "Input: a natural language question such as 'What hook did this creator use?' "
        "or 'Which videos mention purging?' or 'What claims are made about acne treatment?'. "
        "Returns the most relevant transcript excerpts."
    ),
)


# ── Tool 3: check_compliance ──────────────────────────────────────────────────

def _check_compliance_fn(text: str) -> str:
    """
    Input: any text to check for EU health advertising compliance.
    Returns a human-readable compliance report.
    """
    try:
        from src.compliance.checker import compliance_report
        return compliance_report(text)
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        return f"Compliance check failed: {str(e)}"


check_compliance_tool = Tool(
    name="check_compliance",
    func=_check_compliance_fn,
    description=(
        "Use this tool to check whether any text is compliant with EU healthcare "
        "advertising law. Input: a script, transcript excerpt, brief draft, or any "
        "text that makes claims about a skincare or medical product. "
        "Returns a compliance verdict with flagged phrases if non-compliant."
    ),
)


# ── Export all tools ──────────────────────────────────────────────────────────

ALL_TOOLS = [ingest_video_tool, query_corpus_tool, check_compliance_tool]
