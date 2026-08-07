"""
config.py
---------
Single source of truth for all environment variables and settings.
Every other module imports from here — never call os.getenv directly elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Cloud, secrets are in st.secrets — copy them into os.environ
# so the rest of the codebase can use os.getenv uniformly.
try:
    import streamlit as st
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass


def _require(key: str) -> str:
    """Raise a clear error if a required env var is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Check your .env file against .env.example"
        )
    return value


# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "copilot-mvp")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = _require("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "copilot-mvp")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
# Raised from 5 to 8 after an eval showed short/low-chunk-count videos getting
# crowded out of top-5 once the corpus grew — see data/eval/RAG_SUMMARY.md.
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "8"))
# Coarse safety floor: retrieved chunks scoring below this are treated as "not
# relevant" rather than fed to the LLM as if they were good context. This does
# NOT reliably distinguish "right video" from "wrong but topically similar
# video" (both can score similarly) — it only catches queries with no
# meaningfully relevant content in the corpus at all. See query_corpus_fn.
MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.15"))
