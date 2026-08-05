"""
embedder.py
-----------
Embeds transcript chunks and upserts them to Pinecone.
Handles batching to stay within Pinecone's upsert limits.
"""

from __future__ import annotations

import hashlib
from typing import Generator

from openai import OpenAI
from pinecone import Pinecone

from src.utils.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from src.utils.logger import logger

# Pinecone recommends batches of 100 vectors max on the free tier
UPSERT_BATCH_SIZE = 100


def _get_pinecone_index():
    """Initialise and return the Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _make_chunk_id(video_id: str, chunk_index: int) -> str:
    """
    Deterministic ID for a chunk.
    Re-ingesting the same video will overwrite existing vectors (idempotent).
    """
    raw = f"{video_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _batch(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive batches of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using the configured OpenAI embedding model.

    Returns:
        List of embedding vectors (one per input text).
    """
    client = _get_openai_client()
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Embed and upsert a list of chunk dicts to Pinecone.

    Args:
        chunks: Output of transcript.chunk_transcript() —
                list of {"text": str, "metadata": dict}

    Returns:
        Total number of vectors upserted.
    """
    index = _get_pinecone_index()
    total_upserted = 0

    for batch in _batch(chunks, UPSERT_BATCH_SIZE):
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        vectors = [
            {
                "id": _make_chunk_id(
                    chunk["metadata"]["video_id"],
                    chunk["metadata"]["chunk_index"],
                ),
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"],  # store raw text for retrieval
                },
            }
            for chunk, embedding in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        logger.info(f"Upserted batch of {len(vectors)} vectors (total so far: {total_upserted})")

    return total_upserted


def ingest_transcript(text: str, title: str, channel: str = "Unknown", url: str = "") -> dict:
    """
    Ingest a raw transcript string (no YouTube API call).
    Used when YouTube blocks cloud IPs.
    """
    import hashlib
    from src.ingestion.transcript import chunk_transcript

    video_id = hashlib.md5(text[:200].encode()).hexdigest()[:11]
    metadata = {
        "video_id": video_id,
        "url": url,
        "title": title,
        "channel": channel,
        "char_count": len(text),
    }
    chunks = chunk_transcript(text, metadata)
    vectors_upserted = upsert_chunks(chunks)

    return {
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "url": url,
        "chunk_count": len(chunks),
        "vectors_upserted": vectors_upserted,
    }


def ingest_video(url: str, title: str | None = None) -> dict:
    """
    Full ingestion pipeline: URL → transcript → embed → Pinecone.
    Called by the ingest_video agent tool.

    Returns:
        Summary dict with video_id, title, chunk_count, vectors_upserted.
    """
    from src.ingestion.transcript import process_video

    chunks = process_video(url, title)
    vectors_upserted = upsert_chunks(chunks)

    summary = {
        "video_id": chunks[0]["metadata"]["video_id"],
        "title": chunks[0]["metadata"]["title"],
        "channel": chunks[0]["metadata"].get("channel", "Unknown channel"),
        "url": url,
        "chunk_count": len(chunks),
        "vectors_upserted": vectors_upserted,
    }

    logger.info(f"Ingestion complete: {summary}")
    return summary
