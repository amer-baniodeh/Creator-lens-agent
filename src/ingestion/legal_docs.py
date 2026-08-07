"""
legal_docs.py
-------------
Chunks and ingests German/EU legal source documents (HWG, UWG, ...) into a
separate Pinecone namespace, so compliance checks can be grounded in the
actual legal text via RAG instead of an LLM's unverified general knowledge.

Source files live in data/legal/*.txt (plain text, § markers preserved).
Run notebook 05 to (re-)ingest them after adding or updating a source file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.utils.logger import logger

# Separate namespace so legal text never mixes into video transcript search
# results (and vice versa) — same Pinecone index, isolated corpus.
LEGAL_NAMESPACE = "eu-regulations"

# Matches German law section markers on their own line: "§ 1", "§ 3a", "§ 12"
_SECTION_PATTERN = re.compile(r"^§\s*(\d+[a-z]?)\s*$", re.MULTILINE)


def chunk_legal_text(text: str, law_name: str, source_url: str = "") -> list[dict]:
    """
    Split a law's full text into per-section chunks (one § per chunk).

    Chunking by legal section — instead of fixed character count — keeps
    each chunk a complete, independently citable unit ("§3 HWG"), so
    compliance verdicts can point to a specific provision.

    Args:
        text: Full text of the law, with "§ N" markers on their own line.
        law_name: Short name for citation, e.g. "HWG", "UWG".
        source_url: Where the text was sourced from, for citation/audit.

    Returns:
        List of dicts: {"id": str, "text": str, "metadata": dict}
    """
    matches = list(_SECTION_PATTERN.finditer(text))

    if not matches:
        # No § markers found (e.g. a single-section extract) — one chunk.
        sections = [("1", text.strip())]
    else:
        sections = []
        for i, m in enumerate(matches):
            section_num = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((section_num, body))

    chunks = []
    for section_num, body in sections:
        section_label = f"§{section_num}"
        chunk_id = hashlib.md5(f"{law_name}_{section_label}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": f"{section_label} {law_name}: {body}",
            "metadata": {
                "law_name": law_name,
                "section": section_label,
                "source_url": source_url,
                "doc_type": "legal_regulation",
            },
        })

    logger.info(f"Chunked {law_name} into {len(chunks)} section(s)")
    return chunks


def ingest_legal_document(file_path: str, law_name: str, source_url: str = "") -> dict:
    """
    Read a plain-text legal document from disk, chunk it by section,
    embed, and upsert to the eu-regulations Pinecone namespace.

    Returns:
        Summary dict with law_name, section_count, vectors_upserted, namespace.
    """
    from src.ingestion.embedder import upsert_chunks

    text = Path(file_path).read_text(encoding="utf-8")
    chunks = chunk_legal_text(text, law_name, source_url)
    vectors_upserted = upsert_chunks(chunks, namespace=LEGAL_NAMESPACE)

    summary = {
        "law_name": law_name,
        "section_count": len(chunks),
        "vectors_upserted": vectors_upserted,
        "namespace": LEGAL_NAMESPACE,
    }
    logger.info(f"Legal ingestion complete: {summary}")
    return summary
