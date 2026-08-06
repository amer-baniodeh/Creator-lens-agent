"""
transcript.py
-------------
Fetches and chunks YouTube transcripts.
Used by notebook 01 and the ingest_video agent tool.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Optional
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    WEBSHARE_PROXY_USERNAME,
    WEBSHARE_PROXY_PASSWORD,
    HTTP_PROXY,
    HTTPS_PROXY,
)
from src.utils.logger import logger


def _build_transcript_api() -> YouTubeTranscriptApi:
    """
    Build a YouTubeTranscriptApi instance, optionally with a proxy.
    YouTube blocks cloud provider IPs (AWS, GCP, Azure). If proxy env vars are
    set, route requests through them so cloud deployments work.
    """
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        from youtube_transcript_api.proxies import WebshareProxyConfig
        logger.info("Using Webshare proxy for YouTube transcript requests")
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD,
            )
        )
    if HTTP_PROXY or HTTPS_PROXY:
        from youtube_transcript_api.proxies import GenericProxyConfig
        logger.info("Using generic HTTP proxy for YouTube transcript requests")
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=HTTP_PROXY or HTTPS_PROXY,
                https_url=HTTPS_PROXY or HTTP_PROXY,
            )
        )
    return YouTubeTranscriptApi()


def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from any standard YouTube URL format.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://youtube.com/shorts/VIDEO_ID
    """
    parsed = urlparse(url)

    if parsed.netloc in ("youtu.be",):
        return parsed.path.lstrip("/")

    if "youtube.com" in parsed.netloc:
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1]
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_video_info(video_id: str) -> dict:
    """Fetch video title and channel name from YouTube page metadata."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        title_match = re.search(r'"title":"(.*?)"', html)
        channel_match = re.search(r'"ownerChannelName":"(.*?)"', html)

        return {
            "video_title": title_match.group(1) if title_match else video_id,
            "channel": channel_match.group(1) if channel_match else "Unknown channel",
        }
    except Exception as e:
        logger.warning(f"Could not fetch video info for {video_id}: {e}")
        return {"video_title": video_id, "channel": "Unknown channel"}


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """
    Fetch the full transcript text for a YouTube video.

    Args:
        video_id: The YouTube video ID.
        languages: Preferred language codes, e.g. ["en", "de"]. Falls back to any available.

    Returns:
        The full transcript as a single string.

    Raises:
        NoTranscriptFound: If no transcript is available in any language.
        TranscriptsDisabled: If the video has transcripts disabled.
    """
    langs = languages or ["en", "de", "fr"]
    api = _build_transcript_api()

    try:
        transcript_list = api.fetch(video_id, languages=langs)
    except NoTranscriptFound:
        logger.warning(f"No transcript in {langs} for {video_id}, trying any available language")
        available = api.list(video_id)
        generated = [t for t in available if t.is_generated]
        if generated:
            transcript_list = generated[0].fetch()
        else:
            transcript_list = list(available)[0].fetch()

    text = " ".join(entry.text for entry in transcript_list)
    # Clean up common transcript artifacts (line breaks, multiple spaces)
    text = re.sub(r"\s+", " ", text).strip()

    logger.info(f"Fetched transcript for {video_id} — {len(text)} characters")
    return text


def chunk_transcript(
    text: str,
    metadata: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a transcript into overlapping chunks and attach metadata.

    Args:
        text: Full transcript string.
        metadata: Dict of video metadata (url, title, video_id, etc.)
        chunk_size: Target size in characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with keys: text, metadata (including chunk_index).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    documents = [
        {
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_total": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return documents


def process_video(url: str, title: Optional[str] = None) -> list[dict]:
    """
    Full pipeline: URL → transcript → chunks with metadata.
    This is the main entry point called by the ingest_video tool.

    Args:
        url: YouTube video URL.
        title: Optional human-readable title. If not provided, video_id is used.

    Returns:
        List of chunk dicts ready for embedding and upsert.
    """
    video_id = extract_video_id(url)
    video_info = fetch_video_info(video_id)
    transcript = fetch_transcript(video_id)

    metadata = {
        "video_id": video_id,
        "url": url,
        "title": title or video_info["video_title"],
        "channel": video_info["channel"],
        "char_count": len(transcript),
    }

    return chunk_transcript(transcript, metadata)
