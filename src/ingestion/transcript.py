"""
transcript.py
-------------
Fetches and chunks YouTube transcripts.
Used by notebook 01 and the ingest_video agent tool.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.request
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import logger

# EXPERIMENTAL — not committed. Diagnosed a real hang: on some networks (confirmed
# on this one) IPv6 to Google/YouTube is silently black-holed rather than refused,
# so socket.create_connection's default per-address loop can burn up to
# net.inet.tcp.keepinit (75s on macOS) on a dead IPv6 attempt before falling back
# to a working IPv4 one — and a single fetch_transcript() call makes several
# separate requests (watch page, possibly twice for the EU consent-cookie
# redirect, the innertube API call, the caption content itself), each able to
# independently eat that penalty, which is what stacks into multi-minute hangs.
#
# Reorders (doesn't remove) getaddrinfo results so IPv4 is always tried first —
# IPv6 stays available as a fallback for networks where it actually works, but
# on this one, connections should now succeed on the first attempt instead of
# waiting out a dead IPv6 one first. Patches the process-wide socket.getaddrinfo,
# since both urllib.request (used by fetch_video_info) and requests/urllib3
# (used by fetch_transcript) ultimately call through it.
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(results, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


socket.getaddrinfo = _getaddrinfo_ipv4_first


class _TimeoutHTTPAdapter(HTTPAdapter):
    """Applies a default timeout to every request unless the caller overrides
    it — youtube_transcript_api's internal requests.Session sets none at all,
    so without this, a request that DOES end up on a bad address (e.g. if a
    host ever has no IPv4 address at all) can hang indefinitely instead of
    failing fast."""

    def __init__(self, *args, timeout: float = 10, **kwargs):
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._timeout
        return super().send(request, **kwargs)


def _build_transcript_api() -> YouTubeTranscriptApi:
    session = requests.Session()
    adapter = _TimeoutHTTPAdapter(timeout=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return YouTubeTranscriptApi(http_client=session)


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

        def _extract(pattern: str, fallback: str) -> str:
            m = re.search(pattern, html)
            if not m:
                return fallback
            # The captured group is a raw JSON string body; wrap in quotes and
            # let json.loads handle all escape sequences (&, \", \\, etc).
            try:
                return json.loads(f'"{m.group(1)}"')
            except json.JSONDecodeError:
                return m.group(1)

        return {
            "video_title": _extract(r'"title":"(.*?)"', video_id),
            "channel": _extract(r'"ownerChannelName":"(.*?)"', "Unknown channel"),
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
