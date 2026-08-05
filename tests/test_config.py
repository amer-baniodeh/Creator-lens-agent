"""
test_config.py
--------------
Sanity checks for environment variable loading.
Run with: pytest tests/test_config.py -v
"""

import pytest


def test_config_loads_without_error():
    """Config module should import cleanly when .env is present."""
    try:
        from src.utils import config
        assert config.OPENAI_LLM_MODEL == "gpt-4o-mini"
        assert config.CHUNK_SIZE == 500
        assert config.CHUNK_OVERLAP == 50
        assert config.TOP_K_RESULTS == 5
    except EnvironmentError as e:
        pytest.skip(f"Skipping — .env not configured: {e}")


def test_extract_video_id():
    """URL parsing should handle all common YouTube URL formats."""
    from src.ingestion.transcript import extract_video_id

    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    from src.ingestion.transcript import extract_video_id
    with pytest.raises(ValueError):
        extract_video_id("https://vimeo.com/123456")


def test_compliance_blocklist_hit():
    """Known forbidden phrases should always be flagged."""
    from src.compliance.checker import check_compliance

    result = check_compliance("This product clinically proven to cure acne.", use_llm_fallback=False)
    assert result["compliant"] is False
    assert result["source"] == "blocklist"
    assert len(result["flagged_phrases"]) > 0


def test_compliance_clean_text():
    """Generic marketing copy should pass the blocklist check."""
    from src.compliance.checker import check_compliance

    result = check_compliance(
        "I started using this routine three months ago and my skin feels healthier.",
        use_llm_fallback=False,
    )
    assert result["compliant"] is True
    assert result["source"] == "clean"
