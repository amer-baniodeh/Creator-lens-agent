"""
checker.py
----------
Compliance checker for EU healthcare advertising law.
Two-layer approach:
  1. Fast: regex match against a forbidden-phrase blocklist.
  2. LLM fallback: for ambiguous cases, ask GPT to classify the claim.

Used by notebook 03 and the check_compliance agent tool.
"""

from __future__ import annotations

import re

from openai import OpenAI

from src.utils.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from src.utils.logger import logger


# ── Blocklist ─────────────────────────────────────────────────────────────────
# Phrases that are categorically forbidden under EU health advertising law
# (Directive 2001/83/EC and national transpositions like HWG in Germany).
# Extend this list as you discover new edge cases.

FORBIDDEN_PHRASES: list[str] = [
    # Cure / treatment claims
    "cures",
    "cures acne",
    "eliminates acne",
    "treats acne permanently",
    "heals skin",
    "clears skin permanently",
    # Clinical / medical authority claims
    "clinically proven",
    "clinically tested",
    "dermatologist approved",
    "doctor recommended",
    "medically proven",
    "scientifically proven",
    "FDA approved",
    "CE certified treatment",
    # Guarantee language
    "guaranteed results",
    "results guaranteed",
    "100% effective",
    "works every time",
    "no side effects",
    # Diagnosis / prescription framing
    "diagnoses",
    "prescription-free cure",
    "replaces your doctor",
    "no need for a dermatologist",
    # Exaggerated before/after language
    "complete transformation",
    "permanent solution",
    "forever clear",
]

# Compile into a single regex for fast matching (case-insensitive, word boundaries)
_BLOCKLIST_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in FORBIDDEN_PHRASES) + r")\b",
    re.IGNORECASE,
)

# ── LLM classifier prompt ─────────────────────────────────────────────────────
_CLASSIFIER_SYSTEM_PROMPT = """
You are a compliance expert in EU healthcare advertising law (Directive 2001/83/EC).
Your job is to identify whether a piece of text makes any illegal or misleading 
medical claims about a skincare product.

A claim is non-compliant if it:
- Promises to cure, eliminate, or permanently resolve a medical condition
- Claims clinical or medical authority without substantiation
- Makes guarantees about results
- Implies the product can replace professional medical advice

Respond ONLY with a JSON object in this exact format:
{
  "compliant": true or false,
  "reason": "one sentence explanation",
  "flagged_phrases": ["phrase1", "phrase2"]
}
""".strip()


def _llm_classify(text: str) -> dict:
    """Use GPT to classify ambiguous compliance cases."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Check this text for compliance:\n\n{text}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    import json
    return json.loads(response.choices[0].message.content)


def check_compliance(text: str, use_llm_fallback: bool = True) -> dict:
    """
    Check a piece of text for EU health advertising compliance.

    Args:
        text: The content to check (transcript chunk, creator script, brief draft).
        use_llm_fallback: If True, run the LLM classifier even when no blocklist
                          match is found, to catch subtle violations.

    Returns:
        {
            "compliant": bool,
            "flagged_phrases": list[str],
            "source": "blocklist" | "llm" | "clean",
            "reason": str,
        }
    """
    # Layer 1: fast blocklist regex
    matches = _BLOCKLIST_PATTERN.findall(text)

    if matches:
        unique_matches = list(set(m.lower() for m in matches))
        logger.warning(f"Compliance blocklist hit: {unique_matches}")
        return {
            "compliant": False,
            "flagged_phrases": unique_matches,
            "source": "blocklist",
            "reason": f"Text contains {len(unique_matches)} forbidden phrase(s) under EU health advertising law.",
        }

    # Layer 2: LLM classifier for edge cases
    if use_llm_fallback:
        result = _llm_classify(text)
        result["source"] = "llm"
        if not result["compliant"]:
            logger.warning(f"LLM compliance flag: {result['reason']}")
        return result

    # Clean
    return {
        "compliant": True,
        "flagged_phrases": [],
        "source": "clean",
        "reason": "No forbidden phrases detected.",
    }


def compliance_report(text: str) -> str:
    """
    Human-readable compliance report for a given text.
    Used as the agent tool's output string.
    """
    result = check_compliance(text)

    if result["compliant"]:
        return "✅ Compliant. No forbidden phrases or illegal claims detected."

    phrases = ", ".join(f'"{p}"' for p in result["flagged_phrases"])
    return (
        f"🚨 Non-compliant (detected by {result['source']}).\n"
        f"Reason: {result['reason']}\n"
        f"Flagged phrases: {phrases}"
    )
