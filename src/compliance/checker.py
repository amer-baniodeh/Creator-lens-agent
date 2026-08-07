"""
checker.py
----------
Compliance checker for German/EU healthcare advertising law.
Two-layer approach:
  1. Fast: regex match against a forbidden-phrase blocklist.
  2. RAG-grounded LLM: for cases the blocklist misses, retrieve the actual
     relevant legal provisions (HWG, UWG) from Pinecone and ask GPT to judge
     compliance using ONLY that retrieved text — not its general training
     knowledge of "EU law", which is unverified and can't be cited.

Used by notebook 03, notebook 05, and the check_compliance agent tool.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from src.utils.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from src.utils.logger import logger


# ── Blocklist ─────────────────────────────────────────────────────────────────
# Phrases that are categorically forbidden under German/EU health advertising
# law (HWG, UWG, Directive 2001/83/EC). Extend this list as you discover new
# edge cases. This is the fast, free, zero-latency first pass — real legal
# grounding happens in layer 2 below.

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

# ── Ungrounded LLM classifier (fallback when no legal corpus is ingested) ─────
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
    """Ungrounded classifier — relies on GPT's general training knowledge.
    Used only as a fallback when no legal corpus has been ingested yet."""
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
    return json.loads(response.choices[0].message.content)


# ── RAG-grounded LLM classifier ────────────────────────────────────────────────
_GROUNDED_CLASSIFIER_SYSTEM_PROMPT = """
You are a compliance expert in German and EU healthcare advertising law.
You will be given a piece of marketing/influencer content AND excerpts from
the actual relevant legal provisions (e.g. HWG, UWG). Judge compliance using
ONLY the provided legal excerpts — do not rely on general knowledge of the law.

The provided excerpts include internal numbering (Absatz numbers like "(1)",
"(2)", and numbered points like "1.", "2.", "9."). Cite the MOST SPECIFIC
sub-provision that supports your verdict, not just the top-level section —
e.g. "§11 Abs. 1 Nr. 9 HWG" (a specific numbered point), not just "§11 HWG".
If the violation spans a whole Absatz with no single numbered point fitting
cleanly, cite the Absatz — e.g. "§3 Abs. 1 HWG". Only fall back to a bare
section number ("§5a UWG") when the provision has no internal sub-numbering.

If the excerpts don't clearly cover the claim, say so honestly in "reason"
rather than guessing, and set "compliant" based on the best reading of what
was actually provided.

IMPORTANT — do not over-apply §11 Abs. 1 Nr. 7 HWG (health improved by use /
harmed by non-use). That provision targets GENERALIZED PROMOTIONAL PROMISES
about what will happen to the reader — not a first-person account of one
person's own subjective experience. A personal testimonial is only a
violation if it generalizes into a promise ("this WILL work for you") or
pairs with an explicit before/after or guarantee framing. Merely describing
how a product felt or worked for the speaker is NOT, by itself, a claim that
health is improved through use.

Examples (for calibration, not literal matches):
- COMPLIANT: "My skin feels softer since I started using this." — a personal,
  hedged, first-person observation. No generalized promise to the reader.
- COMPLIANT: "Results vary by skin type, but I noticed a difference in a
  few days." — explicitly hedges instead of promising an outcome.
- NON-COMPLIANT: "Use this or your skin will only get worse." — generalizes
  into a promotional promise/threat directed at the reader, not a personal
  account.
- NON-COMPLIANT: "Trust me, this WILL work for you." — direct guaranteed-
  outcome promise to the reader.

Respond ONLY with a JSON object in this exact format:
{
  "compliant": true or false,
  "reason": "one to two sentence explanation grounded in the cited provision(s)",
  "cited_sections": ["§11 Abs. 1 Nr. 9 HWG", "§5a Abs. 4 UWG"],
  "flagged_phrases": ["phrase1", "phrase2"]
}
""".strip()


def _retrieve_relevant_law(text: str, k: int = 3) -> list[dict]:
    """
    Embed the claim and retrieve the most relevant legal provisions from the
    eu-regulations Pinecone namespace (populated by notebook 05 / legal_docs.py).

    Returns an empty list if the legal corpus hasn't been ingested yet, or if
    retrieval fails for any reason — callers should fall back gracefully.
    """
    from src.ingestion.embedder import embed_texts, _get_pinecone_index
    from src.ingestion.legal_docs import LEGAL_NAMESPACE

    try:
        query_vec = embed_texts([text])[0]
        index = _get_pinecone_index()
        result = index.query(
            vector=query_vec,
            top_k=k,
            namespace=LEGAL_NAMESPACE,
            include_metadata=True,
        )
        return [
            {
                "law_name": m["metadata"].get("law_name", "Unknown"),
                "section": m["metadata"].get("section", "?"),
                "text": m["metadata"].get("text", ""),
                "score": m["score"],
            }
            for m in result.get("matches", [])
        ]
    except Exception as e:
        logger.warning(f"Legal RAG retrieval failed (has the legal corpus been ingested?): {e}")
        return []


def _llm_classify_grounded(text: str) -> dict:
    """
    RAG-grounded classifier: retrieves real legal text relevant to the claim,
    then asks GPT to judge compliance citing specific provisions.
    Falls back to the ungrounded classifier if no legal corpus is ingested yet.
    """
    relevant_law = _retrieve_relevant_law(text)

    if not relevant_law:
        result = _llm_classify(text)
        result["cited_sections"] = []
        result["grounded"] = False
        return result

    law_context = "\n\n".join(
        f"[{law['section']} {law['law_name']}]\n{law['text']}"
        for law in relevant_law
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": _GROUNDED_CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Relevant legal provisions:\n{law_context}\n\n"
                    f"Content to check:\n{text}"
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["grounded"] = True
    return result


def check_compliance(text: str, use_llm_fallback: bool = True) -> dict:
    """
    Check a piece of text for German/EU health advertising compliance.

    Args:
        text: The content to check (transcript chunk, creator script, brief draft).
        use_llm_fallback: If True, run the RAG-grounded LLM classifier even when
                          no blocklist match is found, to catch subtle violations.

    Returns:
        {
            "compliant": bool,
            "flagged_phrases": list[str],
            "cited_sections": list[str],
            "source": "blocklist" | "llm_rag" | "llm" | "clean",
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
            "cited_sections": [],
            "source": "blocklist",
            "reason": f"Text contains {len(unique_matches)} forbidden phrase(s) under German/EU health advertising law.",
        }

    # Layer 2: RAG-grounded LLM classifier for edge cases
    if use_llm_fallback:
        result = _llm_classify_grounded(text)
        result["source"] = "llm_rag" if result.get("grounded") else "llm"
        result.setdefault("cited_sections", [])
        if not result["compliant"]:
            logger.warning(f"LLM compliance flag: {result['reason']}")
        return result

    # Clean
    return {
        "compliant": True,
        "flagged_phrases": [],
        "cited_sections": [],
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

    phrases = ", ".join(f'"{p}"' for p in result.get("flagged_phrases", []))
    sections = ", ".join(result.get("cited_sections", []))

    lines = [
        f"🚨 Non-compliant (detected by {result['source']}).",
        f"Reason: {result['reason']}",
    ]
    if sections:
        lines.append(f"Relevant law: {sections}")
    if phrases:
        lines.append(f"Flagged phrases: {phrases}")

    return "\n".join(lines)
