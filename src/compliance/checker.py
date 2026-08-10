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

Verdicts are a 4-level graded scale (not binary), returned via OpenAI
Structured Outputs (a strict JSON schema, not just "JSON mode") so the verdict
is guaranteed to be one of the defined levels — not free-text dressed up as JSON.

Used by notebook 03, notebook 05, notebook 06, and the check_compliance agent tool.
"""

from __future__ import annotations

import re
from typing import Literal

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field

from src.utils.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from src.utils.logger import logger


def _parse_structured(client: OpenAI, model: str, messages: list[dict], response_format):
    """
    Wraps client.chat.completions.parse() with a fallback for reasoning-family
    models (e.g. gpt-5.x) that reject an explicit temperature=0 — some only
    support their default temperature. Try deterministic first, fall back to
    the model's default if it refuses.
    """
    try:
        return client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0,
            response_format=response_format,
        )
    except BadRequestError as e:
        if "temperature" in str(e).lower():
            logger.info(f"Model '{model}' doesn't support temperature=0 — retrying with default.")
            return client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
            )
        raise


# ── Verdict scale ────────────────────────────────────────────────────────────
VERDICT_LABELS: dict[int, str] = {
    0: "Fully compliant",
    1: "Compliant — minor note",
    2: "Grey area — needs legal review",
    3: "Not compliant",
}


class ComplianceClassification(BaseModel):
    """Structured Outputs schema — OpenAI enforces this shape and the verdict
    enum server-side; the model cannot return a value outside 0-3 or omit a
    field, unlike plain 'JSON mode' which only guarantees valid JSON syntax."""

    verdict: Literal[0, 1, 2, 3] = Field(
        description=(
            "0 = fully compliant. "
            "1 = compliant, but a minor stylistic/wording note worth flagging to the "
            "creative team (no legal risk). "
            "2 = grey area — the claim is genuinely ambiguous under the available legal "
            "text and should go to human legal review rather than be auto-decided. "
            "3 = not compliant."
        )
    )
    reason: str = Field(description="1-2 sentence explanation grounded in the cited provision(s), or in the blocklist match.")
    notes: str = Field(
        default="",
        description=(
            "Optional extra context — e.g. what would resolve a grey-area verdict, or "
            "what to tighten for a minor-note verdict. Empty string if nothing to add."
        ),
    )
    cited_sections: list[str] = Field(
        default_factory=list,
        description="Specific legal sub-provisions cited, e.g. '§11 Abs. 1 Nr. 9 HWG'. Empty if none apply.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        description="Specific problematic phrases quoted from the text, if any.",
    )


# ── Blocklist ─────────────────────────────────────────────────────────────────
# Phrases that are categorically forbidden under German/EU health advertising
# law (HWG, UWG, Directive 2001/83/EC). Extend this list as you discover new
# edge cases. This is the fast, free, zero-latency first pass — real legal
# grounding happens in layer 2 below. A blocklist hit is always verdict 3.

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

# ── Few-shot exemplars (one per verdict level) ─────────────────────────────────
# Written fresh, deliberately NOT reused from data/eval/compliance_labeled_set.json
# or the holdout set — using eval examples as prompt content would let the model
# "see the answers" and make the eval numbers meaningless.
_FEW_SHOT_EXAMPLES = """
Calibration examples (for the verdict scale, not literal phrase matches):

Text: "I love how this moisturizer feels on my skin — lightweight and not greasy at all."
{"verdict": 0, "reason": "Purely sensory, subjective observation with no medical or efficacy claim.", "notes": "", "cited_sections": [], "flagged_phrases": []}

Text: "Honestly this is probably the best moisturizer I've tried in a while, you guys need it."
{"verdict": 1, "reason": "Personal opinion with hyperbolic language ('best... I've tried') but no medical or guarantee claim, so no legal risk.", "notes": "Consider softening the superlative for brand tone consistency — not a compliance issue.", "cited_sections": [], "flagged_phrases": []}

Text: "This helped calm my skin down during a really stressful week."
{"verdict": 2, "reason": "Ambiguous whether 'calm my skin' describes a subjective sensory experience or implies a therapeutic/calming health effect — the retrieved legal text does not clearly resolve this.", "notes": "Recommend legal review before approving for a health-adjacent product line.", "cited_sections": ["§11 Abs. 1 Nr. 7 HWG"], "flagged_phrases": []}

Text: "This cured my eczema in three days, doctors were shocked."
{"verdict": 3, "reason": "Direct cure claim for a medical condition combined with an implied medical-authority endorsement.", "notes": "", "cited_sections": ["§3 HWG", "§11 Abs. 1 Nr. 2 HWG"], "flagged_phrases": ["cured my eczema"]}
""".strip()

# ── Ungrounded LLM classifier (fallback when no legal corpus is ingested) ─────
_CLASSIFIER_SYSTEM_PROMPT = f"""
You are a compliance expert in EU healthcare advertising law (Directive 2001/83/EC).
Your job is to grade whether a piece of text makes any illegal or misleading
medical claims about a skincare product, using the 4-level verdict scale below.

A claim leans toward non-compliant (verdict 3) if it:
- Promises to cure, eliminate, or permanently resolve a medical condition
- Claims clinical or medical authority without substantiation
- Makes guarantees about results
- Implies the product can replace professional medical advice

{_FEW_SHOT_EXAMPLES}
""".strip()


def _llm_classify(text: str, model: str = OPENAI_LLM_MODEL) -> dict:
    """Ungrounded classifier — relies on GPT's general training knowledge.
    Used only as a fallback when no legal corpus has been ingested yet."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = _parse_structured(
        client, model,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Check this text for compliance:\n\n{text}"},
        ],
        response_format=ComplianceClassification,
    )
    return response.choices[0].message.parsed.model_dump()


# ── RAG-grounded LLM classifier ────────────────────────────────────────────────
_GROUNDED_CLASSIFIER_SYSTEM_PROMPT = f"""
You are a compliance expert in German and EU healthcare advertising law.
You will be given a piece of marketing/influencer content AND excerpts from
the actual relevant legal provisions (e.g. HWG, UWG). Grade compliance using
the 4-level verdict scale below, based ONLY on the provided legal excerpts —
do not rely on general knowledge of the law.

The provided excerpts include internal numbering (Absatz numbers like "(1)",
"(2)", and numbered points like "1.", "2.", "9."). Cite the MOST SPECIFIC
sub-provision that supports your verdict, not just the top-level section —
e.g. "§11 Abs. 1 Nr. 9 HWG" (a specific numbered point), not just "§11 HWG".
If the violation spans a whole Absatz with no single numbered point fitting
cleanly, cite the Absatz — e.g. "§3 Abs. 1 HWG". Only fall back to a bare
section number ("§5a UWG") when the provision has no internal sub-numbering.

If the excerpts don't clearly cover the claim, or the claim is genuinely
ambiguous under what's provided, use verdict 2 (grey area — needs legal
review) rather than guessing between compliant and not compliant.

IMPORTANT — do not over-apply §11 Abs. 1 Nr. 7 HWG (health improved by use /
harmed by non-use). That provision targets GENERALIZED PROMOTIONAL PROMISES
about what will happen to the reader — not a first-person account of one
person's own subjective experience. A personal testimonial is only a
violation if it generalizes into a promise ("this WILL work for you") or
pairs with an explicit before/after or guarantee framing. Merely describing
how a product felt or worked for the speaker is NOT, by itself, a claim that
health is improved through use.

{_FEW_SHOT_EXAMPLES}
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


def _llm_classify_grounded(text: str, model: str = OPENAI_LLM_MODEL) -> dict:
    """
    RAG-grounded classifier: retrieves real legal text relevant to the claim,
    then asks GPT to grade compliance citing specific provisions.
    Falls back to the ungrounded classifier if no legal corpus is ingested yet.
    """
    relevant_law = _retrieve_relevant_law(text)

    if not relevant_law:
        result = _llm_classify(text, model=model)
        result["cited_sections"] = []
        result["grounded"] = False
        return result

    law_context = "\n\n".join(
        f"[{law['section']} {law['law_name']}]\n{law['text']}"
        for law in relevant_law
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = _parse_structured(
        client, model,
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
        response_format=ComplianceClassification,
    )
    result = response.choices[0].message.parsed.model_dump()
    result["grounded"] = True
    return result


def _finalize(result: dict) -> dict:
    """Attach derived convenience fields shared by every code path."""
    verdict = result["verdict"]
    result["verdict_label"] = VERDICT_LABELS[verdict]
    result["compliant"] = verdict in (0, 1)
    result["needs_review"] = verdict == 2
    return result


def check_compliance(text: str, use_llm_fallback: bool = True, model: str = OPENAI_LLM_MODEL) -> dict:
    """
    Check a piece of text for German/EU health advertising compliance.

    Args:
        text: The content to check (transcript chunk, creator script, brief draft).
        use_llm_fallback: If True, run the RAG-grounded LLM classifier even when
                          no blocklist match is found, to catch subtle violations.
        model: OpenAI model to use for the LLM layer. Defaults to the configured
               OPENAI_LLM_MODEL — override for eval/comparison purposes only.

    Returns:
        {
            "verdict": int (0-3),
            "verdict_label": str,
            "compliant": bool,       # derived: verdict in (0, 1)
            "needs_review": bool,    # derived: verdict == 2
            "flagged_phrases": list[str],
            "cited_sections": list[str],
            "source": "blocklist" | "llm_rag" | "llm" | "clean",
            "reason": str,
            "notes": str,
        }
    """
    # Layer 1: fast blocklist regex — always verdict 3, no LLM call needed
    matches = _BLOCKLIST_PATTERN.findall(text)

    if matches:
        unique_matches = list(set(m.lower() for m in matches))
        logger.warning(f"Compliance blocklist hit: {unique_matches}")
        return _finalize({
            "verdict": 3,
            "flagged_phrases": unique_matches,
            "cited_sections": [],
            "source": "blocklist",
            "reason": f"Text contains {len(unique_matches)} forbidden phrase(s) under German/EU health advertising law.",
            "notes": "",
        })

    # Layer 2: RAG-grounded LLM classifier for edge cases
    if use_llm_fallback:
        result = _llm_classify_grounded(text, model=model)
        result["source"] = "llm_rag" if result.get("grounded") else "llm"
        result.setdefault("cited_sections", [])
        result = _finalize(result)
        if result["verdict"] >= 2:
            logger.warning(f"LLM compliance flag (verdict={result['verdict']}): {result['reason']}")
        return result

    # Clean — verdict 0, no LLM call
    return _finalize({
        "verdict": 0,
        "flagged_phrases": [],
        "cited_sections": [],
        "source": "clean",
        "reason": "No forbidden phrases detected.",
        "notes": "",
    })


def compliance_report(text: str) -> str:
    """
    Human-readable compliance report for a given text.
    Used as the agent tool's output string.
    """
    result = check_compliance(text)
    verdict = result["verdict"]
    icon = {0: "✅", 1: "✅", 2: "⚠️", 3: "🚨"}[verdict]

    lines = [f"{icon} {result['verdict_label']} (verdict {verdict}/3, detected by {result['source']})."]
    lines.append(f"Reason: {result['reason']}")

    if result.get("notes"):
        lines.append(f"Notes: {result['notes']}")

    sections = ", ".join(result.get("cited_sections", []))
    if sections:
        lines.append(f"Relevant law: {sections}")

    phrases = ", ".join(f'"{p}"' for p in result.get("flagged_phrases", []))
    if phrases:
        lines.append(f"Flagged phrases: {phrases}")

    return "\n".join(lines)
