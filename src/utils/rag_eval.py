"""
rag_eval.py
-----------
RAG answer-quality evaluation: retrieves + answers a question exactly like the
production RAG chain, then judges the result on two axes:
  - faithfulness: is the answer actually supported by the retrieved context,
    or does it drift into unsupported/hallucinated claims?
  - correctness: does the answer address what was asked?

The key methodological point versus a naive "ask GPT if this seems faithful"
check: the faithfulness judge is shown the ACTUAL retrieved context alongside
the answer, and asked to verify each claim against it — not just judge
plausibility. Grounded faithfulness, not vibes.
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from src.utils.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_LLM_MODEL,
    PINECONE_INDEX_NAME,
    TOP_K_RESULTS,
    MIN_RELEVANCE_SCORE,
)

# Fixed, structured marker for "nothing supports an answer" — matches the one
# used by the production query_corpus tool (src/agent/tools.py). Detectable
# programmatically, unlike freeform "I don't know" prose.
NOT_FOUND_MARKER = "NOT_FOUND_IN_KNOWLEDGE_BASE"

_RAG_ANSWER_PROMPT = """You are a Creative Intelligence Copilot. Answer the question based ONLY on the
following transcript excerpts.

The context below is a set of separate blocks pulled from different videos —
most retrieved blocks will be irrelevant to this specific question, and that is
normal, not a sign the answer is missing. Check EVERY block individually before
deciding. If even ONE block directly answers the question, use it — do not let
the surrounding irrelevant blocks talk you out of an answer that IS present.

Only respond with EXACTLY this and nothing else if, after checking every single
block, none of them answer the question: {not_found_marker}
Do not guess or speculate when no block supports an answer — but do not decline
just because most of the context is unrelated noise, either.

Always cite which video/source the information comes from.

Context:
{{context}}

Question: {{question}}

Answer:""".format(not_found_marker=NOT_FOUND_MARKER)

_FAITHFULNESS_JUDGE_PROMPT = """You are checking whether an AI-generated answer is faithful to the source
context it was given — i.e. every factual claim in the answer is actually
supported by the context, with no unsupported additions or hallucinations.

Context the answer was generated from:
{context}

Answer to check:
{answer}

Respond ONLY with a JSON object:
{{
  "faithfulness_score": <1-5, where 5 = fully supported by context, 1 = mostly unsupported/hallucinated>,
  "unsupported_claims": ["claim 1 not found in context", ...],
  "reasoning": "one sentence explanation"
}}"""

_CORRECTNESS_JUDGE_PROMPT = """You are checking whether an AI-generated answer correctly addresses the
question asked, according to the given criteria for what a good answer should cover.

Question: {question}
Criteria for a correct answer: {criteria}
Actual answer given: {answer}

Respond ONLY with a JSON object:
{{
  "correctness_score": <1-5, where 5 = fully satisfies the criteria, 1 = does not address it>,
  "reasoning": "one sentence explanation"
}}"""


def retrieve_and_answer(question: str, k: int = TOP_K_RESULTS) -> dict:
    """
    Retrieve relevant chunks and generate an answer, capturing the exact
    context the answer was based on (needed for grounded faithfulness judging).

    Two layers against hallucination, matching the production query_corpus tool:
    1. Score gate — if nothing retrieved clears MIN_RELEVANCE_SCORE, skip
       generation entirely and return NOT_FOUND_MARKER without calling the LLM.
    2. Prompt instruction — even when context is retrieved, the LLM is told to
       return the same structured marker rather than guess if the context
       doesn't actually answer the question.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    scored_docs = vectorstore.similarity_search_with_score(question, k=k)
    relevant = [(doc, score) for doc, score in scored_docs if score >= MIN_RELEVANCE_SCORE]

    if not relevant:
        best_score = max((s for _, s in scored_docs), default=0.0)
        return {
            "question": question,
            "context": "",
            "answer": NOT_FOUND_MARKER,
            "retrieved_video_ids": [],
            "n_chunks_retrieved": 0,
            "best_score_below_floor": round(best_score, 4),
        }

    docs = [doc for doc, _ in relevant]
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('title', 'Unknown')}]\n{d.page_content}" for d in docs
    )
    retrieved_video_ids = sorted({d.metadata.get("video_id") for d in docs if d.metadata.get("video_id")})

    llm = ChatOpenAI(model=OPENAI_LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = _RAG_ANSWER_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt).content

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "retrieved_video_ids": retrieved_video_ids,
        "n_chunks_retrieved": len(docs),
    }


def _judge_json(prompt: str) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict, precise evaluation judge. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def judge_faithfulness(answer: str, context: str) -> dict:
    """
    Score whether every claim in the answer is actually supported by the retrieved context.

    An explicit NOT_FOUND_MARKER answer is trivially faithful by construction —
    it makes zero claims, so there is nothing to be unfaithful about. Short-circuit
    this deterministically rather than asking the LLM judge: an early version of
    this eval let the judge score NOT_FOUND answers itself, and it consistently
    misjudged them as "unsupported claims", which would silently punish the
    exact honest-refusal behavior this whole hardening effort was built to reward.
    """
    if answer.strip() == NOT_FOUND_MARKER:
        return {
            "faithfulness_score": 5,
            "unsupported_claims": [],
            "reasoning": "Explicit decline rather than a guess — trivially faithful, no claims made.",
        }
    return _judge_json(_FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer))


def judge_correctness(question: str, criteria: str, answer: str) -> dict:
    """Score whether the answer addresses the question per the given criteria."""
    return _judge_json(_CORRECTNESS_JUDGE_PROMPT.format(question=question, criteria=criteria, answer=answer))


def generate_rag_summary_md(eval_dir: str = "data/eval") -> Path:
    """
    Regenerate data/eval/RAG_SUMMARY.md from data/eval/runs/*_rag.json — the
    same one-glance-index pattern as the compliance eval summary, scoped to
    RAG answer-quality runs specifically (distinguished by filename suffix).
    """
    eval_path = Path(eval_dir)
    runs_dir = eval_path / "runs"
    run_files = sorted(runs_dir.glob("*_rag.json"))

    lines = [
        "# RAG answer quality — evaluation history",
        "",
        "Auto-generated by `src/utils/rag_eval.py` from `data/eval/runs/*_rag.json`. "
        "Do not edit by hand — re-run the generator instead (see notebook 04, final cell).",
        "",
        "**Corpus-size caveat:** with only a handful of videos ingested, this measures "
        "generation faithfulness/correctness given whatever gets retrieved — it does NOT "
        "prove retrieval quality will hold up once the corpus is much larger. Treat as a "
        "generation-quality baseline, not a retrieval benchmark.",
        "",
        "| Date (UTC) | N | Avg faithfulness | Avg correctness | Notes |",
        "|---|---|---|---|---|",
    ]

    for path in run_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        date = data["run_at"].replace("T", " ")[:16]
        lines.append(
            f"| {date} | {data['n']} | {data['avg_faithfulness']:.2f}/5 | "
            f"{data['avg_correctness']:.2f}/5 | {data.get('note', '—')} |"
        )

    lines.append("")
    summary_path = eval_path / "RAG_SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
