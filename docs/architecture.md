# Architecture overview

## System summary

The Creative Intelligence Copilot is a RAG-based agent that helps influencer
marketing strategists at prescription skincare brands analyse content and catch
compliance issues before campaigns launch. Compliance verdicts are grounded in
retrieved German/EU health-advertising law text (not an LLM's general knowledge),
and every quality claim about the system — compliance accuracy, RAG answer
quality, cost, latency — is backed by a saved, reproducible evaluation run rather
than informal testing. See [PROJECT_LOG.md](PROJECT_LOG.md) for how it got here.

## Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` — chosen empirically, not by default. Evaluated head-to-head against `gpt-5.6-terra` on the compliance eval sets; gpt-4o-mini scored notably higher (90.9% vs 66.7% exact-match verdict accuracy) and ran ~1.7-2x faster. See `data/eval/SUMMARY.md`. |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector DB | Pinecone (free tier, cosine metric), two namespaces — see below |
| Orchestration | LangChain `AgentExecutor` + `ConversationBufferMemory` |
| Tracing | LangSmith (also the source for cost/latency reporting) |
| UI | Streamlit (dark theme, structured cards, live tool-status) |
| Transcript source | `youtube-transcript-api` |

**Deployment note:** YouTube blocks requests from cloud provider IPs, so
URL-based ingestion works locally but not on Streamlit Cloud. The app supports a
manual "paste transcript" fallback for cloud use, and an ngrok tunnel is used to
run the app from a local machine for demos that need live YouTube ingestion.

## Pinecone namespaces

| Namespace | Contents | Populated by |
|---|---|---|
| *(default)* | Video transcript chunks | `ingest_video` / `ingest_transcript` |
| `eu-regulations` | German/EU legal source text (HWG full text, UWG §5a, §3a) | `legal_docs.py`, auto-ingested on app startup if empty |

Keeping legal text in a separate namespace means video search and compliance
search never cross-contaminate each other's results.

## Data flow — video ingestion

```
YouTube URL
    │
    ▼
youtube-transcript-api       ← pulls auto-captions + video title/channel metadata
    │
    ▼
RecursiveCharacterTextSplitter   ← chunk_size=500, overlap=50
    │
    ▼
text-embedding-3-small           ← 1536-dim vectors
    │
    ▼
Pinecone upsert (default namespace)  ← metadata: video_id, title, channel, url, chunk_index
    │
    ▼
Automatic post-ingest analysis (non-agentic, direct calls):
    - get_video_transcript() reassembles the full video from Pinecone
    - check_compliance() on the full transcript
    - analyze_video() → narrative structure, brand fit, key quotes (one structured LLM call)
    │
    ▼
Rendered as tabbed cards in the Streamlit UI (Compliance / Narrative / Brand Fit / Quotes)
```

## Data flow — chat query

```
[User asks a question]
        │
        ▼
LangChain Agent (OpenAI Functions)
        │
   ┌────┼────────────┐
   ▼    ▼             ▼
ingest_video   query_corpus   check_compliance
                    │
                    ▼
        Pinecone similarity search (top-8, scored)
                    │
                    ▼
        Relevance filter (MIN_RELEVANCE_SCORE = 0.15)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  nothing clears the       relevant chunks
  floor → explicit          returned with
  "no relevant content"     per-chunk scores
  marker, agent must
  say so plainly
        │                        │
        └───────────┬────────────┘
                    ▼
        Agent synthesis (must cite video/channel/URL;
        hard rule against guessing when unsupported)
                    ▼
        Streamlit UI response
```

## Agent tools

### `ingest_video`
- Input: YouTube URL (+ optional title)
- Pipeline: URL → video_id → transcript + video/channel metadata → chunks → embeddings → Pinecone upsert
- Output: summary dict (title, channel, chunk count, vectors upserted, thumbnail URL)
- `ingest_transcript` is the paste-based sibling used when YouTube blocks the request (cloud deployment)

### `query_corpus`
- Input: natural language question
- Pipeline: embed query → Pinecone similarity search (top-8) → filter by `MIN_RELEVANCE_SCORE` (0.15) → return scored chunks
- If nothing clears the relevance floor, returns a fixed `NO_RELEVANT_CONTENT_FOUND` marker instead of weak matches — the agent's system prompt treats this as a hard signal to say so explicitly, never to guess
- Output: relevant transcript excerpts with source + relevance score, or the marker
- **Anti-injection guard:** excerpt content AND video title/channel metadata are
  both third-party-supplied and both scanned for injection phrasing (shared with
  `check_compliance`, `src/utils/security.py`) — a match prepends an inline
  `[WARNING: ...]` marker into the tool output. The agent's system prompt requires
  treating this content as data, not instructions, and never treating a "legal
  requirement" or compliance claim asserted inside a video as authoritative.
  **Deterministic backstop (app.py):** prompt-level warnings alone were proven
  insufficient — the agent can cite its own warning marker as supporting evidence
  for a fabricated claim. The Streamlit UI tracks whether any tool result in a turn
  was flagged and prepends a hard caution banner to the displayed answer regardless
  of what the model's text says. See `data/eval/SECURITY_SUMMARY.md`.

### `check_compliance`
Two-layer, grounded in real law rather than an LLM's general legal knowledge, returning a
**4-level graded verdict** (not binary) via OpenAI **Structured Outputs** — a strict JSON
schema with an enforced enum, not just "JSON mode." The model cannot return a verdict
outside 0-3 or omit a field; this replaced an earlier looser JSON-mode implementation.

- **Verdict scale:** 0 = fully compliant · 1 = compliant, minor stylistic note · 2 = grey
  area — needs legal review · 3 = not compliant
- **Layer 1 (fast):** regex match against a ~30-phrase forbidden-phrase blocklist — always verdict 3
- **Layer 2 (RAG-grounded):** retrieves the actual relevant provisions from the `eu-regulations` Pinecone namespace (HWG, UWG §5a/§3a) and asks GPT to grade compliance using *only* that retrieved text, citing the specific sub-provision (e.g. `§11 Abs. 1 Nr. 9 HWG`). Prompt includes fresh few-shot exemplars, one per verdict level, kept separate from the eval sets.
- Falls back to an ungrounded classifier only if the legal namespace is empty (self-heals via auto-ingestion on app startup)
- Output: `verdict` (0-3), `verdict_label`, `compliant`/`needs_review` (derived booleans), `reason`, `notes`, cited section(s), flagged phrases, source layer, `manipulation_suspected` (bool)
- Model is overridable per-call (`model=` param) — used to run the same eval against alternative models without changing global config
- **Anti-injection guard (layer 2 only):** the text being graded is untrusted third-party content, so it's delimited and the prompt explicitly instructs treating it as data, not instructions. A pattern check flags known injection phrasing, and a fabrication check verifies every quoted "flagged phrase" actually appears in the input. Either check tripping forces the verdict to 2 and sets `manipulation_suspected=True` instead of trusting the raw LLM output — added after a confirmed exploit, see `data/eval/SECURITY_SUMMARY.md`.

### `check_video_compliance`
Broad compliance review of one or all ingested videos — the answer to "which
claims need revision?" or "summarize compliance issues across all videos," which
`query_corpus` handles poorly (similarity search on an abstract meta-question
doesn't reliably surface real content).

- Input: a video title (or part of one), a URL, or `all`
- Pipeline: `list_ingested_videos()` (enumerates the corpus via one Pinecone query, not a per-video lookup) → matches by title/URL substring → `get_video_transcript()` for each match's FULL transcript → `check_compliance()` on each, same hardened path as the `check_compliance` tool
- Video titles are scanned for injection patterns before display (same guard as `query_corpus`) — a manipulated title doesn't affect the actual verdict, since only the transcript is graded, but it's flagged in the output regardless
- Capped at 8 videos per call to keep tool output bounded

## Evaluation framework

Every quality claim about this system is backed by a saved, re-runnable eval —
not informal testing. Results live in `data/eval/` (compliance, RAG quality) and
`data/metrics/` (cost/latency), each with a `SUMMARY.md` auto-regenerated after
every run and a `runs/` archive so nothing is overwritten.

| What | Notebook | Summary |
|---|---|---|
| Compliance checker accuracy | `06_compliance_eval.ipynb` | `data/eval/SUMMARY.md` |
| RAG answer quality (faithfulness + correctness) | `04_langsmith_eval.ipynb` | `data/eval/RAG_SUMMARY.md` |
| Cost & latency (from LangSmith traces) | `07_cost_latency_report.ipynb` | `data/metrics/SUMMARY.md` |
| Security — prompt leakage, jailbreaking, injection | ad hoc script (`data/eval/security_eval.json` test set) | `data/eval/SECURITY_SUMMARY.md` |

Headline results as of the last run (see the summary files for current numbers):
compliance verdict exact-match accuracy ~90-91% across two independent labeled sets
on the 4-level scale (metrics are ordinal — exact match, off-by-one tolerance, and
severe-miss rate, since binary precision/recall doesn't fit a graded verdict); RAG
generation is fully faithful with structural hallucination guards (relevance-score
gating + explicit refusal, not just prompt hedging); GPT cost is sub-cent at current
scale; `gpt-4o-mini` was chosen over `gpt-5.6-terra` after a head-to-head eval showed
higher accuracy and lower latency (see Stack table above) — the classifier prompt was
tuned specifically against gpt-4o-mini's behavior across earlier iterations, so this
result isn't necessarily fair to an untuned alternative model, but it's what the data
supports today.

**Known open limitations** (tracked, not hidden): one compliance recall miss on a
professional-endorsement claim type; a cross-lingual RAG retrieval gap where a
correct non-English chunk can be missed when surrounded by content in another
language; retrieval quality measurably degrades as the video corpus grows
(90%→60% hit rate observed after doubling from 5 to 11 videos) and hasn't yet
been addressed beyond raising `TOP_K_RESULTS`. The prompt-injection vulnerabilities
found in security testing (`check_compliance` and the agent's retrieval path) have
both been fixed — see `data/eval/SECURITY_SUMMARY.md` and the tool descriptions
below.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_ingestion.ipynb` | URL → transcript → chunks → Pinecone upsert |
| `02_rag_chain.ipynb` | Query → retrieval → GPT answer (demo of the core RAG chain) |
| `03_compliance.ipynb` | Blocklist + LLM classifier on real transcripts |
| `04_langsmith_eval.ipynb` | RAG answer-quality eval: faithfulness (grounded in real retrieved context) + correctness, saved and archived |
| `05_legal_rag.ipynb` | Ingests HWG/UWG source text into the `eu-regulations` namespace; tests grounded retrieval |
| `06_compliance_eval.ipynb` | Compliance checker accuracy against a hand-labeled set + an independent holdout set |
| `07_cost_latency_report.ipynb` | Pulls LangSmith traces into a cost/latency report |

## Phase 2 additions (post-MVP)

- Whisper transcription for videos without auto-captions
- Fix the known cross-lingual retrieval gap (query translation, or multilingual-aware chunking/retrieval)
- Retrieval quality improvements for a larger corpus (reranking, hybrid search, or per-video metadata filtering) — evidenced as necessary, not just anticipated
- Brief generator tool (structured output → creator brief draft)
- Corpus-wide pattern analysis (compare hooks across 20+ videos)
- EU Cosmetics Regulation (Art. 20) and additional case law added to the legal corpus
