# Project log

Source material for the final presentation. Each entry is a milestone: a decision, a
blocker and how it was resolved, or a meaningful system change. Kept intentionally
high-level — implementation detail belongs in code/commits, not here.

**How to keep this updated:** add a new entry at the bottom after anything
presentation-worthy — a new capability, a blocker + resolution, or a decision that
changed direction. Skip routine bug fixes and refactors; this is a highlight reel,
not a commit log.

---

## 1. MVP scaffold and first blocker

The project arrived with a working architecture already designed: YouTube ingestion →
Pinecone RAG → LangChain agent with three tools (ingest, query, compliance-check) →
Streamlit UI. First run failed — notebook 1 hit a dependency error.

**Cause:** `requirements.txt` was pinned to an older LangChain version, but the actual
environment had a newer major version installed, which had moved or renamed several
core imports (text splitter, agent classes, tool base class) and the YouTube transcript
library had also changed its API shape.

**Resolution:** updated all import paths and dependency versions to match what's
actually installed, then completed the remaining pipeline notebooks (RAG chain,
compliance checks, LangSmith evaluation) to match the original architecture doc.

## 2. First deploy — GitHub + Streamlit Cloud

Pushed to GitHub, deployed to Streamlit Cloud. Two blockers on the way:
- A heavy ML dependency pulled in by a notebook helper bloated the build and failed
  installation on the cloud — removed; it wasn't actually needed in production.
- The deployed app couldn't find the project's own source code — a Python path issue
  specific to how Streamlit Cloud runs the entry file. Fixed.

## 3. YouTube blocks cloud hosting providers (ongoing constraint)

Video ingestion worked locally but failed on Streamlit Cloud — YouTube blocks requests
from cloud provider IP ranges (AWS, GCP, etc.) across the board, not a bug specific to
this app.

**Options considered:** a paid rotating-proxy service would route around the block, but
adds a recurring cost dependency for an MVP.

**Decision:** kept URL-based ingestion for local use, added a manual "paste transcript"
fallback so the app is fully usable when deployed to the cloud, and used a local tunnel
(ngrok) to run the app from a laptop for live demos that need real YouTube ingestion.

## 4. Early UX bugs found through hands-on testing

Testing surfaced two usability problems: the chat input visually jumped position
mid-conversation, and the agent would ask the user to re-supply content that was
already sitting in the knowledge base instead of searching it.

**Fixed:** corrected the chat layout, and gave the agent explicit knowledge-base context
plus an instruction to search before asking questions. Also added automatic video/channel
source attribution on every answer, and an automatic compliance + narrative + brand-fit
summary that runs the moment a video is ingested, rather than waiting to be asked.

## 5. Compliance checker: from keyword list to grounded legal reasoning

**Problem identified:** the original compliance checker combined a hardcoded forbidden-
phrase list with an LLM asked to judge compliance from its own general training
knowledge of "EU law" — plausible-sounding, but not verifiable or citable.

**Decision:** sourced the actual German law text this brand needs to comply with (HWG
in full, plus the relevant UWG sections on misleading omissions and unfair competition)
and built a second retrieval pipeline just for legal text. Compliance verdicts are now
grounded in retrieved law and cite the specific provision, instead of a generic verdict.

**Still open:** the EU Cosmetics Regulation and a relevant 2025 court ruling were
identified as relevant but not yet sourced (the EU's official legal database blocked
automated fetching) — tracked as a follow-up, not blocking current functionality.

## 6. UI redesign — from functional to presentable

Full visual pass: dark theme, compliance results shown as structured cards (verdict,
cited law, flagged phrases) instead of plain text, the automatic video summary
reorganized into tabs, ingested videos shown as thumbnail cards with a compliance
status indicator, and the agent's tool usage made visible ("Searching transcripts...")
instead of a generic loading spinner.

**Why:** the app was functional but not something to demo or present as-is.

## 7. Building an evaluation framework

Up to this point, every claim about quality was informal — manual testing, no measured
metric for compliance accuracy, RAG answer quality, or cost. Before presenting the
system, we needed real numbers.

**Approach:** prioritized compliance-checker accuracy first, since it's the most
defensible metric available — ground truth comes from the actual law text, not opinion.
Built a hand-labeled test set of realistic marketing claims, ran the checker against it,
and measured precision/recall/F1.

**What the first measurement found:** the newly-grounded checker had perfect recall
(never missed a real violation) but weak precision — it was over-flagging harmless
personal testimonials ("my skin feels softer") as violations, over-applying one specific
legal provision too broadly.

**Fix and validation:** tightened the reasoning given to the model for that specific
provision, re-measured — precision improved substantially. To rule out the fix simply
memorizing the test examples rather than genuinely improving, built a second, independent
set of claims never used while designing the fix, and re-ran the check. Accuracy held up
on the unseen data, confirming the improvement generalizes.

**Outcome:** compliance accuracy went from ~73% to ~97% on the original test set, and
~94% on the unseen holdout set. All eval runs are now archived and summarized in one
place, so future changes can be measured against this baseline instead of relying on
manual spot-checks.

## Current state (updated as of the most recent entry above)

- **Compliance checker:** grounded in real law, evaluated, ~94-97% accurate on hand-
  labeled claims across two independent test sets.
- **Known gaps:** RAG answer-quality metrics and cost/latency tracking not yet built
  (LangSmith has been capturing this data throughout — just not yet pulled into a
  report). One known compliance recall miss (a professional-endorsement claim type)
  identified but not yet fixed.
- **Not yet done:** a broader ingested video corpus (currently only 1-2 videos — RAG
  quality is hard to judge meaningfully until this grows), YouTube ingestion on the
  cloud-hosted version is still blocked (paste-transcript workaround in place), a
  non-technical user test, and final presentation slides/rehearsal.
