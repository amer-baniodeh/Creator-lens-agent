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

## 8. Cost and latency, from LangSmith traces

Tracing has been on since the very first notebook, so every ingestion, chat turn, and
compliance check run through the project so far had already been recorded — just never
pulled into a report. No new instrumentation was needed, only a summary built on top of
existing traces.

**What it found:** across ~2 days of development testing (398 traced runs), total GPT
spend was under a cent. A full agent chat turn (reasoning + tool calls + answer) costs
roughly $0.0005 and takes ~8-9 seconds on average; a direct RAG answer without agent
tool-selection overhead is cheaper and faster. Useful floor for later cost-at-scale
conversations, and a concrete latency number to set expectations against in a demo.

**Caveat surfaced:** embedding calls bypass the traced code path, so these cost figures
cover GPT generation only, not embeddings — noted rather than presented as complete.

## 9. RAG answer-quality evaluation — and where the real bottleneck is

The original RAG eval (built early on) judged "faithfulness" by showing the LLM judge
only the question and final answer — in practice that measures whether an answer
*sounds* grounded, not whether it verifiably is. Rebuilt it to show the judge the actual
retrieved text and check every claim against it, and replaced generic placeholder
questions with ones grounded in the real ingested videos (5 videos at time of writing,
spanning German and English content).

**What it found:** answer generation is fully trustworthy — a perfect faithfulness
score across all test questions, including correctly saying "not enough information"
rather than guessing when appropriate. Correctness (does the answer address what was
asked) was noticeably lower, and every miss traced back to retrieval, not generation:
one clear case where an English question failed to retrieve the German source video
that actually had the answer, and cases where a long video's full content wasn't
reachable within the small number of chunks pulled per query.

**Why this matters going forward:** it confirms, with evidence rather than assumption,
that generation quality is solid and retrieval is the thing to improve as the video
corpus grows — a much more specific target than "RAG quality" in the abstract.

## 10. Growing the corpus — retrieval degrades exactly as predicted

Ingested 6 more real skincare videos (English and German, different creators and
topics) to directly test the previous entry's hypothesis: does retrieval hold up as
more content is added? Re-ran the identical 10 questions from the baseline RAG eval
against the larger corpus (5 → 11 videos) as a controlled before/after comparison.

**What it found:** retrieval hit rate dropped from 90% to 60%. Two questions that
previously retrieved the correct video were completely crowded out once more,
topically-similar content existed to compete with — both affecting short videos with
few chunks, which get statistically outnumbered by longer new videos in similarity
search. Faithfulness dipped slightly too, in a way that first looked like a new
hallucination risk (the model appearing to attribute an answer to the wrong source) —
investigated properly in the next entry, and it turned out not to be that.

**Why this matters:** this is no longer a prediction — it's measured evidence that the
current fixed top-5 retrieval will not scale as the corpus grows toward the target of
20+ videos, without some form of improvement (larger k, reranking, or better filtering).
Gives a concrete, evidenced problem to solve rather than a vague "improve RAG" goal.

## 11. Hardening against hallucination — and catching two self-inflicted bugs along the way

Went back to the apparent hallucination signal from the previous entry with a proper
investigation rather than taking it at face value. Neither flagged case turned out to
be a real hallucination — one was the automated judge itself making an error, the
other was an incomplete eval label where the system's answer was actually correct.
Real hallucination did not occur in that test set.

Hardened the system anyway, since a prose-based "I don't know" has no structural
guarantee behind it: raised retrieval breadth (5→8 chunks), added a minimum relevance
score below which the system returns a fixed, explicit "nothing relevant found" signal
instead of ever handing the model weak matches to guess from, and made both the agent
and the RAG chain require explicit refusal — not a hedge — when retrieved content
doesn't support an answer. Verified directly: a genuinely out-of-scope question now
cleanly returns the explicit no-content signal rather than any risk of a guessed answer.

**Two bugs caught by testing this properly, not assumed away:** the automated judge
didn't know about the new "not found" signal and was scoring every honest refusal as
a hallucination — fixed. The first version of the refusal instruction over-corrected
and started declining even when the right information was clearly present but
surrounded by irrelevant retrieved content — rewritten to check every retrieved piece
individually before declining, which recovered most of the lost ground.

**Honest result, not a clean win:** faithfulness is now confirmed solid (not just
apparently solid). Correctness on the eval set is lower than before this work, not
higher — the system now gives fewer wrong or overconfident answers, at the cost of
more explicit "not found" responses. That's a real tradeoff, not a pure improvement,
and one specific gap remains open: a correct non-English chunk can still get missed
when it's surrounded by content in a different language. Documented rather than
papered over.

## Current state (updated as of the most recent entry above)

- **Compliance checker:** grounded in real law, evaluated, ~94-97% accurate on hand-
  labeled claims across two independent test sets.
- **Cost/latency:** near-negligible cost at current scale (sub-cent for all testing so
  far); full chat turn averages ~8-9 seconds.
- **RAG quality:** generation is confirmed faithful, with structural safeguards against
  guessing now in place (not just prompt instructions). Retrieval remains the
  bottleneck and measurably degrades as the corpus grows — the clearest concrete next
  improvement to make, alongside the known cross-lingual retrieval gap.
- **Corpus:** 11 videos ingested (up from 5), mix of English and German content.
- **Known gaps:** one known compliance recall miss (a professional-endorsement claim
  type); one known cross-lingual RAG retrieval gap. Neither fixed yet, both documented.
- **Not yet done:** improving retrieval to handle a larger corpus, YouTube ingestion on
  the cloud-hosted version is still blocked (paste-transcript workaround in place), a
  non-technical user test, and final presentation slides/rehearsal.
