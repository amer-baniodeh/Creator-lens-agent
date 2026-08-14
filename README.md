# Creative Intelligence Copilot

An AI agent that helps influencer marketing strategists at a prescription skincare
brand review creator videos — pulling transcripts, summarizing them, and checking
the language against German/EU health-advertising law before a campaign goes live.

Built with LangChain, Pinecone, OpenAI, and Streamlit.

## Models

- **LLM:** `gpt-4o-mini` — picked after testing it head-to-head against `gpt-5.6-terra`
  on the compliance eval; gpt-4o-mini was both more accurate and faster.
- **Embeddings:** `text-embedding-3-small`

## Setup

```bash
git clone <repo-url>
cd copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI, Pinecone, and LangSmith keys
```

Create a Pinecone index named `copilot-mvp` (1536 dimensions, cosine metric).

## Running it

```bash
streamlit run app/app.py
```

Paste a YouTube link and it pulls the transcript, checks it for compliance, and
gives you a summary. You can also just chat with it — ask about a specific quote or
claim, or ask a broad question like "which claims need revision across all videos,"
which runs a full-transcript compliance review rather than a quick search.

YouTube blocks cloud servers from fetching videos, so live ingestion only works
locally (or through an ngrok tunnel). On the cloud-hosted version, use the
"paste transcript" option instead.

## Notebooks

`notebooks/` walks through the pipeline step by step, in order — ingestion, the
RAG chain, compliance, the legal corpus, and evals. Each one explains itself at
the top.

## How well it works

- **Compliance checker:** high-80s to ~90% exact-match accuracy on a 4-level verdict
  scale (fully compliant / minor note / grey area / not compliant), holding up on
  both the labeled set and an untouched holdout set.
- **RAG answers:** fully faithful to retrieved sources — it doesn't make things up,
  and says so explicitly when it can't find an answer. Correctness is lower and
  traced to retrieval (not generation) as the corpus grows — a known limitation,
  not a fixed one.
- **Security:** red-teamed against prompt leakage, jailbreaking, and prompt
  injection — including a real exploit found and fixed during testing, where text
  embedded in a video's title or transcript could manipulate a compliance verdict
  or the chat agent's answer. Untrusted content is now delimited, scanned, and
  backed by a deterministic warning to the user whenever something looks off,
  rather than trusting the model to catch it on its own.
- **Cost:** under a cent across all testing so far. A full agent turn (question →
  tool use → answer) costs about $0.0005 and takes 8-9 seconds.

Full numbers and methodology in `data/eval/SUMMARY.md`, `data/eval/RAG_SUMMARY.md`,
`data/eval/SECURITY_SUMMARY.md`, and `data/metrics/SUMMARY.md`.

## More detail

See [docs/architecture.md](docs/architecture.md) for the system design and
[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) for how it got built.

## Developer

Amer Baniodeh
