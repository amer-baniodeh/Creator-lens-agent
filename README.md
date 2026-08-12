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
gives you a summary. You can also just chat with it and ask questions about videos
already in the knowledge base.

YouTube blocks cloud servers from fetching videos, so live ingestion only works
locally (or through an ngrok tunnel). On the cloud-hosted version, use the
"paste transcript" option instead.

## Notebooks

`notebooks/` walks through the pipeline step by step, in order — ingestion, the
RAG chain, compliance, the legal corpus, and evals. Each one explains itself at
the top.

## How well it works

- **Compliance checker:** ~90% exact-match accuracy on a 4-level verdict scale
  (fully compliant / minor note / grey area / not compliant), holding up on both
  the labeled set and an untouched holdout set.
- **RAG answers:** fully faithful to retrieved sources — it doesn't make things up,
  and says so explicitly when it can't find an answer. Correctness is lower and
  traced to retrieval (not generation) as the corpus grows — a known limitation,
  not a fixed one.
- **Cost:** under a cent across all testing so far. A full agent turn (question →
  tool use → answer) costs about $0.0005 and takes 8-9 seconds.

Full numbers and methodology in `data/eval/SUMMARY.md`, `data/eval/RAG_SUMMARY.md`,
and `data/metrics/SUMMARY.md`.

## More detail

See [docs/architecture.md](docs/architecture.md) for the system design and
[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) for how it got built.
