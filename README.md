# Creative Intelligence Copilot

A RAG-based AI agent that helps influencer marketing strategists at prescription
skincare brands analyse creator content and catch compliance issues before campaigns
go live.

Built with LangChain, Pinecone, OpenAI, and Streamlit.

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/YOUR_USERNAME/copilot.git
cd copilot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

Required keys:
- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com/api-keys)
- `PINECONE_API_KEY` — [app.pinecone.io](https://app.pinecone.io)
- `LANGCHAIN_API_KEY` — [smith.langchain.com](https://smith.langchain.com)

### 3. Create your Pinecone index

In the Pinecone console, create an index with:
- **Name:** `copilot-mvp` (or whatever you set in `PINECONE_INDEX_NAME`)
- **Dimensions:** `1536`
- **Metric:** `cosine`
- **Region:** `us-east-1`

### 4. Run the notebooks (Day 1–3)

```bash
jupyter notebook notebooks/
```

Work through them in order: `01_ingestion` → `02_rag_chain` → `03_compliance` → `04_langsmith_eval` → `05_legal_rag`.

`05_legal_rag` ingests real legal source text (see below) into a separate Pinecone
namespace so compliance verdicts cite actual law instead of an LLM's unverified
general knowledge. Run it at least once before relying on cited-section output.

### 5. Run the Streamlit app

```bash
streamlit run app/app.py
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
copilot/
├── notebooks/               # Jupyter notebooks (one per pipeline layer)
│   ├── 01_ingestion.ipynb
│   ├── 02_rag_chain.ipynb
│   ├── 03_compliance.ipynb
│   └── 04_langsmith_eval.ipynb
├── src/
│   ├── ingestion/
│   │   ├── transcript.py    # YouTube → transcript → chunks
│   │   └── embedder.py      # Embed chunks → Pinecone upsert
│   ├── compliance/
│   │   └── checker.py       # Blocklist + LLM compliance checker
│   ├── agent/
│   │   ├── tools.py         # Three LangChain Tool objects
│   │   └── agent.py         # AgentExecutor with memory
│   └── utils/
│       ├── config.py        # Central env var loader
│       └── logger.py        # Shared logger
├── app/
│   ├── app.py               # Streamlit UI
│   └── .streamlit/
│       └── secrets.toml.example
├── data/
│   ├── raw/                 # Raw transcript JSON (gitignored)
│   └── processed/           # Chunked data (gitignored)
├── tests/
│   └── test_config.py
├── docs/
│   └── architecture.md
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

## Legal corpus (RAG-grounded compliance)

`check_compliance` used to fall back on an LLM classifier reasoning from general
training knowledge of "EU law" — plausible-sounding but not citable or verifiable.
It now retrieves real legal text from a separate Pinecone namespace (`eu-regulations`)
and grounds every non-blocklist verdict in an actual cited provision (e.g. `§3 HWG`).

**Currently ingested** (`data/legal/`, sourced from gesetze-im-internet.de):
- HWG (Heilmittelwerbegesetz) — full text, all sections
- §5a UWG (Irreführung durch Unterlassen)
- §3a UWG (Rechtsbruch)

**TODO — not yet ingested:**
- [ ] EU Cosmetics Regulation 1223/2009, Article 20 (Product claims) — EUR-Lex
      blocked automated fetches; needs a manual pull or an alternate mirror.
- [ ] Cologne 2025 ruling summary — influencer/compliance case law referenced
      as relevant but not yet sourced.

To add a document: drop a `.txt` file into `data/legal/` (preserve `§ N` section
markers on their own line for German-style laws), then add an entry to the
`documents` list in notebook `05_legal_rag.ipynb` and re-run it.

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (secrets are gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `app/app.py`
3. Add your secrets under **Advanced settings → Secrets** (copy from `secrets.toml.example`)
4. Deploy
