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

Work through them in order: `01_ingestion` → `02_rag_chain` → `03_compliance` → `04_langsmith_eval`.

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

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (secrets are gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `app/app.py`
3. Add your secrets under **Advanced settings → Secrets** (copy from `secrets.toml.example`)
4. Deploy
