# Architecture overview

## System summary

The Creative Intelligence Copilot is a RAG-based agent that helps influencer 
marketing strategists at prescription skincare brands analyse content and catch 
compliance issues before campaigns launch.

## Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector DB | Pinecone (free tier, cosine metric) |
| Orchestration | LangChain `AgentExecutor` + `ConversationBufferMemory` |
| Tracing | LangSmith |
| UI | Streamlit |
| Transcript source | `youtube-transcript-api` |

## Data flow

```
YouTube URL
    │
    ▼
youtube-transcript-api       ← pulls auto-captions (no audio processing needed)
    │
    ▼
RecursiveCharacterTextSplitter   ← chunk_size=500, overlap=50
    │
    ▼
text-embedding-3-small           ← 1536-dim vectors
    │
    ▼
Pinecone upsert                  ← with metadata: video_id, title, url, chunk_index
    │
    ▼
                    [User asks a question]
                            │
                            ▼
                    LangChain Agent
                    (OpenAI Functions)
                    /       |       \
                   /        |        \
          ingest_video  query_corpus  check_compliance
                            │
                            ▼
                    Pinecone similarity search (top-5)
                            │
                            ▼
                    GPT-4o-mini RAG chain
                            │
                            ▼
                    Streamlit UI response
```

## Agent tools

### `ingest_video`
- Input: YouTube URL (+ optional title)
- Pipeline: URL → video_id → transcript → chunks → embeddings → Pinecone upsert
- Output: summary string (title, chunk count, vectors upserted)

### `query_corpus`
- Input: natural language question
- Pipeline: embed query → Pinecone similarity search → return top-5 chunks → GPT synthesis
- Output: grounded answer with source attribution

### `check_compliance`
- Input: any text
- Layer 1: regex match against 30-phrase forbidden blocklist (EU Directive 2001/83/EC)
- Layer 2: GPT-4o-mini classifier for edge cases
- Output: compliant/non-compliant verdict with flagged phrases

## Notebooks

| Notebook | What it tests |
|---|---|
| `01_ingestion.ipynb` | URL → transcript → chunks → Pinecone upsert |
| `02_rag_chain.ipynb` | Query → retrieval → GPT answer |
| `03_compliance.ipynb` | Blocklist + LLM classifier on real transcripts |
| `04_langsmith_eval.ipynb` | Evaluation dataset + correctness/faithfulness scores |

## Phase 2 additions (post-MVP)

- Whisper transcription for videos without auto-captions
- Pinecone namespaces for multi-brand / multi-campaign isolation
- Brief generator tool (structured output → creator brief draft)
- Corpus-wide pattern analysis (compare hooks across 20+ videos)
- Multi-language support (DE, FR influencer content)
