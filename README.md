# CV-based-RAG-System

RAG over CV PDFs: PyMuPDF text extraction → semantic chunking → FAISS vector
store → reranked retrieval → LLM answer, served with Streamlit.

Every provider is swappable through environment variables, and every default
has a free, no-credit-card tier.

## Setup

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
pip install -r requirements.txt
cp .env.example .env               # then paste your keys in
python check_providers.py          # verifies keys, models and the FAISS index
streamlit run main.py
```

Rebuild the vector store after changing `pdf/` or the embedding model:

```bash
python langchain-document.py
```

## Free API keys

| Role | Provider | Free tier | Get a key |
| --- | --- | --- | --- |
| LLM (default) | Google Gemini `gemini-2.5-flash` | ~15 req/min, ~1.5k req/day | https://aistudio.google.com/apikey |
| LLM (alternative) | Groq `llama-3.3-70b-versatile` | 30 req/min, ~1k req/day | https://console.groq.com/keys |
| Embeddings + rerank | Jina AI | trial token pool | https://jina.ai/api-dashboard/ |
| Embeddings (no key) | `all-MiniLM-L6-v2`, runs locally | unlimited | — |

Free-tier limits move around; the numbers above are a starting point, not a
contract.

## Configuration

Set these in `.env`:

| Variable | Values | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini`, `groq` | `gemini` |
| `EMBED_PROVIDER` | `jina`, `local` | `jina` |
| `RERANK_PROVIDER` | `jina`, `none` | `jina` |
| `LLM_MODEL` | any model id for the chosen provider | per-provider default |
| `EMBED_MODEL` | any model id for the chosen provider | per-provider default |

### Switching to Groq

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=...
```

### Running with no embedding key at all

`EMBED_PROVIDER=local` uses sentence-transformers on the CPU, so Groq is then
the only key you need. This is the setup the committed index is built for:
MiniLM at 384 dimensions, no reranker.

Switching embedding models always means rebuilding the index, because the
dimensions have to match:

```bash
# .env: EMBED_PROVIDER=local  and  RERANK_PROVIDER=none
python langchain-document.py
```

## Files

| File | Purpose |
| --- | --- |
| `providers.py` | Builds the LLM, embeddings and reranker from env vars |
| `langchain-document.py` | Indexer: PDFs → semantic chunks → `faiss_index/` |
| `main.py` | Streamlit app and the `query_rag` retrieval + answer loop |
| `check_providers.py` | Smoke test for keys, models and the index |
