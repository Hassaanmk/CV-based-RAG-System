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
| `LLM_PROVIDER` | `gemini`, `groq` | `gemini` (repo uses `groq`) |
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
the only key you need. It is not part of the default install, because it pulls
in torch (~2.5 GB with the CUDA wheels) and does not fit Streamlit Community
Cloud's build or memory limits:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install langchain-huggingface==1.2.2 sentence-transformers==5.6.1 torchvision==0.28.0
```

Switching embedding models always means rebuilding the index, because the
dimensions have to match:

```bash
# .env: EMBED_PROVIDER=local
python langchain-document.py
```

## Deploying to Streamlit Community Cloud

The committed `faiss_index/` is built with Jina embeddings at 768 dimensions,
which is what the deployed app expects — no torch, no model download at boot.

1. Push the repo to GitHub (`faiss_index/` and `pdf/` must be committed).
2. Create the app at https://share.streamlit.io, main file `main.py`.
3. Paste the contents of `.streamlit/secrets.toml.example` into the app's
   **Secrets** box, with your real keys.

`providers.py` reads configuration from environment variables first and falls
back to `st.secrets`, so the same code runs locally off `.env` and on Cloud off
secrets. `.streamlit/config.toml` sets the theme and hides Streamlit's error
details from visitors.

## Files

| File | Purpose |
| --- | --- |
| `providers.py` | Builds the LLM, embeddings and reranker from env vars |
| `langchain-document.py` | Indexer: PDFs → semantic chunks → `faiss_index/` |
| `main.py` | Streamlit app and the `query_rag` retrieval + answer loop |
| `check_providers.py` | Smoke test for keys, models and the index |
