"""Pluggable free-tier providers for the LLM, embeddings and reranker.

Everything here is selectable with environment variables so the RAG pipeline
never hard-codes a vendor:

    LLM_PROVIDER    = gemini | groq          (default: gemini)
    EMBED_PROVIDER  = jina   | local         (default: jina)
    RERANK_PROVIDER = jina   | none          (default: jina)

Both default providers have a no-credit-card free tier. `local` embeddings run
on the CPU with sentence-transformers and need no API key at all.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Fix OpenMP conflict warning (faiss + torch in the same process)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Model defaults per provider, overridable with LLM_MODEL / EMBED_MODEL.
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}
DEFAULT_EMBED_MODELS = {
    "jina": "jina-embeddings-v2-base-en",
    "local": "sentence-transformers/all-MiniLM-L6-v2",
}


def _secret(name):
    """Read a Streamlit Cloud secret, if we happen to be running under Streamlit.

    Streamlit Community Cloud normally exports secrets as environment variables
    too, but reading st.secrets directly makes the app work either way — and
    keeps this module importable from plain scripts, where streamlit may not be
    initialised.
    """
    try:
        import streamlit as st

        return st.secrets[name]
    except Exception:
        return None


def _env(name, default=None):
    value = os.getenv(name) or _secret(name)
    return value if value else default


def _require(key_name):
    key = _env(key_name)
    if not key:
        raise RuntimeError(
            f"{key_name} is not set. Copy .env.example to .env and fill it in "
            f"(or add it to the app's Streamlit secrets)."
        )
    return key


def get_llm():
    """Return a chat model for the provider named by LLM_PROVIDER."""
    provider = _env("LLM_PROVIDER", "gemini").lower()
    model = _env("LLM_MODEL", DEFAULT_LLM_MODELS.get(provider))

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_retries=2,
            google_api_key=_require("GOOGLE_API_KEY"),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=0,
            max_retries=2,
            api_key=_require("GROQ_API_KEY"),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}. Use one of: "
        f"{', '.join(DEFAULT_LLM_MODELS)}."
    )


def get_embeddings():
    """Return an embeddings model for the provider named by EMBED_PROVIDER.

    The FAISS index is tied to whichever model built it, so changing this means
    re-running langchain-document.py to rebuild faiss_index/.
    """
    provider = _env("EMBED_PROVIDER", "jina").lower()
    model = _env("EMBED_MODEL", DEFAULT_EMBED_MODELS.get(provider))

    if provider == "jina":
        from langchain_community.embeddings import JinaEmbeddings

        return JinaEmbeddings(
            jina_api_key=_require("JINA_API_KEY"),
            model_name=model,
        )

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "EMBED_PROVIDER=local needs the optional extras: "
                "pip install langchain-huggingface sentence-transformers"
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(
        f"Unknown EMBED_PROVIDER {provider!r}. Use one of: "
        f"{', '.join(DEFAULT_EMBED_MODELS)}."
    )


def get_reranker(top_n=3):
    """Return a document compressor, or None when reranking is disabled."""
    provider = _env("RERANK_PROVIDER", "jina").lower()

    if provider in ("none", "off", "false"):
        return None

    if provider == "jina":
        from langchain_community.document_compressors import JinaRerank

        return JinaRerank(jina_api_key=_require("JINA_API_KEY"), top_n=top_n)

    raise ValueError(f"Unknown RERANK_PROVIDER {provider!r}. Use jina or none.")


def describe():
    """Human-readable summary of the active configuration."""
    llm_provider = _env("LLM_PROVIDER", "gemini").lower()
    embed_provider = _env("EMBED_PROVIDER", "jina").lower()
    return {
        "llm": f"{llm_provider}:{_env('LLM_MODEL', DEFAULT_LLM_MODELS.get(llm_provider))}",
        "embeddings": f"{embed_provider}:{_env('EMBED_MODEL', DEFAULT_EMBED_MODELS.get(embed_provider))}",
        "reranker": _env("RERANK_PROVIDER", "jina").lower(),
    }
