"""Smoke-test the configured providers before running the Streamlit app.

    python check_providers.py

Each check is independent, so a missing key for a provider you are not using
only fails that one line.
"""

import os

import providers

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")


def check(label, fn):
    try:
        detail = fn()
    except Exception as exc:
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"OK    {label}: {detail}")
    return True


def check_llm():
    llm = providers.get_llm()
    reply = llm.invoke([("human", "Reply with the single word: pong")])
    return repr(reply.content.strip()[:40])


def check_embeddings():
    emb = providers.get_embeddings()
    vector = emb.embed_query("curriculum vitae")
    return f"{len(vector)} dimensions"


def check_reranker():
    from langchain_core.documents import Document

    reranker = providers.get_reranker(top_n=1)
    if reranker is None:
        return "disabled (RERANK_PROVIDER=none)"
    ranked = reranker.compress_documents(
        [Document(page_content="Python engineer"), Document(page_content="Chef")],
        query="software developer",
    )
    return f"top hit {ranked[0].page_content!r}"


def check_index():
    from langchain_community.vectorstores import FAISS

    store = FAISS.load_local(
        INDEX_DIR, providers.get_embeddings(), allow_dangerous_deserialization=True
    )
    dim = store.index.d
    return f"{store.index.ntotal} chunks, {dim} dimensions"


if __name__ == "__main__":
    active = providers.describe()
    print(f"config: {active}\n")

    results = [
        check("llm", check_llm),
        check("embeddings", check_embeddings),
        check("reranker", check_reranker),
        check("faiss index", check_index),
    ]

    print()
    if all(results):
        print("All checks passed. Run: streamlit run main.py")
    else:
        raise SystemExit("Some checks failed — see above.")
