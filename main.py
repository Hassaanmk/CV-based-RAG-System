"""Streamlit front-end for the CV RAG pipeline.

Loads the FAISS index that langchain-document.py built, then answers questions
against it. The LLM, embeddings and reranker all come from providers.py, so
swap vendors with env vars — no code changes needed.

    streamlit run main.py
"""

import html
import os

import streamlit as st
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS

import providers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
PDF_DIR = os.path.join(BASE_DIR, "pdf")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "The context is a document from CV of some individuals. "
    "If you cannot answer based on the context, say so clearly. "
    "Give answers based on context only."
)

SAMPLE_QUESTIONS = [
    "What programming languages does the candidate know?",
    "Summarise the work experience.",
    "What is the highest level of education?",
    "Which projects involve machine learning?",
]

st.set_page_config(
    page_title="CV Intelligence",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Styling only — no logic here. Kept in one block so the layout is easy to tweak.
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}

      .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 2rem 2.25rem;
        border-radius: 18px;
        margin-bottom: 1.75rem;
        color: #fff;
      }
      .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.2;
        color: #fff;
      }
      .hero p {
        margin: 0.5rem 0 0;
        font-size: 0.97rem;
        opacity: 0.92;
      }

      .source-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 3px solid #7c3aed;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.86rem;
        line-height: 1.5;
      }
      .source-head {
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.75;
        margin-bottom: 0.4rem;
      }

      div[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Build the retrieval stack once per server process, not once per rerun."""
    # Same embedding model that built the index — mismatch = garbage retrieval.
    embeddings = providers.get_embeddings()
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store, providers.get_llm(), providers.get_reranker(top_n=3)


def query_rag(question, vectorstore, model, compressor, k=10):
    """Pull the best chunks for a question, then ask the LLM to answer from them."""
    # Prefer reranking when it's configured; otherwise plain similarity search.
    if compressor is None:
        docs = vectorstore.similarity_search(question, k=k)
    else:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        )
        # LangChain dropped get_relevant_documents(); invoke() is the current API.
        docs = compression_retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    ai_msg = model.invoke(messages)
    return ai_msg.content, docs


def render_sources(docs):
    """Show the retrieved chunks behind an answer, newest answer expanded."""
    with st.expander(f"Sources ({len(docs)})"):
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "document")
            score = doc.metadata.get("relevance_score")
            label = f"{i} · {source}"
            if score is not None:
                label += f" · match {score:.0%}"
            snippet = doc.page_content.strip()
            if len(snippet) > 700:
                snippet = snippet[:700].rstrip() + "…"
            # CV text is untrusted input for this HTML block — escape it, and
            # keep the original line breaks readable.
            snippet = html.escape(snippet).replace("\n", "<br>")
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-head">{html.escape(label)}</div>'
                f"{snippet}</div>",
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="hero">
      <h1>📄 CV Intelligence</h1>
      <p>Ask anything about the indexed résumés — answers are grounded in the
      documents themselves, with the exact passages shown as sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

try:
    vector_store, llm, reranker = load_pipeline()
except Exception as exc:
    st.error("The app could not start.")
    st.caption(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Indexed documents")
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    for name in pdfs:
        st.markdown(f"- {os.path.splitext(name)[0]}")
    st.caption(f"{vector_store.index.ntotal} searchable passages")

    st.divider()
    st.subheader("Try asking")
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample_{i}"):
            st.session_state.pending = sample

    st.divider()
    if st.button("Clear conversation", key="clear"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("docs"):
            render_sources(message["docs"])

typed = st.chat_input("Ask a question about the CVs…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                answer, docs = query_rag(question, vector_store, llm, reranker)
            except Exception as exc:
                st.error("Could not generate an answer. Please try again.")
                st.caption(str(exc))
                st.stop()
        st.markdown(answer)
        render_sources(docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "docs": docs}
    )
