import streamlit as st
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS

import providers

embeddings = providers.get_embeddings()

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = providers.get_llm()


def query_rag(question, vectorstore=vector_store, model=llm, k=10):
    """Query function for RAG"""
    # Retrieve relevant documents, reranked when a reranker is configured
    compressor = providers.get_reranker(top_n=3)
    if compressor is None:
        docs = vectorstore.similarity_search(question, k=k)
    else:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": k}))
        # get_relevant_documents(question) does not work in new versions
        docs = compression_retriever.invoke(question)

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create messages for LLM
    messages = [
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context. The context is a document from CV of some individuals."
            "If you cannot answer based on the context, say so clearly."
            "Give answers based on context only."
        ),
        (
            "human",
            f"""Context:
{context}

Question: {question}"""
        )
    ]

    # Generate answer using invoke
    ai_msg = model.invoke(messages)

    return ai_msg.content, docs

# Streamlit UI
st.set_page_config(page_title="RAG System for ByteCorp Assignment", page_icon="🔍")
st.title("RAG System for ByteCorp Assignment")

active = providers.describe()
st.caption(
    f"LLM `{active['llm']}` · embeddings `{active['embeddings']}` · rerank `{active['reranker']}`"
)

# Main interface
question = st.text_input("Ask a question:", placeholder="Enter your question here...")

if question:
    with st.spinner("Generating answer..."):
        answer, docs = query_rag(question, vector_store, k=10)

        st.subheader("Answer:")
        st.write(answer)

        with st.expander("📚 Sources"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content[:300] + "...")
                st.divider()
